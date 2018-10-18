# -*- coding: utf-8 -*-
from __future__ import division

import torch
import torch.nn as nn

from modules.utils import aeq, sequence_mask


class Bottle(nn.Module):
    def forward(self, decoder_output):
        if len(decoder_output.size()) <= 2:
            return super(Bottle, self).forward(decoder_output)
        size = decoder_output.size()[:2]
        out = super(Bottle, self).forward(decoder_output.view(size[0] * size[1], -1))
        return out.contiguous().view(size[0], size[1], -1)


class Bottle2(nn.Module):
    def forward(self, decoder_output):
        if len(decoder_output.size()) <= 3:
            return super(Bottle2, self).forward(decoder_output)
        size = decoder_output.size()
        out = super(Bottle2, self).forward(decoder_output.view(size[0] * size[1],
                                                       size[2], size[3]))
        return out.contiguous().view(size[0], size[1], size[2], size[3])


class BottleLinear(Bottle, nn.Linear):
    pass


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the decoder_output query.

    Constructs a unit mapping a query `q` of size `hidden_size`
    and a source matrix `H` of size `n x hidden_size`, to an output
    of size `hidden_size`.
    .. mermaid::
       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G
    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].
    However they
    differ on how they compute the attention score.
    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
    Args:
       hidden_size (int): dimensionality of query and key
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """

    def __init__(self, hidden_size, attn_type="dot"):
        super(GlobalAttention, self).__init__()

        self.hidden_size = hidden_size
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), \
            ("Please select a valid attention type.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = BottleLinear(
                hidden_size, hidden_size, bias=False)
            self.linear_query = nn.Linear(hidden_size, hidden_size, bias=True)
            self.v = BottleLinear(hidden_size, 1, bias=False)

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=out_bias)

        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()

    def score(self, decoder_output, encoder_outputs):
        """
        Args:
          decoder_output (`FloatTensor`): sequence of queries `[batch_sizse x tgt_len x hidden_size]`
          encoder_outputs (`FloatTensor`): sequence of sources `[batch_sizse x src_len x hidden_size]`
        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch_sizse x tgt_len x src_len]`
        """

        # Check decoder_output sizes
        src_batch, src_len, src_dim = encoder_outputs.size()
        tgt_batch, tgt_len, tgt_dim = decoder_output.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.hidden_size, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                #  h_t_ = decoder_output.view(tgt_batch * tgt_len, tgt_dim)
                #  h_t_ = self.linear_in(h_t_)
                #  decoder_output = h_t_.view(tgt_batch, tgt_len, tgt_dim)
                decoder_output = self.linear_in(decoder_output)

            # (batch_sizse, t_len, d) x (batch_sizse, d, s_len) --> (batch_sizse, t_len, s_len)
            # [batch_sizse, t_len, s_len]
            return torch.bmm(decoder_output, encoder_outputs.transpose(1, 2))
        else:
            hidden_size = self.hidden_size
            wq = self.linear_query(decoder_output.view(-1, hidden_size))
            wq = wq.view(tgt_batch, tgt_len, 1, hidden_size)
            wq = wq.expand(tgt_batch, tgt_len, src_len, hidden_size)

            uh = self.linear_context(
                encoder_outputs.contiguous().view(-1, hidden_size))
            uh = uh.view(src_batch, 1, src_len, hidden_size)
            uh = uh.expand(src_batch, tgt_len, src_len, hidden_size)

            # (batch_sizse, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, hidden_size)).view(tgt_batch, tgt_len, src_len)

    def forward(self, decoder_output, encoder_outputs, encoder_inputs_length=None):
        """
        Args:
          decoder_output (`FloatTensor`): query vectors `[batch_sizse x tgt_len x hidden_size]`
          memory_bank (`FloatTensor`): source vectors `[batch_sizse x src_len x hidden_size]`
          encoder_inputs_length (`LongTensor`): the source context lengths `[batch_size]`
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[tgt_len x batch_sizse x hidden_size]`
          * Attention distribtutions for each query
             `[tgt_len x batch_sizse x src_len]`
        """

        # one step decoder_output
        if decoder_output.dim() == 2:
            one_step = True
            # insert one dimension
            decoder_output = decoder_output.unsqueeze(1)
        else:
            one_step = False

        batch_sizse, sourceL, hidden_size = encoder_outputs.size()
        batch_size_, targetL, hidden_sizse_ = decoder_output.size()

        aeq(batch_sizse, batch_size_)
        aeq(hidden_size, hidden_sizse_)
        aeq(self.hidden_size, hidden_size)

        # compute attention scores, as in Luong et al.
        align = self.score(decoder_output, encoder_outputs) #[batch_size, t_len, s_len]

        if encoder_inputs_length is not None:
            # obtain mask for memory_lenghts
            mask = sequence_mask(encoder_inputs_length)
            mask = mask.to(device=encoder_outputs.device)

            mask = mask.unsqueeze(1)  # Make it broadcastable.

            # Fills elements of self tensor with value where mask is one. masked_fill_(mask, value)
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.softmax(align) #

        # each context vector c_t is the weighted average
        # over all the source hidden states
        context_vecotr = torch.bmm(align_vectors, encoder_outputs) #[batch_size, t_len , hidden_size]

        # concatenate
        concated_cv = torch.cat((context_vecotr, decoder_output), dim=2) #[batch_size, t_len, 2*hidden_size]
        attn_h = self.linear_out(concated_cv) #[batch_size, t_len, hidden_size]

        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h) # tanh activation

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_size_, hidden_sizse_ = attn_h.size()
            aeq(batch_sizse, batch_size_)
            aeq(hidden_size, hidden_sizse_)
            batch_size_, sourceL_ = align_vectors.size()
            aeq(batch_sizse, batch_size_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous() # [t_len, batch_size, hidden_size]
            align_vectors = align_vectors.transpose(0, 1).contiguous() # [t_len, batch_size, s_len]

            # Check output sizes
            targetL_, batch_size_, hidden_sizse_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch_sizse, batch_size_)
            aeq(hidden_size, hidden_sizse_)
            targetL_, batch_size_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch_sizse, batch_size_)
            aeq(sourceL, sourceL_)

        return attn_h, align_vectors

