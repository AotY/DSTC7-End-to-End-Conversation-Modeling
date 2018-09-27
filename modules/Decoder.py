# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from modules.utils import aeq, rnn_factory
from modules.GlobalAttention import GlobalAttention


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.
    Modules need to implement this to utilize beam search decoding.
    """

    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    """
        Instance Variables:
        hidden, hidden state, format the hidden state of RNN, LSTM, GRU into a tuple.
                RNN & GRU, h_n
                LSTM, (h_n, c_n)
                to (h_n, [c_n], )
        input_feed, store the current output for providing input agumenting for the next time-step.
        Methods:
            update_state,
            repeat_beam_size_times,
            beam_update,
            detach,
    """

    def __init__(self, hidden_size, rnn_state):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnn_state: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnn_state, tuple):
            self.hidden = (rnn_state,)
        else:
            self.hidden = rnn_state

        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)

        h_size = (batch_size, hidden_size)

        self.input_feed = Variable(self.hidden[0].data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnn_state, input_feed, coverage=None):
        if not isinstance(rnn_state, tuple):
            self.hidden = (rnn_state,)
        else:
            self.hidden = rnn_state

        self.input_feed = input_feed

        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]


class DecoderBase(nn.Module):
    """
    Base decoder class. Specifies the interface used by different decoder types
    and required by :obj:`modules.Decoder`.
        .. mermaid::
       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : if attn_type is not None, see GlobalAttention()
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embedding (:obj:`embedding`): embedding module to use
    """

    def __init__(self, rnn_type,
                 bidirectional_encoder, num_layers,
                 hidden_size, attn_type=None,
                 dropout=0.0, embedding=None):

        super(DecoderBase, self).__init__()

        assert embedding is not None

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.attn_type = attn_type

        # Build the RNN.
        self.rnn = rnn_factory(rnn_type,
                               input_size=self._input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout)

        # Set up the standard attention.
        if self.attn_type is not None:
            self.attn = GlobalAttention(
                self.hidden_size, attn_type=self.attn_type)

    def init_decoder_state(self, encoder_final):
        # def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size, tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final]))
        else:  # GRU RNN
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embedding.embedding_size

    def _check_args(self, tgt, memory_bank, state):
        assert isinstance(state, RNNDecoderState)
        tgt_len, tgt_batch = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: a dict, keys:{'std', 'converage', ...},
                        distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Check
        self._check_args(tgt, memory_bank, state)

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)

        # Update the state with the result.
        final_output = decoder_outputs[-1] #
        state.update_state(decoder_final, final_output.unsqueeze(0))

        # print('decoder_final shape: {}'.format(decoder_final.shape))
        # print('decoder_outputs shape: {}'.format(decoder_outputs.shape))
        # print('attns shape: {}'.format(attns.shape))

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack((decoder_outputs, ))

        for k in attns:
            attns[k] = torch.stack((attns[k], ))

        return decoder_outputs, state, attns


class StdRNNDecoder(DecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`DecoderBase` for options.
    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`
    Implemented without input_feeding
    """

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embedding.embedding_size

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overrided by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [tgt_len x batch].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """

        # Initialize local and return variables.
        attns = {}

        embedded = self.embedding(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            rnn_output, decoder_final = self.rnn(embedded, state.hidden[0])
        else:
            # LSTM
            rnn_output, decoder_final = self.rnn(embedded, state.hidden)

        # Check
        tgt_len, tgt_batch = tgt.size()
        output_len, output_batch, _ = rnn_output.size()

        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)

        # END
        print('rnn_output shape: {}'.format(
            rnn_output.shape))  # [50, 128, 512]

        print('decoder_final[0] shape: {}'.format(decoder_final[0].shape))
        print('decoder_final[1] shape: {}'.format(decoder_final[1].shape))

        print('rnn_output.transpose(0, 1).contiguous shape: {}'.format(
            rnn_output.transpose(0, 1).contiguous().shape))  # [128, 50, 512]

        print('memory_bank shape: {}'.format(memory_bank.shape)) #[48, 128, 512]

        print('memory_bank.transpose(0, 1) shape: {}'.format(
            memory_bank.transpose(0, 1).shape))  # [128, 48, 512]

        print('memory_lengths: {}'.format(memory_lengths))

        # Calculate the attention.
        if self.attn_type is not None:
            # attention forward
            decoder_outputs, p_attn = self.attn.forward(
                rnn_output.transpose(0, 1).contiguous(),
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)

            #
            attns["std"] = p_attn
        else:
            decoder_outputs = rnn_output

        # dropout
        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns


class InputFeedRNNDecoder(DecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.
    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`
    .. mermaid::
       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embedding.embedding_size + self.hidden_size

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        # lack of schedule sampling forward operation ???????
        """
        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}

        emb = self.embedding(tgt)
        assert emb.dim() == 3  # tgt_len x batch x embedding_dim

        hidden = state.hidden
        # Input feed concatenates hidden state with
        # input at every time step.
        memory_bank_t = memory_bank.transpose(0, 1)

        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)
            decoder_output, hidden, p_attn, input_feed = self._run_forward_one(
                decoder_input, memory_bank_t, hidden, memory_lengths=memory_lengths)
            # input_feed = decoder_output
            decoder_outputs += [decoder_output]
            if p_attn is not None:
                attns["std"] += [p_attn]
        # Return result.
        return hidden, decoder_outputs, attns

    def _run_forward_one(self, decoder_input, memory_bank_t, hidden, memory_lengths=None):
        rnn_output, hidden = self.rnn(decoder_input, hidden)
        if self.attn_type is not None:
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank_t, memory_lengths=memory_lengths)
        else:
            decoder_output, p_attn = (rnn_output, None)

        decoder_output = self.dropout(decoder_output)
        return decoder_output, hidden, p_attn, decoder_output


if __name__ == '__main__':
    pass
