# -*- coding: utf-8 -*-
from __future__ import division

import torch
import torch.nn as nn

from modules.utils import aeq, rnn_factory
from modules.global_attention import GlobalAttention


class DecoderState(object):
    """Interface for grouping together the current decoder_state of a recurrent
    decoder. In the simplest case just represents the hidden decoder_state of
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
        hidden, hidden decoder_state, format the hidden decoder_state of RNN, LSTM, GRU into a tuple.
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

    def __init__(self, hidden_size, decoder_state):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            decoder_state: final hidden decoder_state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(decoder_state, tuple):
            self.hidden = (decoder_state,)
        else:
            self.hidden = decoder_state

        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)

        self.input_feed = self.hidden[0].data.new(*h_size).zero_().unsqueeze(0)
        self.input_feed.requires_grad = False

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, decoder_state, input_feed=None, coverage=None):
        if not isinstance(decoder_state, tuple):
            self.hidden = (decoder_state,)
        else:
            self.hidden = decoder_state

        if input_feed is not None:
            self.input_feed = input_feed

        if coverage is not None:
            self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        #  vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                #  for e in self._all]
        vars = [e.data.repeat(1, beam_size, 1) for e in self._all]
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
          G[Decoder decoder_state]
          H[Decoder decoder_state]
          I[Outputs]
          F[encoder_outputs]
          A--embedded-->C
          A--embedded-->D
          A--embedded-->E
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
                 bidirectional_encoder,
                 num_layers,
                 hidden_size,
                 attn_type=None,
                 dropout=0.0,
                 embedding=None):

        super(DecoderBase, self).__init__()

        assert embedding is not None

        # Basic attributes.
        self.decodes_type = 'rnn'
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
            self.attn = GlobalAttention(self.hidden_size, attn_type=self.attn_type)

    def init_decoder_state(self, encoder_final):
        # def init_decoder_state(self, src, encoder_outputs, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                #  h = torch.cat([h[decoder_output:h.size(decoder_output):2], h[1:h.size(decoder_output):2]], 2)
                h = torch.cat((h[: h.shape[0]//2], h[h.shape[0]//2: ]), dim=2) # [num_layers, batch_size, hidden_size]
            return h

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size, tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final]))
        else:  # GRU RNN
            return RNNDecoderState(self.hidden_size, _fix_enc_hidden(encoder_final))

    # apply to encoder outputs to get weighted average
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embedding.embedding_size

    def _check_args(self, inputs, encoder_outputs, decoder_state):
        assert isinstance(decoder_state, RNNDecoderState)
        inputs_len, tgt_batch = inputs.size()
        _, memory_batch, _ = encoder_outputs.size()
        aeq(tgt_batch, memory_batch)

    def forward(self, inputs, encoder_outputs, decoder_state, encoder_inputs_length=None):
        """
        Args:
            inputs (`LongTensor`): sequences of padded tokens
                                `[inputs_len x batch]`.
            encoder_outputs (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            decoder_state (:obj:`DecoderState`):
                 decoder decoder_state object to initialize the decoder
            encoder_inputs_length (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[inputs_len x batch x hidden]`.
                * decoder_state: final hidden decoder_state from the decoder
                * attns: a dict, keys:{'std', 'converage', ...},
                        distribution over src at each inputs
                        `[inputs_len x batch x src_len]`.
        """

        # Check
        self._check_args(inputs, encoder_outputs, decoder_state)

        # Run the forward pass of the RNN.
        decoder_final, decoder_output, attns = self._run_forward_pass(
            inputs, encoder_outputs, decoder_state, encoder_inputs_length)

        # Update the decoder_state with the result.
        final_output = decoder_output[-1]

        decoder_state.update_state(decoder_final, final_output.unsqueeze(0))

        return decoder_state, decoder_output, attns


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

    def _run_forward_pass(self, inputs, encoder_outputs, decoder_state, encoder_inputs_length=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overrided by all subclasses.
        Args:
            inputs (LongTensor): a sequence of input tokens tensors
                                 [inputs_len x batch].
            encoder_outputs (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            decoder_state (FloatTensor): hidden decoder_state from the encoder RNN for
                                 initializing the decoder.
            encoder_inputs_length (LongTensor): the source encoder_outputs lengths.
        Returns:
            decoder_final (Variable): final hidden decoder_state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """

        # Initialize local and return variables.
        attns = {}

        embedded = self.embedding(inputs)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            decoder_output, decoder_final = self.rnn(embedded, decoder_state.hidden[0])
        else:
            # LSTM
            decoder_output, decoder_final = self.rnn(embedded, decoder_state.hidden)

        # Check
        inputs_len, tgt_batch = inputs.size()
        output_len, output_batch, _ = decoder_output.size()

        aeq(inputs_len, output_len)
        aeq(tgt_batch, output_batch)

        # Calculate the attention.
        if self.attn_type is not None:
            # attention forward
            #  decoder_output, p_attn = self.attn(
                #  decoder_output.transpose(0, 1),
                #  encoder_outputs.transpose(0, 1))
            # decoder_output -> [1, batch_size, hidden_size], encoder_outputs ->
            # [1, batch_size, hidden_sizes] -> [batch_size, 1, hidden_size]
            decoder_output, p_attn = self.attn(decoder_output.transpose(0, 1),
                                               encoder_outputs.transpose(0, 1),
                                               encoder_inputs_length)
            attns["std"] = p_attn
        else:
            decoder_output = decoder_output

        # dropout
        decoder_output = self.dropout(decoder_output)

        return decoder_final, decoder_output, attns


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
          H[encoder_outputs n-1]
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

    def _run_forward_pass(self, inputs, encoder_outputs, decoder_state, encoder_inputs_length=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        # lack of schedule sampling forward operation ???????
        """
        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}

        embedded = self.embedding(inputs)
        assert embedded.dim() == 3  # inputs_len x batch x embedding_dim

        hidden = decoder_state.hidden
        # Input feed concatenates hidden decoder_state with
        # input at every time step.
        memory_bank_t = encoder_outputs.transpose(0, 1)

        for i, embedded_t in enumerate(embedded.split(1)):
            embedded_t = embedded_t.squeeze(0)
            decoder_input = torch.cat((embedded_t, input_feed), dim=1)
            decoder_output, hidden, p_attn, input_feed = self._run_forward_one(
                decoder_input, memory_bank_t, hidden, encoder_inputs_length=encoder_inputs_length)
            # input_feed = decoder_output
            decoder_outputs += [decoder_output]
            if p_attn is not None:
                attns["std"] += [p_attn]
        # Return result.
        return hidden, decoder_outputs, attns

    def _run_forward_one(self, decoder_input, memory_bank_t, hidden, encoder_inputs_length=None):
        decoder_output, hidden = self.rnn(decoder_input, hidden)
        if self.attn_type is not None:
            decoder_output, p_attn = self.attn(
                decoder_output,
                memory_bank_t, encoder_inputs_length=encoder_inputs_length)
        else:
            decoder_output, p_attn = (decoder_output, None)

        decoder_output = self.dropout(decoder_output)
        return decoder_output, hidden, p_attn, decoder_output


if __name__ == '__main__':
    pass
