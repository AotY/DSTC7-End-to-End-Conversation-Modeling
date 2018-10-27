#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn

from modules.utils import init_wt_normal, init_linear_wt, init_lstm_orth, init_gru_orth
from modules.global_attn import GlobalAttn
from modules.utils import rnn_factory



class LuongAttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 dropout,
                 padding_idx,
                 turn_type,
                 tied,
                 attn_type,
                 device):

        super(LuongAttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.turn_type = turn_type
        self.attn_type = attn_type

        # embedding
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_size,
                                      padding_idx)
        init_wt_normal(self.embedding.weight)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # attn
        self.attn = GlobalAttn(self.attn_type, self.hidden_size, device)

        self.rnn = rnn_factory(
            rnn_type,
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        if turn_type == 'attention': # history attn
            self.attn_history = GlobalAttn(self.attn_type, self.hidden_size, device)
            self.concat_history_linear = nn.Linear(hidden_size * 3, hidden_size)
            init_linear_wt(self.concat_history_linear)
        else:
            # concat linear
            self.concat_linear = nn.Linear(hidden_size * 2, hidden_size)
            init_linear_wt(self.concat_linear)

        # linear
        self.linear = nn.Linear(hidden_size, vocab_size)
        init_linear_wt(self.linear)

        if tied and embedding_size == hidden_size:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state,  c_encoder_outputs, h_encoder_outputs=None):
        '''
		Args:
			input: [1, batch_size]  LongTensor
			hidden_state: [num_layers, batch_size, hidden_size]
			c_encoder_outputs: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
			h_encoder_outputs: [len, batch_size, hidden_size]
			hidden_state: (h_n, c_n)
        '''

        # embedded
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # Get current hidden state from input word and last hidden state
        # output: [1, batch_size, hidden_size]
        output, hidden_state = self.rnn(embedded, hidden_state)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average<Paste>
        attn_weights = self.attn(output.squeeze(0), c_encoder_outputs)  # [batch_size, max_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, max_len]

        # [batch_size, 1, hidden_size]
        context = torch.bmm(attn_weights, c_encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)  # [1, batch_size, hidden_size]

        if h_encoder_outputs is not None and self.turn_type == 'attention':
            attn_weights_h = self.attn_history(output.squeeze(0), h_encoder_outputs)  # [batch_size, max_len]
            attn_weights_h = attn_weights_h.unsqueeze(1)  # [batch_size, 1, max_len]
            context_h = torch.bmm(attn_weights_h, h_encoder_outputs.transpose(0, 1))
            context_h = context_h.transpose(0, 1)  # [1, batch_size, hidden_size]

            concat_input = torch.cat((context, context_h, output), dim=2)
            concat_output = torch.tanh(self.concat_history_linear(concat_input))

        else:

            # [1, batch_size, hidden_size * 2]
            concat_input = torch.cat((context, output), dim=2)

            # [1, batch_size, hidden_size]
            concat_output = torch.tanh(self.concat_linear(concat_input))

        # linear
        output = self.linear(concat_output)

        # [1, batch_size, hidden_size]
        # softmax
        output = self.softmax(output)

        return output, hidden_state, attn_weights
