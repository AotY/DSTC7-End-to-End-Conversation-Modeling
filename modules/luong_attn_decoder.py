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
#  from modules.global_attn import GlobalAttn
from modules.attention import Attention
from modules.utils import rnn_factory


class LuongAttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 dropout,
                 tied,
                 turn_type,
                 attn_type,
                 device):

        super(LuongAttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attn_type = attn_type

        self.turn_type = turn_type

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        # dropout
        self.dropout = nn.Dropout(dropout)

        # c_attn
        self.c_attn = Attention(hidden_size)

        self.rnn = rnn_factory(
            rnn_type,
            input_size=self.embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        if self.turn_type == 'hred_attn':
            self.h_attn = Attention(hidden_size)
        elif self.turn_type == 'hred':
            self.h_linear = nn.Linear(hidden_size + hidden_size, hidden_size)

        # linear
        self.linear = nn.Linear(hidden_size, vocab_size)

        if tied and self.embedding_size == hidden_size:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self,
                input,
                hidden_state,
                c_encoder_outputs,
                c_encoder_inputs_length,
                h_encoder_outputs=None,
                h_encoder_inputs_length=None):

        '''
		Args:
			input: [1, batch_size]  or [len, batch_size]
			hidden_state: [num_layers, batch_size, hidden_size]
			c_encoder_outputs: [seq_len, batch, hidden_size]
			h_encoder_outputs: [len, batch_size, hidden_size]
			hidden_state: (h_n, c_n)
        '''

        # embedded
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # Get current hidden state from input word and last hidden state
        output, hidden_state = self.rnn(embedded, hidden_state)

        # c attention
        c_attn_output, c_attn_weights = self.c_attn(output, c_encoder_outputs, c_encoder_inputs_length)

        # h attention
        h_attn_output = None
        if h_encoder_outputs is not None:
            if self.turn_type == 'hred_attn':
                h_attn_output, h_attn_weights = self.h_attn(c_attn_output, h_encoder_outputs, h_encoder_inputs_length)
            elif self.turn_type == 'hred':
                #  con_output = output + h_encoder_outputs[-1].unsqueeze(0)
                h_concat_input = torch.cat((c_attn_output, h_encoder_outputs[-1].unsqueeze(0)), dim=2)
                h_concat_output = torch.tanh(self.h_linear(h_concat_input))

        if h_attn_output is not None:
            output = self.linear(h_attn_output)
        else:
            output = self.linear(c_attn_output)

        # softmax
        output = self.softmax(output)

        return output, hidden_state, c_attn_weights
