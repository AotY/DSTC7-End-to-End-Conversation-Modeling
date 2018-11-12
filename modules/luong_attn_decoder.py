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

        # h_attn
        self.h_attn = Attention(hidden_size)

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

        if self.turn_type == 'weight':
            self.h_attn = Attention(hidden_size)

        # linear
        self.linear = nn.Linear(hidden_size, vocab_size)

        if tied and self.embedding_size == hidden_size:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self,
                input,
                hidden_state,
                h_encoder_outputs=None,
                h_decoder_lengths=None):
        '''
		Args:
			input: [1, batch_size]
			hidden_state: [num_layers, batch_size, hidden_size]

			h_encoder_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            h_decoder_lengths: [batch_size] * turn_num or [batch_size] * max_len
        '''
        #  print(input.shape)
        #  print(hidden_state.shape)

        # embedded
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # Get current hidden state from input word and last hidden state
        output, hidden_state = self.rnn(embedded, hidden_state)

        h_attn_output = None
        h_attn_weights = None
        if h_encoder_outputs is not None and h_decoder_lengths is not None:
            h_attn_output, h_attn_weights = self.h_attn(output, h_encoder_outputs, h_decoder_lengths)

        if h_attn_output is not None:
            output = self.linear(h_attn_output)
        else:
            output = self.linear(output)

        # log softmax
        output = self.softmax(output)

        return output, hidden_state, h_attn_weights
