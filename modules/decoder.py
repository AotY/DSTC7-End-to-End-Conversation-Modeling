#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""


import torch
import torch.nn as nn

from modules.utils import rnn_factory
from modules.utils import init_wt_normal, init_linear_wt, init_lstm_orth, init_gru_orth

'''
Decoder:
    vocab_size:
    embedding_size:
    hidden_size:
    layer_nums:
'''


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 dropout,
                 tied):

        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding.embedding_dim
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # embedding
        self.embedding = embedding

        # dropout
        self.dropout = nn.Dropout(dropout)

        # rnn
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

        # linear
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        init_linear_wt(self.linear)

        if tied and hidden_size == embedding_size:
            self.linear.weight = self.embedding.weight

        self.h_linear = nn.Linear(hidden_size, hidden_size)

        # log softmax
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self,
                input,
                hidden_state,
                c_encoder_outputs=None,
                h_encoder_outputs=None):
        '''
        input: [1, batch_size]  LongTensor
        hidden_state: [num_layers, batch_size, hidden_size]
        c_encoder_outputs: [max_len, batch_size, hidden_size * 2]
        h_encoder_outputs: history encoder outputs, for hred model [1, batch_size, hidden_size]

        output: [1, batch_size, hidden_size]
        hidden_state: (h_n, c_n)
        '''

        # embedded
        embedded = self.embedding(input) #[1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # rnn
        output, hidden_state = self.rnn(embedded, hidden_state)

        if h_encoder_outputs is not None:
            output += h_encoder_outputs
            output = self.h_linear(output)

        # [1, batch_size, hidden_size]
        # linear
        output = self.linear(output)

        # softmax
        output = self.softmax(output)

        return output, hidden_state, None

