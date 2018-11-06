#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Encoder based on lstm
"""


import math
import torch
import torch.nn as nn

from modules.utils import init_lstm_wt
from modules.utils import rnn_factory
from modules.utils import init_wt_normal
from modules.utils import init_gru_orth, init_lstm_orth

class SimpleEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding,
                 rnn_type,
                 hidden_size,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):

        super(SimpleEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding.embedding_dim
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirection_num = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.bidirection_num

        # embedding
        self.embedding = embedding

        # dropout
        self.dropout = nn.Dropout(dropout)

        # rnn
        self.rnn = rnn_factory(
            rnn_type,
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

    def forward(self, inputs, hidden_state):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [seq_len, batch_size, num_directions * hidden_size]
            max_output: [1, batch_size, hidden_size * num_directions]
            hidden_state: (h_n, c_n)
        '''
        # embedded
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        # [batch_size, hidden_size]
        outputs, hidden_state = self.rnn(embedded, hidden_state)

        return outputs, hidden_state

    def init_hidden(self, batch_size, device):
        initial_state_scale = math.sqrt(3.0 / self.hidden_size)

        initial_state1 = torch.rand((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)
        nn.init.uniform_(initial_state1, a=-initial_state_scale, b=initial_state_scale)
        if self.rnn_type == 'LSTM':
            initial_state2 = torch.rand((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)
            nn.init.uniform_(initial_state2, a=-initial_state_scale, b=initial_state_scale)
            return (initial_state1, initial_state2)
        else:
            return initial_state1
