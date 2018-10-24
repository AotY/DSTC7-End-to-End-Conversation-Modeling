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

class SimpleEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0,
                 padding_idx=0):

        super(SimpleEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.bidirection_num = 2 if bidirectional else 1
        self.num_layers = num_layers

        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, self.padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # LSTM
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=bidirectional)
        init_lstm_wt(self.lstm)

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

        if self.bidirection_num == 2:
            #  [seq, batch_size, hidden_size * 2] -> [seq, batch_size, hidden_size]
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # [batch_size, hidden_size]
        outputs, hidden_state = self.lstm(embedded, hidden_state)

        return outputs, hidden_state

    def init_hidden(self, batch_size, device):
        initial_state_scale = math.sqrt(3.0 / self.hidden_size)

        initial_state1 = torch.rand((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)
        initial_state2 = torch.rand((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)

        nn.init.uniform_(initial_state1, a=-initial_state_scale, b=initial_state_scale)
        nn.init.uniform_(initial_state2, a=-initial_state_scale, b=initial_state_scale)
        return (initial_state1, initial_state2)
