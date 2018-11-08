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

from modules.utils import init_wt_normal
from modules.utils import rnn_factory
from modules.utils import init_lstm_wt, init_gru_orth, init_lstm_orth

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding,
                 rnn_type,
                 hidden_size,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):

        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding.embedding_dim
        self.rnn_type = rnn_type
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
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

    def forward(self, inputs, inputs_length, hidden_state):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [seq_len, batch_size, num_directions * hidden_size]
            max_output: [1, batch_size, hidden_size * num_directions]
            hidden_state: (h_n, c_n)
        '''
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        # [batch_size, seq_len, embedding_size]
        embedded = embedded.transpose(0, 1)

        # sort lengths
        _, sorted_indexes = torch.sort(inputs_length, dim=0, descending=True)
        new_inputs_length = inputs_length[sorted_indexes]

        # restore to original indexes
        _, restore_indexes = torch.sort(sorted_indexes, dim=0)

        # new embedded
        embedded = embedded[sorted_indexes].transpose(0, 1)  # [seq_len, batch_size, embedding_size]

        # pack
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, new_inputs_length)

        # [batch_size, hidden_size]
        outputs_packed, hidden_state = self.rnn(packed_embedded, hidden_state)

        # unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_packed)

        # to original sequence
        outputs = outputs.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        if self.rnn_type == 'LSTM':
            hidden_state = tuple([item.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous() for item in hidden_state])
        else:
            hidden_state = hidden_state.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        return outputs, hidden_state

    def init_hidden(self, batch_size, device):
        initial_state1 = torch.zeros((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)
        if self.rnn_type == 'LSTM':
            initial_state2 = torch.zeros((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)
            return (initial_state1, initial_state2)
        else:
            return initial_state1

        """
        initial_state_scale = math.sqrt(3.0 / self.hidden_size)

        initial_state1 = torch.rand((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)
        nn.init.uniform_(initial_state1, a=-initial_state_scale, b=initial_state_scale)
        if self.rnn_type == 'LSTM':
            initial_state2 = torch.rand((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)
            nn.init.uniform_(initial_state2, a=-initial_state_scale, b=initial_state_scale)
            return (initial_state1, initial_state2)
        else:
            return initial_state1

        """
