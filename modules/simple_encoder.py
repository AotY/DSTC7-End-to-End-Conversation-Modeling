#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Encoder based on lstm
"""
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
                 num_layers=2,
                 bidirectional=True,
                 dropout=0.0):

        super(SimpleEncoder, self).__init__()

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

    def forward(self, inputs, lengths=None):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [seq_len, batch_size, num_directions * hidden_size]
            max_output: [1, batch_size, hidden_size * num_directions]
            hidden_state: (h_n, c_n)
        '''
        if lengths is not None:
            # sort lengths
            lengths, sorted_indexes = torch.sort(lengths, dim=0, descending=True)
            # restore to original indexes
            _, restore_indexes = torch.sort(sorted_indexes, dim=0)

            inputs = inputs.transpose(0, 1)[sorted_indexes].transpose(0, 1)

        # embedded
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        # [batch_size, hidden_size]
        outputs, hidden_state = self.rnn(embedded)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        if lengths is not None:
            outputs = outputs.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        return outputs, hidden_state
