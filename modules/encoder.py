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

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0,
                 padding_idx=0):

        super(Encoder, self).__init__()

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
        # embedded
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        # [batch_size, seq_len, embedding_size]
        embedded = embedded.transpose(0, 1)

        # sort lengths
        _, sorted_indexes = torch.sort(inputs_length, dim=0, descending=True)
        new_inputs_length = inputs_length[sorted_indexes]

        # restore to origianl indexs
        _, restore_indexes = torch.sort(sorted_indexes, dim=0)

        # new embedded
        embedded = embedded[sorted_indexes].transpose(0, 1)  # [seq_len, batch_size, embedding_size]

        # pack
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, new_inputs_length)

        # batch_size, hidden_size]
        outputs, hidden_state = self.lstm(packed_embedded, hidden_state)

        # unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        #  if self.bidirection_num == 2:
            # [seq, batch_size, hidden_size * 2] -> [seq, batch_size, hidden_size]
            #  outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # to original sequence
        outputs = outputs.contiguous()
        outputs = outputs.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()
        hidden_state = tuple([item.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous() for item in hidden_state])

        print('encoder outputs shape: {}'.format(outputs.shape))
        max_output, _ = outputs.max(dim=0)
        max_output.unsqueeze_(0) #[1, batch_size, hidden_size * 2]
        print('max_output shape: {}'.format(max_output.shape))

        # dropout
        # outputs = self.dropout(outputs)
        return outputs, hidden_state, max_output

    def init_hidden(self, batch_size, device):
        initial_state_scale = math.sqrt(3.0 / self.hidden_size)

        initial_state1 = torch.rand((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)
        initial_state2 = torch.rand((self.num_layers * self.bidirection_num, batch_size, self.hidden_size), device=device)

        nn.init.uniform_(initial_state1, a=-initial_state_scale, b=initial_state_scale)
        nn.init.uniform_(initial_state2, a=-initial_state_scale, b=initial_state_scale)
        return (initial_state1, initial_state2)
