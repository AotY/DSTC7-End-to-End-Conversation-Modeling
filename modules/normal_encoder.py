#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Normal Encoder.
"""
import torch
import torch.nn as nn
from modules.utils import rnn_factory
from modules.utils import init_gru_orth, init_lstm_orth
from misc.vocab import PAD_ID


class NormalEncoder(nn.Module):
    def __init__(self,
                 config,
                 embedding):

        super(NormalEncoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.bidirection_num = 2 if config.bidirectional else 1
        self.hidden_size = config.hidden_size // self.bidirection_num

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # rnn
        self.rnn = rnn_factory(
            config.rnn_type,
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=config.encoder_num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
            #  bias=False
        )

        if config.rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

    def forward(self, inputs, lengths, hidden_state=None, sort=True):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [seq_len, batch_size, num_directions * hidden_size]
            max_output: [1, batch_size, hidden_size * num_directions]
            hidden_state: (h_n, c_n)
        '''
        if lengths is None:
            raise ValueError('lengths is none.')

        total_length = inputs.size(0)

        #  print('inputs: ', inputs)
        #  print('lengths: ', lengths)
        sorted_lengths = lengths
        if not sort:
            # sort lengths
            sorted_lengths, sorted_indexes = torch.sort(lengths, dim=0, descending=True)
            #  print('sorted_lengths: ', sorted_lengths)
            #  print('sorted_indexes: ', sorted_indexes)

            # restore to original indexes
            _, restore_indexes = torch.sort(sorted_indexes, dim=0)
            #  print('restore_indexes: ', restore_indexes)

            # [max_len, batch_size]
            inputs = inputs.index_select(1, sorted_indexes)
            #  print('inputs: ', inputs)

        # embedded
        embedded = self.embedding(inputs)
        #  print('embedded: ', embedded)
        embedded = self.dropout(embedded)
        #  print('embedded: ', embedded)

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths)
        #  print('embedded: ', embedded)

        if hidden_state is not None:
            outputs, hidden_state = self.rnn(embedded, hidden_state)
        else:
            outputs, hidden_state = self.rnn(embedded)

        #  print('outputs: ', outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, padding_value=PAD_ID, total_length=total_length)
        #  print('outputs: ', outputs)

        if not sort:
            # [max_len, batch_size, hidden_state]
            outputs = outputs.index_select(1, restore_indexes).contiguous()
            #  print('outputs: ', outputs)

            # [num_layer * bidirection_num, batch_size, hidden_state /
            # bidirection_num]
            hidden_state = hidden_state.index_select(1, restore_indexes).contiguous()
            #  print('hidden_state: ', hidden_state)

        return outputs, hidden_state
