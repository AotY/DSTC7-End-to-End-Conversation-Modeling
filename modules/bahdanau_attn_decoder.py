#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
BahdanauAttnDecoder
"""
import torch
import torch.nn as nn

from modules.global_attn import GlobalAttn
from modules.utils import init_gru_orth, init_lstm_orth, init_linear_wt, init_wt_normal
from modules.utils import rnn_factory

class BahdanauAttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 dropout,
                 tied,
                 turn_type='hred',
                 attn_type='concat',
                 device='cuda'):

        super(BahdanauAttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.turn_type = turn_type

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        # dropout
        self.dropout = nn.Dropout(dropout)

        # attn
        self.attn = GlobalAttn(attn_type, hidden_size, device)

        # rnn
        self.rnn = rnn_factory(
            rnn_type,
            input_size=self.embedding_size + self.embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        # linear
        self.linear = nn.Linear(self.hidden_size,
                                self.vocab_size)

        if turn_type == 'hred_attn':
            selef.h_attn = GlobalAttn(attn_type, hidden_size, device)
        elif turn_type == 'hred':
            self.h_linear = nn.Linear(hidden_size + hidden_size, hidden_size)

        if tied and self.embedding_size == hidden_size:
            self.linear.weight = self.embedding.weight
        else:
            init_linear_wt(self.linear)

        # log softmax
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden_state, c_encoder_outputs, h_encoder_outputs=None):
        '''
        Args:
            input: [1, batch_size]  LongTensor
            hidden_state: [num_layers, batch_size, hidden_size]
            c_encoder_outputs: [max_len, batch_size, hidden_size]
            h_encoder_outputs: [num, batch_size, hidden_size]

        return:
            output: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
        '''
        # embedded
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)
        #  print(embedded.shape)

        # attn_weights
        if self.rnn_type == 'GRU':
            attn_weights = self.attn(hidden_state[-1], c_encoder_outputs)  # [batch_size, 1, max_len]
        else:
            attn_weights = self.attn(hidden_state[0][-1], c_encoder_outputs)  # [batch_size, 1, max_len]

        context = attn_weights.bmm(c_encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1) #[1, batch_size, hidden_size]

        # Combine embedded input word and attened context, run through RNN
        rnn_input = torch.cat((embedded, context), dim=2)

        # rnn
        output, hidden_state = self.rnn(rnn_input, hidden_state)

        if h_encoder_outputs is not None:
            if self.turn_type == 'hred_attn':
                pass
            elif self.turn_type == 'hred':
                #  con_output = output + h_encoder_outputs[-1].unsqueeze(0)
                con_output = torch.cat((output, h_encoder_outputs[-1].unsqueeze(0)), dim=2)
                output = self.h_linear(con_output)

        # linear
        output = self.linear(output)

        # softmax
        output = self.softmax(output)

        return output, hidden_state, attn_weights
