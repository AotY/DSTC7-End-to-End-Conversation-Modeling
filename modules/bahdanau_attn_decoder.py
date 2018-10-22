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
import torch.nn.functional as F

from modules.global_attn import GlobalAttn

class BahdanauAttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 dropout_ratio,
                 padding_idx,
                 tied,
                 attn_type='concat',
                 device=None):

        super(BahdanauAttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        # embedding
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_size, self.padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout_ratio)

        # attn
        self.attn = GlobalAttn(attn_type, self.hidden_size, device)

        # hidden_size -> embedding_size, for attn
        if self.hidden_size != self.embedding_size:
            self.hidden_embedded_linear = nn.Linear(self.hidden_size, self.embedding_size)

        # LSTM, * 2 because using concat
        self.lstm = nn.LSTM(self.embedding_size * 2,
                            self.hidden_size,
                            self.num_layers,
                            dropout=dropout_ratio)
        # linear
        self.linear = nn.Linear(self.hidden_size,
                                self.vocab_size)

        if tied:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden_state, encoder_outputs):
        '''
        input: [1, batch_size]  LongTensor
        hidden_state: [num_layers, batch_size, hidden_size]
        output: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
        hidden_state: (h_n, c_n)
        '''

        # embedded
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)
        #  print(embedded.shape)

        # attn_weights
        # Calculate attention weights and apply to encoder outputs
        # LSTM hidden_state (h, c)
        # hidden_state[0][-1]: [batch_size, hidden_size],
        # encoder_outputs: [max_len, batch_size, hidden_size]
        # [batch_size, max_len]
        attn_weights = self.attn(hidden_state[0][-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, max_len]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)

        # Combine embedded input word and attened context, run through RNN
        if self.hidden_size != self.embedding_size:
            context = self.hidden_embedded_linear(context)

        # [1, batch_size, embedding_size * 2]
        input_combine = torch.cat((context, embedded), dim=2)

        # lstm
        output, hidden_state = self.lstm(input_combine, hidden_state)

        # [1, batch_size, hidden_size]

        # linear
        output = self.linear(output)

        # softmax
        output = self.softmax(output)

        return output, hidden_state, attn_weights
