#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn

from modules.utils import init_lstm_wt
from modules.global_attn import GlobalAttn



class LuongAttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 dropout_ratio,
                 padding_idx,
                 tied,
                 attn_type,
                 device):

        super(LuongAttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.attn_type = attn_type

        # embedding
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_size, self.padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout_ratio)

        # attn
        self.attn = GlobalAttn(self.attn_type, self.hidden_size, device)

        # encoder_max_output + embedded ->
        self.encoder_concat_linear = nn.Linear(hidden_size * 2 + embedding_size, embedding_size)

        # LSTM
        self.lstm = nn.LSTM(self.embedding_size,
                            self.hidden_size,
                            self.num_layers,
                            dropout=dropout_ratio)
        init_lstm_wt(self.lstm)

        # concat linear
        self.concat_linear = nn.Linear(hidden_size * 3, hidden_size)

        # linear
        self.linear = nn.Linear(hidden_size, vocab_size)

        if tied and embedding_size == hidden_size:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, encoder_max_output, encoder_outputs):
        '''
        input: [1, batch_size]  LongTensor
        hidden_state: [num_layers, batch_size, hidden_size]
        output: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
        hidden_state: (h_n, c_n)
        '''

        # embedded
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        encoder_concat = self.encoder_concat_linear(torch.cat((encoder_max_output, embedded), dim=2))

        # Get current hidden state from input word and last hidden state
        # output: [1, batch_size, hidden_size]
        output, hidden_state = self.lstm(encoder_concat, hidden_state)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average<Paste>
        attn_weights = self.attn(output.squeeze(0), encoder_outputs)  # [batch_size, max_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, max_len]
        # [batch_size, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)  # [1, batch_size, hidden_size]

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        # [1, batch_size, hidden_size * 2]
        concat_input = torch.cat((context, output), dim=2)

        # [1, batch_size, hidden_size]
        concat_output = torch.tanh(self.concat_linear(concat_input))

        # linear
        output = self.linear(concat_output)

        # [1, batch_size, hidden_size]
        # softmax
        output = self.softmax(output)

        return output, hidden_state, attn_weights
