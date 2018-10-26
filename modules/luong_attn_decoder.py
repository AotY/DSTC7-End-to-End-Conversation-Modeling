#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn

from modules.utils import init_lstm_wt, init_linear_wt
from modules.global_attn import GlobalAttn
from modules.utils import rnn_factory



class LuongAttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 dropout,
                 padding_idx,
                 tied,
                 attn_type,
                 device):

        super(LuongAttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.attn_type = attn_type

        # embedding
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_size,
                                      padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # attn
        self.attn = GlobalAttn(self.attn_type, self.hidden_size, device)

        # history attn
        self.attn_history = GlobalAttn(self.attn_type, self.hidden_size, device)

        # encoder_max_output + embedded ->
        self.encoder_concat_linear = nn.Linear(hidden_size * 2 + embedding_size, embedding_size)

        # rnn
        self.rnn = rnn_factory(
            rnn_type,
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        #  init_lstm_wt(self.lstm)

        #  self.reduce_linear = nn.Linear(hidden_size * 2, hidden_size)
        #  init_linear_wt(self.reduce_linear)

        # concat linear
        self.concat_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_linear_history = nn.Linear(hidden_size * 3, hidden_size)

        # linear
        self.linear = nn.Linear(hidden_size, vocab_size)

        if tied and embedding_size == hidden_size:
            self.linear.weight = self.embedding.weight

        # log softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, encoder_max_output, encoder_outputs, history_encoder_outputs=None):
        '''
		Args:
			input: [1, batch_size]  LongTensor
			hidden_state: [num_layers, batch_size, hidden_size]
			encoder_outputs: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
			history_encoder_outputs: [len, batch_size, hidden_size]
			hidden_state: (h_n, c_n)
        '''

        # embedded
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        #  encoder_concat = self.encoder_concat_linear(torch.cat((encoder_max_output, embedded), dim=2))

        # Get current hidden state from input word and last hidden state
        # output: [1, batch_size, hidden_size]
        output, hidden_state = self.rnn(embedded, hidden_state)

        # hidden_size * 2 -> hidden_size
        #  reduced_encoder_outputs = self.reduce_linear(encoder_outputs)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average<Paste>
        attn_weights = self.attn(output.squeeze(0), encoder_outputs)  # [batch_size, max_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, max_len]

        # [batch_size, 1, hidden_size]
        context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)  # [1, batch_size, hidden_size]

        if history_encoder_outputs is not None:
            attn_weights_history = self.attn_history(output.squeeze(0), history_encoder_outputs)  # [batch_size, max_len]
            attn_weights_history = attn_weights_history.unsqueeze(1)  # [batch_size, 1, max_len]
            context_history = torch.bmm(attn_weights_history, history_encoder_outputs.transpose(0, 1))
            context_history = context_history.transpose(0, 1)  # [1, batch_size, hidden_size]

            concat_input = torch.cat((context, context_history, output), dim=2)
            concat_output = torch.tanh(self.concat_linear_history(concat_input))

        else:

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
