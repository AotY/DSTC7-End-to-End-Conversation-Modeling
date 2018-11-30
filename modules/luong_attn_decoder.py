#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import init_linear_wt
from modules.utils import init_lstm_orth, init_gru_orth
from modules.attention import Attention
from modules.utils import rnn_factory


class LuongAttnDecoder(nn.Module):
    def __init__(self, config, embedding):

        super(LuongAttnDecoder, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # q_attn
        self.q_attn = Attention(config.hidden_size)

        # c_attn
        self.c_attn = Attention(config.hidden_size)

        # f_attn
        self.f_attn = Attention(config.hidden_size)

        """
        # c_attn
        self.c_linearK = nn.Linear(self.hidden_size, config.hidden_size)
        self.c_linearV = nn.Linear(self.hidden_size, config.hidden_size)

        init_linear_wt(self.c_linearK)
        init_linear_wt(self.c_linearV)

        # f_attn
        self.f_linearK = nn.Linear(self.embedding_size, config.hidden_size)
        self.f_linearV = nn.Linear(self.embedding_size, config.hidden_size)

        init_linear_wt(self.f_linearK)
        init_linear_wt(self.f_linearV)
        """

        self.rnn = rnn_factory(
            config.rnn_type,
            input_size=self.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        if config.rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        # linear  [q, c, f]
        self.linear = nn.Linear(config.hidden_size * 4, config.vocab_size)
        init_linear_wt(self.linear)

        #  if config.tied and self.embedding_size == config.hidden_size:
            #  self.linear.weight = self.embedding.weight
        #  else:
            #  init_linear_wt(self.linear)

        # log_softmax
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self,
                input,
                hidden_state,
                q_encoder_outputs,
                q_encoder_length,
                c_encoder_outputs,
                c_encoder_length,
                f_encoder_outputs,
                f_encoder_length):
        '''
        Args:
            input: [1, batch_size] or [max_len, batch_size]
            hidden_state: [num_layers, batch_size, hidden_size]
            inputs_length: [batch_size, ] or [1, ]

            h_encoder_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            h_encoder_lengths: [batch_size]

            f_encoder_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            f_encoder_lengths: [batch_size]

        '''
        #  if inputs_length is not None:
            #  # sort inputs_length
            #  inputs_length, sorted_indexes = torch.sort(inputs_length, dim=0, descending=True)
            #  # restore to original indexes
            #  _, restore_indexes = torch.sort(sorted_indexes, dim=0)
            #  input = input.transpose(0, 1)[sorted_indexes].transpose(0, 1)

        # embedded
        embedded = self.embedding(input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        #  if inputs_length is not None:
            #  embedded = nn.utils.rnn.pack_padded_sequence(embedded, inputs_length)

        output, hidden_state = self.rnn(embedded, hidden_state)

        #  if inputs_length is not None:
            #  output, _ = nn.utils.rnn.pad_packed_sequence(output)
            #  output = output.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        if h_encoder_outputs is not None:
            # output: [1, batch_size, 1 * hidden_size]
            q_context, q_attn_weights = self.q_attn(output, q_encoder_outputs, q_encoder_length)

        if c_encoder_outputs is not None:
            # output: [1, batch_size, 1 * hidden_size]
            c_context, c_attn_weights = self.c_attn(output, c_encoder_outputs, c_encoder_length)
            #  print(output.shape)
            #  print(c_encoder_outputs.shape)
            #  print(c_inputs_length.shape)

        if f_encoder_outputs is not None:
            # [1, batch_size, hidden_size]
            f_context, f_attn_weights = self.f_attn(output, f_encoder_outputs, f_encoder_length)
            #  f_context = self.f_forward(output, f_encoder_outputs)

        # [1, batch_size, 3 * hidden_size]
        #  output = torch.cat((output, c_context, f_context), dim=2)
        output = torch.cat((output, q_context, c_context, f_context), dim=2)

        # [1, batch_size, vocab_size]
        output = self.linear(output)

        # log_softmax
        output = self.log_softmax(output)

        return output, hidden_state, None

    def f_forward(self, output, f_encoder_outputs, f_encoder_lengths=None):
        """
        output: [1, batch_size, hidden_size]
        f_encoder_outputs: [topk, batch_size, embedding_size] or [topk, batch_size, hidden_size]
        """
        #  print(output.shape)
        #  print(f_encoder_outputs.shape)
        # K [batch_size, topk, hidden_size]
        fK = self.f_linearK(f_encoder_outputs.transpose(0, 1))

        # V [batch_size, topk, hidden_size]
        fV = self.f_linearV(f_encoder_outputs.transpose(0, 1))

        #  print(fK.shape)
        # [batch_size, 1, topk]
        weights = torch.bmm(output.transpose(0, 1), fK.transpose(1, 2))

        weights = F.softmax(weights, dim=2)

        context = torch.bmm(weights, fV).transpose(0, 1)  # [1, batch_size, hidden_size]

        return context
