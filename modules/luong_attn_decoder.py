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
                dec_input,
                dec_hidden,
                dec_context,
                q_enc_outputs,
                q_enc_length,
                c_enc_outputs,
                c_enc_length,
                f_enc_outputs,
                f_enc_length):
        '''
        Args:
            dec_input: [1, batch_size] or [max_len, batch_size]
            dec_hidden: [num_layers, batch_size, hidden_size]
            inputs_length: [batch_size, ] or [1, ]

            h_encoder_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            h_encoder_lengths: [batch_size]

            f_enc_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            f_enc_length: [batch_size]
        '''
        # embedded
        embedded = self.embedding(dec_input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        rnn_input = torch.cat((embedded, dec_context), dim=2) # [1, batch_size, embedding_size + hidden_size]
        output, dec_hidden = self.rnn(rnn_input, dec_hidden)

        # output: [1, batch_size, 1 * hidden_size]
        q_context, q_attn_weights = self.q_attn(output, q_enc_outputs, q_enc_length)

        if c_enc_outputs is not None:
            # output: [1, batch_size, 1 * hidden_size]
            c_context, c_attn_weights = self.c_attn(q_context, c_enc_outputs, c_enc_length)

        if f_enc_outputs is not None:
            # [1, batch_size, hidden_size]
            f_context, f_attn_weights = self.f_attn(c_context, f_enc_outputs, f_enc_length)

        #  output = torch.cat((output, c_context, f_context), dim=2)
        output = torch.cat((output, q_context, c_context, f_context), dim=2)

        # [1, batch_size, vocab_size]
        output = self.linear(output)

        # log_softmax
        output = self.log_softmax(output)

        return output, dec_hidden, q_context, None
