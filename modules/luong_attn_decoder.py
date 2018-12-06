#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

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
        #  if config.turn_type not in ['none', 'concat']:
            #  self.c_attn = Attention(config.hidden_size)

        # f_attn
        if config.model_type == 'kg':
            self.f_attn = Attention(config.hidden_size)

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
        """
        if config.model_type == 'kg':
            self.linear = nn.Linear(config.hidden_size * 4, config.vocab_size)
        else:
            if config.turn_type not in ['none', 'concat']:
                self.linear = nn.Linear(config.hidden_size * 3, config.vocab_size)
            else:
                self.linear = nn.Linear(config.hidden_size * 2, config.vocab_size)
        """
        self.linear = nn.Linear(config.hidden_size * 2, config.vocab_size)

        init_linear_wt(self.linear)

    def forward(self,
                dec_input,
                dec_hidden,
                q_enc_outputs=None,
                q_enc_length=None,
                c_enc_outputs=None,
                c_enc_length=None,
                f_enc_outputs=None,
                f_enc_length=None):
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

        output, dec_hidden = self.rnn(embedded, dec_hidden)

        # output: [1, batch_size, 1 * hidden_size]
        q_context, q_attn_weights = self.q_attn(output, q_enc_outputs, q_enc_length)

        c_context = None
        #  if c_enc_outputs is not None:
            #  # output: [1, batch_size, 1 * hidden_size]
            #  c_context, c_attn_weights = self.c_attn(output, c_enc_outputs, c_enc_length)

        f_context = None
        #  if f_enc_outputs is not None:
            #  # [1, batch_size, hidden_size]
            #  f_context, f_attn_weights = self.f_attn(c_context, f_enc_outputs, f_enc_length)

        output_list = [output, q_context]
        if c_context is not None:
            output_list.append(c_context)

        if f_context is not None:
            output_list.append(f_context)

        output = torch.cat(output_list, dim=2)

        # [, batch_size, vocab_size]
        output = self.linear(output)

        return output, dec_hidden, q_attn_weights
