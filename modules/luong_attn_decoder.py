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
        enc_type = config.enc_type

        if enc_type.count('attn') != 0:
            self.enc_attn = Attention(config.hidden_size)

        # f_attn
        if config.model_type == 'kg':
            self.f_attn = Attention(config.hidden_size)

        self.rnn = rnn_factory(
            config.rnn_type,
            input_size=self.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.decoder_num_layers,
            dropout=config.dropout
        )

        if config.rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        # linear  [q, c, f]
        in_features = 0
        if config.model_type == 'kg':
            if enc_type.count('attn') != 0:
                in_features = config.hidden_size * 3
            else:
                in_features = config.hidden_size * 2
        else:
            if enc_type.count('attn') != 0:
                in_features = config.hidden_size * 2
            else:
                in_features = config.hidden_size
        self.linear = nn.Linear(in_features, config.vocab_size)
        init_linear_wt(self.linear)

    def forward(self,
                dec_input,
                dec_hidden,
                enc_outputs=None,
                enc_length=None,
                f_enc_outputs=None,
                f_enc_length=None):
        '''
        Args:
            h_encoder_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            h_encoder_lengths: [batch_size]

            f_enc_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            f_enc_length: [batch_size]
        '''
        # embedded
        embedded = self.embedding(dec_input)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        rnn_output, dec_hidden = self.rnn(embedded, dec_hidden)

        enc_context = None
        if enc_outputs is not None:
            # [1, batch_size, hidden_size]
            enc_context, _ = self.enc_attn(rnn_output, enc_outputs, enc_length)

        f_context = None
        if f_enc_outputs is not None:
            # [1, batch_size, hidden_size]
            #  if enc_context is not None:
                #  f_context, _ = self.f_attn(enc_context, f_enc_outputs, f_enc_length)
            #  else:
            f_context, _ = self.f_attn(rnn_output, f_enc_outputs, f_enc_length)

        output_list = list()

        output_list.append(rnn_output)

        if enc_context is not None:
            output_list.append(enc_context)

        if f_context is not None:
            output_list.append(f_context)

        concat_output = torch.cat(output_list, dim=2)

        # [, batch_size, vocab_size]
        #  print('concat_output: ', concat_output.shape)
        output = self.linear(concat_output)

        return output, dec_hidden, None
