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

        # h_attn
        self.h_attn = Attention(config.hidden_size)

        # f_attn
        #  if model_type == 'kg':
        #  self.f_attn = Attention(hidden_size)
        self.f_linearA = nn.Linear(self.embedding_size, config.hidden_size + config.latent_size)

        self.f_linearC = nn.Linear(self.embedding_size, config.hidden_size + config.latent_size)

        self.f_concat_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        init_linear_wt(self.f_linearA)
        init_linear_wt(self.f_linearC)

        self.rnn = rnn_factory(
            config.rnn_type,
            input_size=self.embedding_size,
            hidden_size=config.hidden_size + config.latent_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        if config.rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        if config.latent_size > 0:
            self.latent_linear = nn.Linear(config.hidden_size + config.latent_size, config.hidden_size)
            init_linear_wt(self.latent_linear)

        # linear
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

        if config.tied and self.embedding_size == config.hidden_size:
            self.linear.weight = self.embedding.weight
        else:
            init_linear_wt(self.linear)

        # log_softmax
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self,
                inputs,
                hidden_state,
                inputs_length=None,
                h_encoder_outputs=None,
                h_encoder_lengths=None,
                f_encoder_outputs=None,
                f_encoder_lengths=None,
                z=None):
        '''
        Args:
            inputs: [1, batch_size] or [max_len, batch_size]
            hidden_state: [num_layers, batch_size, hidden_size]
            inputs_length: [batch_size, ] or [1, ]

            h_encoder_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            h_encoder_lengths: [batch_size]

            f_encoder_outputs: [turn_num, batch_size, hidden_size] or [1, batch_size, hidden_size]
            f_encoder_lengths: [batch_size]

            z: for latent variable model. [num_layers, batch_size, latent_size]
        '''
        #  print("inputs: ", inputs.shape)
        #  print("hidden_state: ", hidden_state.shape)
        #  print("h_encoder_outputs: ", h_encoder_outputs)
        #  print("f_encoder_outputs: ", f_encoder_outputs.shape)

        if inputs_length is not None:
            # sort inputs_length
            inputs_length, sorted_indexes = torch.sort(inputs_length, dim=0, descending=True)
            # restore to original indexes
            _, restore_indexes = torch.sort(sorted_indexes, dim=0)
            inputs = inputs.transpose(0, 1)[sorted_indexes].transpose(0, 1)

        # embedded
        embedded = self.embedding(inputs)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        if inputs_length is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, inputs_length)

        # Get current hidden state from inputs word and last hidden state
        if z is not None:
            hidden_state = torch.cat((hidden_state, z), dim=2)

        output, hidden_state = self.rnn(embedded, hidden_state)

        if inputs_length is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output)
            output = output.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        if h_encoder_outputs is not None and h_encoder_lengths is not None:
            # output: [1, batch_size, 1 * hidden_size]
            output, h_attn_weights = self.h_attn(output, h_encoder_outputs, h_encoder_lengths)

        if f_encoder_outputs is not None:
            # [1, batch_size, hidden_size]
            output = self.f_forward(output, f_encoder_outputs)

        if self.latent_size > 0:
            output = self.latent_linear(output)

        output = self.linear(output)

        # log_softmax
        output = self.log_softmax(output)
        #  output = F.log_softmax(output, dim=2)

        return output, hidden_state, None

    def f_forward(self, output, f_encoder_outputs, f_encoder_lengths=None):
        """
        output: [1, batch_size, hidden_size]
        f_encoder_outputs: [topk, batch_size, embedding_size]
        """
        #  print('output: ', output.shape)
        #  print('f_encoder_outputs: ', f_encoder_outputs.shape)
        # K [batch_size, topk, hidden_size]
        fK = self.f_linearA(f_encoder_outputs.transpose(0, 1))

        # V [batch_size, topk, hidden_size]
        fV = self.f_linearC(f_encoder_outputs.transpose(0, 1))
        #  print(fK.shape)
        #  print(fV.shape)
        #  print(output.shape)

        # [batch_size, 1, topk]
        weights = torch.bmm(output.transpose(0, 1), fK.transpose(1, 2))

        weights = F.softmax(weights, dim=2)

        o = torch.bmm(weights, fV).transpose(0, 1)  # [1, batch_size, hidden_size]

        # [1, batch_size, 2 * hidden_size]
        output = torch.cat((o, output), dim=2)

        output = self.f_concat_linear(output)

        return output
