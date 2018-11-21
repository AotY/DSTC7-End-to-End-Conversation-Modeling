#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import init_linear_wt
from modules.utils import init_lstm_orth, init_gru_orth
from modules.attention import Attention
from modules.utils import rnn_factory


class LuongAttnDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 dropout,
                 tied,
                 device='cuda',
                 latent_size=0):

        super(LuongAttnDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        # dropout
        self.dropout = nn.Dropout(dropout)

        # h_attn
        self.h_attn = Attention(hidden_size)

        # f_attn
        #  self.f_attn = Attention(hidden_size)
        self.f_linearA = nn.Linear(self.embedding_size,
                                   self.hidden_size + latent_size)

        self.f_linearC = nn.Linear(self.embedding_size,
                                   self.hidden_size + latent_size)
        init_linear_wt(self.f_linearA)
        init_linear_wt(self.f_linearC)

        self.rnn = rnn_factory(
            rnn_type,
            input_size=self.embedding_size,
            hidden_size=hidden_size + latent_size,
            num_layers=num_layers,
            dropout=dropout
        )

        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        if latent_size > 0:
            self.latent_linear = nn.Linear(
                hidden_size + latent_size, hidden_size)
            init_linear_wt(self.latent_linear)

        # linear
        self.linear = nn.Linear(hidden_size, vocab_size)

        if tied and self.embedding_size == hidden_size:
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
                        inputs: [1, batch_size]
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
            inputs_length, sorted_indexes = torch.sort(
                inputs_length, dim=0, descending=True)
            # restore to original indexes
            _, restore_indexes = torch.sort(sorted_indexes, dim=0)

            inputs = inputs.transpose(0, 1)[sorted_indexes].transpose(0, 1)

        # embedded
        embedded = self.embedding(inputs)  # [1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        if inputs_length is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, inputs_length)

        # Get current hidden state from inputs word and last hidden state
        if z is not None:
            hidden_state = torch.cat((hidden_state, z), dim=2)

        output, hidden_state = self.rnn(embedded, hidden_state)

        if inputs_length is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output)
            output = output.transpose(
                0, 1)[restore_indexes].transpose(0, 1).contiguous()

        #  print('output: ', output)

        if h_encoder_outputs is not None and h_encoder_lengths is not None:
            output, h_attn_weights = self.h_attn(
                output, h_encoder_outputs, h_encoder_lengths)
            #  print('h_output: ', h_output)

        if f_encoder_outputs is not None:
            #  output, f_attn_weights = self.f_attn(output, f_encoder_outputs, f_encoder_lengths)
            output = self.f_forward(output, f_encoder_outputs)
            #  print('output: ', output)

        if self.latent_size > 0:
            output = self.latent_linear(output)

        output = self.linear(output)

        # log_softmax
        #  output = self.log_softmax(output)
        output = F.log_softmax(output, dim=2)

        return output, hidden_state, None

    def f_forward(self, output, f_encoder_outputs):
        # M [batch_size, topk, hidden_size]
        fM = self.f_linearA(f_encoder_outputs)

        # C [batch_size, topk, hidden_size]
        fC = self.f_linearC(f_encoder_outputs)

        # [batch_size, num_layers, topk]
        tmpP = torch.bmm(output.transpose(0, 1), fM.transpose(1, 2))

        P = F.softmax(tmpP, dim=2)

        o = torch.bmm(P, fC)  # [batch_size, num_layers, hidden_size]

        u_ = torch.add(o, output.transpose(0, 1))

        # [num_layers, batch_size, hidden_size]
        u_ = u_.transpose(0, 1).contiguous()

        return u_
