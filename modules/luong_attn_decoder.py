#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn

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
        self.f_attn = Attention(hidden_size)

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
            self.latent_linear = nn.Linear(hidden_size + latent_size, hidden_size)
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
        #  print("input: ", inputs.shape)
        #  print("hidden_state: ", hidden_state.shape)
        #  print(inputs_length)
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

        outputs, hidden_state = self.rnn(embedded, hidden_state)

        if inputs_length is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = outputs.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        attn_weights = None
        if h_encoder_outputs is not None and h_encoder_lengths is not None:
            outputs, attn_weights = self.h_attn(outputs, h_encoder_outputs, h_encoder_lengths)

        if f_encoder_outputs is not None and f_encoder_lengths is not None:
            outputs, attn_weights = self.f_attn(outputs, f_encoder_outputs, f_encoder_lengths)

        if self.latent_size > 0:
            outputs = self.latent_linear(outputs)

        outputs = self.linear(outputs)

        # log log_softmax
        outputs = self.log_softmax(outputs)

        return outputs, hidden_state, attn_weights
