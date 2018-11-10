#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from modules.utils import init_wt_normal, init_linear_wt, init_lstm_orth, init_gru_orth
from modules.attention import Attention
from modules.utils import rnn_factory

class SessionDecoder(nn.Module):
    def __init__(self,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 dropout,
                 attn_type):

        super(SessionDecoder, self).__init__()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attn_type = attn_type

        # c_attn
        self.attn = Attention(hidden_size)

        self.rnn = rnn_factory(
            rnn_type,
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

    def forward(self,
                input,
                hidden_state,
                encoder_outputs):
        '''
		Args:
			input: [1, batch_size]  or [len, batch_size]
			hidden_state: [num_layers, batch_size, hidden_size]
			encoder_outputs: [seq_len, batch, hidden_size]
        '''

        # Get current hidden state from input word and last hidden state
        output, hidden_state = self.rnn(input, hidden_state)

        # c attention
        attn_output, attn_weights = self.attn(output, encoder_outputs)

        return attn_output, hidden_state, attn_weights
