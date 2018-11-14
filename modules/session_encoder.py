#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Session encoder, for HRED model.
"""

import math
import torch
import torch.nn as nn

from modules.utils import rnn_factory
from modules.utils import init_gru_orth, init_lstm_orth
from modules.utils import init_wt_normal


class SessionEncoder(nn.Module):
    def __init__(self,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 bidirectional=True,
                 dropout=0.0):
        super(SessionEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirection_num = 2 if bidirectional else 1

        # rnn
        self.rnn = rnn_factory(
            rnn_type,
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // self.bidirection_num,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)


    def forward(self, inputs, lengths=None):
        """
        inputs: [turn_num, batch_size, hidden_size]
        """
        if lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths)

        outputs, hidden_state = self.rnn(inputs)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        return outputs, hidden_state


