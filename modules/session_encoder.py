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

from modules.utils import init_wt_normal
from modules.utils import rnn_factory
from modules.utils import init_lstm_wt, init_gru_orth, init_lstm_orth


class SessionEncoder(nn.Module):
    def __init__(self,
                 rnn_type,
                 hidden_size,
                 dropout=0.0):
        super(SessionEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size

        # rnn
        self.rnn = rnn_factory(
            rnn_type,
            input_size=hidden_size,
            hidden_size=hidden_size,
            #  dropout=dropout
        )

        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)


    def forward(self, input, hidden_state):
        """
        input: [1, batch_size, hidden_size]
        """
        output, hidden_state = self.rnn(input, hidden_state)

        return output, hidden_state

    def init_hidden(self, batch_size, device):
        initial_state1 = torch.zeros((1, batch_size, self.hidden_size), device=device)
        if self.rnn_type == 'LSTM':
            initial_state2 = torch.zeros((1, batch_size, self.hidden_size), device=device)
            return (initial_state1, initial_state2)
        else:
            return initial_state1



