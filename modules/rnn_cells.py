#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Modified From OpenNMT, z-forcing
https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/master/model/layers/rnncells.py
"""

import torch
import torch.nn as nn


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout=0.5):
        super(StackedLSTMCell, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, h_c):
        """
        Args:
            input: [batch_size, input_size]
            h_c: ([num_layers, batch_size, input_size], [num_layers, batch_size, input_size])
        Return:
            last_h_c: ([batch_size, hidden_size], (batch_size, hidden_size])
            h_c: ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
        """
        h_0, c_0 = h_c
        h_list, c_list = list(), list()

        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(input, (h_0[i], c_0[i]))

            input = h_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)

            h_list += [h_i]
            c_list += [c_i]

        last_h_c = (h_list[-1], c_list[-1])

        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)

        h_c = (h_list, c_list)

        return last_h_c, h_c


class StackedGRUCell(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout=0.5):
        super(StackedGRUCell, self).__init__()
        self.dropout= nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, h):
        """
        Args:
            input: [batch_size, input_size]
            h: [num_layers, batch_size, hidden_size]
        Return:
            last_h: [batch_size, hidden_size]
            h_list: [num_layers, batch_size, hidden_size]
        """

        h_list = list()
        for i, layer in enumerate(self.layers):
            h_i = layer(input, h[i])

            input = h_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)

            h_list.append(h_i)

        last_h = h_list[-1]
        h_list = torch.stack(h_list)

        return last_h, h_list

