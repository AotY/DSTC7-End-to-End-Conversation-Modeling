#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Implementation of "Convolutional Sequence to Sequence Learning"
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/cnn_factory.py
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from modules.weight_norm import WeightNormConv2d

import onmt.modules

SCALE_WEIGHT = 0.5 ** 0.5


def shape_transform(input):
    """ 
    Tranform the size of the tensors to fit for conv input. 
    input: [batch_size, max_len, hidden_size]

    return: [batch_size, hidden_size, max_len, 1]
    """
    return torch.unsqueeze(torch.transpose(input, 1, 2), 3)


class GatedConv(nn.Module):
    """ Gated convolution for CNN class """

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = WeightNormConv2d(
            input_size,
            2 * input_size,
            kernel_size=(width, 1),
            stride=(1, 1),
            padding=(width // 2 * (1 - nopad), 0)
        )
        init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout))**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input = self.dropout(input)
        input = self.conv(input)
        out, gate = input.split(int(input.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out


class StackedCNN(nn.Module):
    """ Stacked CNN class """

    def __init__(self, 
                 num_layers,
                 input_size,
                 cnn_kernel_width=3,
                 dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GatedConv(input_size, cnn_kernel_width, dropout)
            )

    def forward(self, input):
        """
        input: [batch_size, hidden_size, max_len, 1]
        return: [batch_size, ]
        """
        for conv in self.layers:
            input = input + conv(input)
            input *= SCALE_WEIGHT
        return input
