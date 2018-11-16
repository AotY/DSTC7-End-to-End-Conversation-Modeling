#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
FeedForward
"""
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 num_layers=1,
                 hidden_size=None,
                 activation='Tanh',
                 bias=True):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.activation = getattr(nn, activation)()
        n_inputs = [input_size] + [hidden_size] * (num_layers - 1)
        n_outputs = [hidden_size] * (num_layers - 1) = [output_size]

        self.linears = nn.ModuleList(
            [nn.Linear(n_in, n_out, bias=bias) for n_in, n_out in zip(n_inputs, n_outputs)])

    def forward(self, input):
        output = input
        for linear in self.Linears:
            output = linear(output)
            output = self.activation(output)

        return output
