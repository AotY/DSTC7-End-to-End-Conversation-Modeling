#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_linear_wt

"""
hidden_state state
num_layers * num_directions, batch_size, hidden_size
->
num_layers, batch_size, hidden_size
"""

class ReduceState(nn.Module):
    def __init__(self,
                 hidden_size):
        super(ReduceState, self).__init__()

        self.hidden_size = hidden_size

        self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
        init_linear_wt(self.reduce_h)

        self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden_state, batch_size):
        h, c = hidden_state  # h, c dim = 2 x b x hidden_size
        batch_size = h.shape[0]
        #  hidden_reduced_h = F.relu(self.reduce_h(h.view(-1, config.hidden_size * 2)))
        #  hidden_reduced_c = F.relu(self.reduce_c(c.view(-1, config.hidden_size * 2)))
        #  return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_size
        hidden_reduce_h = F.relu(self.reduce_h(h.view(-1, batch_size, self.hidden_size * 2))) # [num_layers, batch_size, hidden_size]
        hidden_reduce_c = F.relu(self.recuce_c(c.view(-1, batch_size, self.hidden_size * 2)))

        return (hidden_reduce_h, hidden_reduce_c)


