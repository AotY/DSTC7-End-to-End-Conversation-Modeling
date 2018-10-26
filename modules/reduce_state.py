#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import init_linear_wt

"""
hidden_state state
num_layers * num_directions, batch_size, hidden_size
->
num_layers, batch_size, hidden_size
"""

class ReduceState(nn.Module):
    def __init__(self, rnn_type, hidden_size, num_layers, bidirection_num):
        super(ReduceState, self).__init__()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
        #  init_linear_wt(self.reduce_h)

        if rnn_type == 'LSTM':
            self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)
            #  init_linear_wt(self.reduce_c)

    def forward(self, hidden_state, batch_size):
        # [num_layers * bidirection_num, batch_size, hidden_size]
        if self.rnn_type == 'LSTM':
            h, c = hidden_state
            reduce_h = F.relu(self.reduce_h(torch.cat((h[:h.shape[0] // 2], h[h.shape[0] // 2:]) ,dim=2))) # [num_layers, batch_size, hidden_size]
            reduce_c = F.relu(self.reduce_c(torch.cat((c[:c.shape[0] // 2], c[c.shape[0] // 2:]) ,dim=2))) # [num_layers, batch_size, hidden_size]
            #  reduce_c = F.relu(self.reduce_c(c.view(-1, batch_size, self.hidden_size * 2)))
            return (reduce_h, reduce_c)
        else:
            h = hidden_state
            reduce_h = F.relu(self.reduce_h(torch.cat((h[:h.shape[0] // 2], h[h.shape[0] // 2:]) ,dim=2))) # [num_layers, batch_size, hidden_size]
            return reduce_h

        #  hidden_reduced_h = F.relu(self.reduce_h(h.view(-1, hidden_size * 2)))
        #  hidden_reduced_c = F.relu(self.reduce_c(c.view(-1, hidden_size * 2)))
        #  return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_size


        """
        if self.bidirection_num == 2:
            new_hidden_state = tuple([item[:item.shape[0] // 2, :, :] + item[item.shape[0] // 2:, :, :] for item in hidden_state])
        else:
            new_hidden_state = hidden_state

        return new_hidden_state
        """

