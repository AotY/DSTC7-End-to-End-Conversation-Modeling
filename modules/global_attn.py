#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import init_linear_wt


'''
GlobalAttn
    dot:
    general:
    concat:
'''


class GlobalAttn(nn.Module):
    def __init__(self, attn_method, hidden_size, device):

        super(GlobalAttn, self).__init__()

        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.device = device

        if self.attn_method == 'general':
            self.attn_linear = nn.Linear(hidden_size, self.hidden_size)
            init_linear_wt(self.attn_linear)
        elif self.attn_method == 'concat':
            self.attn_linear = nn.Linear(hidden_size * 2, hidden_size)
            init_linear_wt(self.attn_linear)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size, device=device))

    def forward(self, hidden_state, encoder_outputs):
        """
        hidden_state: [batch_size, 1, hidden_size]
        encoder_outputs: [batch_size, max_len, hidden_size]
        """
        batch_size, max_len, hidden_size = encoder_outputs.shape

        """
        attn_weights = torch.zeros((batch_size, max_len), device=self.device)  # [batch_size, max_len]
        # For each batch of encoder outputs
        for bi in range(batch_size):
            #  weight for each encoder_output
            for li in range(max_len):
                one_encoder_output = encoder_outputs[bi, li, :].unsqueeze(0)  # [1, hidden_size]
                one_hidden_state = hidden_state[bi, :].unsqueeze(0)  # [1, hidden_size]

                attn_weights[bi, li] = self.score(one_hidden_state, one_encoder_output)

        """
        if self.attn_method == 'dot':
            attn_weights = torch.bmm(hidden_state, encoder_outputs.transpose(1, 2))
        elif self.attn_method == 'general':
            attn_weights = torch.bmm(hidden_state, self.attn_linear(encoder_outputs).transpose(1, 2))
        elif self.attn_method == 'concat':
            pass
            #  weight = torch.dot(self.v.view(-1),
                               #  torch.tanh(self.attn_linear(torch.cat((one_hidden_state, one_encoder_output), dim=1))).view(-1))

        # Normalize energies to weights in range 0 to 1
        attn_weights = F.softmax(attn_weights, dim=2)
        return attn_weights

    """
    def score(self, one_hidden_state, one_encoder_output):
        #  print(one_encoder_output.shape)
        #  print(one_hidden_state.shape)

        if self.attn_method == 'general':
            #  weight = one_hidden_state.dot(self.attn_linear(one_encoder_output))
            weight = torch.dot(one_hidden_state.view(-1), self.attn_linear(one_encoder_output).view(-1))

        elif self.attn_method == 'dot':
            weight = torch.dot(one_hidden_state.view(-1), one_encoder_output.view(-1))

        elif self.attn_method == 'concat':
            #  weight = self.v.dot(self.attn_linear(
                #  torch.cat((one_hidden_state, one_encoder_output), dim=1)))
            weight = torch.dot(self.v.view(-1),
                               torch.tanh(self.attn_linear(torch.cat((one_hidden_state, one_encoder_output), dim=1))).view(-1))

        return weight

    """

