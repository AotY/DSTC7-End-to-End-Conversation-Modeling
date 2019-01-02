#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Define the sub layers in encoder (or decoder) layer.
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
"""

import numpy as np
import torch
import torch.nn as nn

from modules.transformer.sdpa import ScaleDotProductAttention
from modules.utils import init_wt_normal

class  MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""
    def __init__(self,
                config):
        super().__init__()

        self.num_heads = config.num_heads

        self.k_size = config.k_size
        self.v_size = config.v_size

        self.q_linear = nn.Linear(config.transformer_size, config.num_heads * config.k_size)
        self.k_linear = nn.Linear(config.transformer_size, config.num_heads * config.k_size)
        self.v_linear = nn.Linear(config.transformer_size, config.num_heads * config.v_size)

        init_wt_normal(self.q_linear.weight, (config.transformer_size + config.k_size))
        init_wt_normal(self.k_linear.weight, (config.transformer_size + config.k_size))

        init_wt_normal(self.v_linear.weight, (config.transformer_size + config.v_size))

        self.attention = ScaleDotProductAttention(temperature=np.power(config.k_size, 0.5), dropout=config.dropout)

        # Applies Layer Normalization over a mini-batch of inputs
        self.layer_norm = nn.LayerNorm(config.transformer_size)

        self.fc = nn.Linear(config.num_heads * config.v_size, config.transformer_size)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, max_len, _]
            k: [batch_size, max_len, _]
            v: [batch_size, max_len, _]
            mask: [batch_size, max_len, ]
        """
        residual = q

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.q_linear(q).view(sz_b, len_q, self.num_heads, self.k_size) #[16, 35, 8 * 32]
        k = self.k_linear(k).view(sz_b, len_k, self.num_heads, self.k_size)
        v = self.v_linear(v).view(sz_b, len_v, self.num_heads, self.v_size)
        #  print('q size: ', q.shape) # [128, 50, 8, 64]
        #  print('k size: ', k.shape)
        #  print('v size: ', v.shape)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.k_size) # # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.k_size)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.v_size)
        #  print('q size: ', q.shape) # [1024, 50, 64]
        #  print('k size: ', k.shape)
        #  print('v size: ', v.shape)

        mask = mask.repeat(self.num_heads, 1, 1) # (n*b) x .. x ..
        output, sdp_attn_weight = self.attention(q, k, v, mask=mask)

        output = output.view(self.num_heads, sz_b, len_q, self.v_size)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.fc(output)
        output = self.dropout(output)

        output = self.layer_norm(output + residual)

        return output, sdp_attn_weight


class PositionwiseFeedForward(nn.Module):
    """A two feed forward layer module."""
    def __init__(self,
                config):
        super().__init__()

        self.cnn1 = nn.Conv1d(config.transformer_size, config.inner_hidden_size, 1) # position wise
        self.cnn2 = nn.Conv1d(config.inner_hidden_size, config.transformer_size, 1) # position wise

        self.layer_norm = nn.LayerNorm(config.transformer_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input):
        """
        Args:
            input: [batch_size, max_len, transformer_size]
        Return: [batch_size, max_len, transformer_size]
        """

        #  print('sub_layer pff input: ', input.shape)
        residual = input
        output = input.transpose(1, 2)

        output = self.cnn1(output)
        output = torch.relu(output)
        output = self.cnn2(output)

        output = output.transpose(1, 2)

        output = self.dropout(output)

        output = self.layer_norm(output + residual)
        #  print('sub_layer pff output: ', output.shape)

        return output
