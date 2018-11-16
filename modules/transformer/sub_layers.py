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
                 n_head,
                 model_dim,
                 k_dim,
                 v_dim,
                 dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.q_linear = nn.Linear(model_dim, n_head * k_dim)
        self.k_linear = nn.Linear(model_dim, n_head * k_dim)

        self.v_linear = nn.Linear(model_dim, n_head * v_dim)

        init_wt_normal(self.q_linear.weight)
        init_wt_normal(self.k_linear.weight)
        init_wt_normal(self.v_linear.weight)

        self.sdp_attn = ScaleDotProductAttention(temperature=np.power(k_dim, 0.5))

        # Applies Layer Normalization over a mini-batch of inputs
        self.layer_norm = nn.LayerNorm(model_dim)

        self.fc = nn.Linear(n_head * v_dim, model_dim)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: [batch_size, max_len, _]
            k: [batch_size, max_len, _]
            v: [batch_size, max_len, _]
            mask: []
        """
        residual = q

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.q_linear(q).view(sz_b, len_q, self.n_head, self.k_dim) #[16, 35, 8 * 32]
        k = self.k_linear(k).view(sz_b, len_k, self.n_head, self.k_dim)
        v = self.v_linear(v).view(sz_b, len_v, self.n_head, self.v_dim)
        #  print('q size: ', q.shape)
        #  print('k size: ', k.shape)
        #  print('v size: ', v.shape)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.k_dim) # # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.k_dim)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.v_dim)
        #  print('q size: ', q.shape)
        #  print('k size: ', k.shape)
        #  print('v size: ', v.shape)

        mask = mask.repeat(self.n_head, 1, 1) # (n*b) x .. x ..
 #[16, 35, 8 * 32]
        output, sdp_attn_weight = self.sdp_attn(q, k, v, mask=mask)

        output = output.view(self.n_head, sz_b, len_q, self.v_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.fc(output)
        output = self.dropout(output)

        output = self.layer_norm(output + residual)

        return output, sdp_attn_weight


class PositionwiseFeedForward(nn.Module):
    """A two feed forward layer module."""
    def __init__(self,
                 in_dim,
                 hid_dim,
                 dropout=0.1):
        super().__init__()

        self.cnn1 = nn.Conv1d(in_dim, hid_dim, 1) # position wise
        self.cnn2 = nn.Conv1d(hid_dim, in_dim, 1) # position wise

        self.layer_norm = nn.LayerNorm(in_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """
        Args:
            input:
        """

        residual = input
        output = input.transpose(1, 2)

        output = self.cnn1(output)
        output = torch.relu(output)
        output = self.cnn2(output)

        output = output.transpose(1, 2)

        output = self.dropout(output)

        output = self.layer_norm(output + residual)

        return output





