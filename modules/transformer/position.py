#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Positional Embedding
"""
import math
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, 
            embedding_size,
            max_len=512,
            device='cuda'):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size).float().to(device)
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1).to(device)
        div_term = (torch.arange(0, embedding_size, 2, device=device).float() * -(math.log(10000.0) / embedding_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        input: [max_len, batch_size] 
        """
        return self.pe[:, :input.size(1)]

