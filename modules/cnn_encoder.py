#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Implementation of "Convolutional Sequence to Sequence Learning"
https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/encoders/cnn_encoder.py
"""

import torch
import torch.nn as nn

from modules.cnn_factory import shape_transform, StackedCNN

SCALE_WEIGHT = 0.5 ** 0.5


class CNNEncoder(nn.Module):
    """
    Encoder built on CNN based on
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self,
                 config,
                 embedding):
        super(CNNEncoder, self).__init__()

        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.linear = nn.Linear(self.embedding_size, config.hidden_size)

        self.cnn = StackedCNN(
            config.encoder_num_layers,
            config.hidden_size,
            config.cnn_kernel_width,
            config.dropout
        )

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input: [max_len, batch_size, n_feat]
            lengths: [batch_size]
        return:

        """
        embedded = self.embedding(input)
        # s_len, batch, emb_dim = embedded.size()

        # [batch_size, max_len, embedding_size]
        embedded = embedded.transpose(0, 1).contiguous()
        # [batch_size * max_len, embedding_size]
        embedded_reshape = embedded.view(
            embedded.size(0) * embedded.size(1), -1)
        # [batch_size * max_len, hidden_size]
        embedded_remap = self.linear(embedded_reshape)
        # [batch_size, max_len, hidden_size]
        embedded_remap = embedded_remap.view(
            embedded.size(0), embedded.size(1), -1)

        # [batch_size, hidden_size, max_len, 1]
        embedded_remap = shape_transform(embedded_remap)

        # []
        output = self.cnn(embedded_remap)

        print('embedded_remap: ', embedded_remap.shape)
        print('output: ', output.shape)
        # [hidden_size, batch_size, max_len]
        # []
        return embedded_remap.squeeze(3).transpose(0, 1).contiguous(), \
            output.squeeze(3).transpose(0, 1).contiguous(), \
            lengths
