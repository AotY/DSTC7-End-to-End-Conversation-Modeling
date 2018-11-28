#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Fact encoder,
[max_len, batch_size, embedding_size] -> [1, batch_size, hidden_size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import init_linear_wt


class NormalCNN(nn.Module):
    def __init__(self,
                 config,
                 embedding,
                 ):
        super(NormalCNN, self).__init__()

        # embedding
        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.dropout = nn.Dropout(config.dropout)

        self.conv2ds = nn.ModuleList()
        self.bn2s = nn.ModuleList()
        self.maxpool2ds = nn.ModuleList()

        # conv2d
        kernel_sizes = [(3, 512), (3, 512), (4, 256),
                        (4, 128), (5, 256), (4, 512)]
        channels = [512, 256, 128, 256, 512, 1024]

        for channel, kernel_size in zip(channels, kernel_sizes):
            self.conv2ds.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    stride=1
                )
            )

        kernel_sizes = [(3, 1), (3, 1), (4, 1),
                        (4, 1), (4, 1), (5, 1)]
        for kernel_size in kernel_sizes:
            self.maxpool2ds.append(
                nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=1
                )
            )

        for channel in channels:
            self.bn2s.append(
                nn.BatchNorm2d(channel)
            )


        self.out_linear = nn.Linear(1024, config.hidden_size)
        init_linear_wt(self.out_linear)

    def forward(self, inputs, lengths=None):
        """
        args:
            inputs: [max_len, batch_size]
        return:
            [1, batch_size, hidden_size]
        """
        embedded = self.embedding(
            inputs)  # [max_len, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # [batch_size, 1, max_len, embedding_size]
        embedded = embedded.transpose(0, 1).unsqueeze(1)
        #  print(embedded.shape)

        # conv
        output = embedded
        for conv2d, bn2, maxpool2d in zip(self.conv2ds, self.bn2s, self.maxpool2ds):
            output = conv2d(output)
            output = bn2(output)
            output = F.relu(output)
            output = maxpool2d(output)
            output = output.transpose(1, 3)

        # [batch_size, 1, 1, 1024]

        output = output.squeeze(2).transpose(0, 1)
        output = self.out_linear(output)  # [1, batch_size, hidden_size]
        #  print('output: ', output.shape)

        return output, None

