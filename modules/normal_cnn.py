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


        # conv2d 120 -> 1
        kernel_sizes = [(4, 512), (3, 512), (3, 256), (2, 256), (4, 512)]
        channels = [512, 256, 256, 512, 512]
        strides = [(2, 1), (2, 1), (2, 1), (2, 1), (1, 1)]

        for channel, kernel_size, stride in zip(channels, kernel_sizes, strides):
            self.conv2ds.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    stride=stride
                )
            )

        #  for channel in channels:
            #  self.bn2s.append(
                #  nn.BatchNorm2d(channel)
            #  )

        self. maxpool2d = nn.MaxPool2d(kernel_size=(4, 1))

        self.out_linear = nn.Linear(channels[-1], config.hidden_size)
        init_linear_wt(self.out_linear)

    def forward(self, inputs, lengths=None):
        """
        args:
            inputs: [max_len, batch_size]
        return:
            [1, batch_size, hidden_size]
        """
        embedded = self.embedding(inputs)  # [max_len, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # [batch_size, 1, max_len, embedding_size]
        embedded = embedded.transpose(0, 1).unsqueeze(1)
        #  print(embedded.shape)

        # conv
        output = embedded
        #  for conv2d, bn2 in zip(self.conv2ds, self.bn2s):
        for conv2d in self.conv2ds:
            output = conv2d(output)
            #  output = bn2(output)
            output = F.relu(output)
            output = output.transpose(1, 3)
            #  print('output: ', output.shape)

        # [batch_size, 1, 1, 1024]
        output = self.maxpool2d(output)

        #  print('cnn output: ', output.shape)
        output = output.squeeze(2).transpose(0, 1)
        output = self.out_linear(output)  # [1, batch_size, hidden_size]

        return output, None
