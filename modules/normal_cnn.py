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

        self.convds = nn.ModuleList()
        self.maxpools = nn.ModuleList()

        # conv2d
        kernel_sizes = [3, 3, 4, 4, 5, 4]
        self.conv2d1 = nn.Conv2d(in_channels=1,
                               out_channels=512,
                               kernel_size=(kernel_sizes[0], self.embedding_size),
                               stride=1)
        # output: [batch_size, output_channels, max_len - 3 + 1, 1]  35 - 3 + 1

        self.conv2d2 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(kernel_sizes[1], 512),
                               stride=1)
        # output: [batch_size, output_channels, -3 + 1, 1]

        self.conv2d3 = nn.Conv2d(in_channels=1,
                               out_channels=128,
                               kernel_size=(kernel_sizes[2], 256),
                               stride=1)

        self.conv2d4 = nn.Conv2d(in_channels=1,
                               out_channels=256,
                               kernel_size=(kernel_sizes[3], 128),
                               stride=1)

        self.conv2d5 = nn.Conv2d(in_channels=1,
                               out_channels=512,
                               kernel_size=(kernel_sizes[4], 256),
                               stride=1)

        self.conv2d6 = nn.Conv2d(in_channels=1,
                               out_channels=1024,
                               kernel_size=(kernel_sizes[5], 512),
                               stride=1)

        self.maxpool3 = nn.MaxPool2d(
            kernel_size=(3, 1),
            stride=1
        )

        self.maxpool4 = nn.MaxPool2d(
            kernel_size=(4, 1),
            stride=1
        )

        self.maxpool5 = nn.MaxPool2d(
            kernel_size=(5, 1),
            stride=1
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
        embedded = self.embedding(inputs) # [max_len, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # [batch_size, 1, max_len, embedding_size]
        embedded = embedded.transpose(0, 1).unsqueeze(1)
        #  print(embedded.shape)

        # conv
        conv2d1_output = self.conv2d1(embedded) # [batch_size, 512, 33, 1]
        conv2d1_output = F.relu(conv2d1_output)
        conv2d1_output = self.maxpool3(conv2d1_output) # [batch_size, 512, 31, 1]
        print('conv2d1_output: ', conv2d1_output.shape)

        conv2d2_input = conv2d1_output.transpose(1, 3) # [batch_size, 1, 31, 512]
        conv2d2_output = self.conv2d2(conv2d2_input) # [batch_size, 256, 29, 1]
        conv2d2_output = F.relu(conv2d2_output)
        conv2d2_output = self.maxpool3(conv2d2_output) # [batch_size, 256, 27, 1]
        print('conv2d2_output: ', conv2d2_output.shape)

        conv2d3_input = conv2d2_output.transpose(1, 3) # [batch_size, 1, 27, 256]
        conv2d3_output = self.conv2d3(conv2d3_input) # [batch_size, 128, 24, 1]
        conv2d3_output = F.relu(conv2d3_output)
        conv2d3_output = self.maxpool4(conv2d3_output) # [batch_size, 128, 21, 1]
        print('conv2d3_output: ', conv2d3_output.shape)

        conv2d4_input = conv2d3_output.transpose(1, 3) # [batch_size, 1, 21, 128]
        conv2d4_output = self.conv2d4(conv2d4_input) # [batch_size, 256, 18, 1]
        conv2d4_output = F.relu(conv2d4_output)
        conv2d4_output = self.maxpool4(conv2d4_output) # [batch_size, 256, 15, 1]
        print('conv2d4_output: ', conv2d4_output.shape)

        conv2d5_input = conv2d4_output.transpose(1, 3) # [batch_size, 1, 15, 256]
        conv2d5_output = self.conv2d5(conv2d5_input) # [batch_size, 512, 11, 1]
        conv2d5_output = F.relu(conv2d5_output)
        conv2d5_output = self.maxpool5(conv2d5_output) # [batch_size, 512, 8, 1]
        print('conv2d5_output: ', conv2d5_output.shape)

        conv2d6_input = conv2d5_output.transpose(1, 3) # [batch_size, 1, 8, 512]
        conv2d6_output = self.conv2d6(conv2d6_input) # [batch_size, 1024, 5, 1]
        conv2d6_output = F.relu(conv2d6_output)
        conv2d6_output = self.maxpool5(conv2d6_output) # [batch_size, 1024, 1, 1]
        print('conv2d6_output: ', conv2d6_output.shape)

        output = conv2d6_output.squeeze(3).squeeze(2).unsqueeze(0)
        output = self.out_linear(output) # [1, batch_size, hidden_size]
        print('output: ', output.shape)

        return output, None

