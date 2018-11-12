#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
A Structured Self-attentive Sentence Embedding
https://arxiv.org/abs/1703.03130
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import rnn_factory
from modules.utils import init_wt_normal
from modules.utils import init_gru_orth, init_lstm_orth
from modules.utils import sequence_mask


class SelfAttentive(nn.Module):
    def __init__(self,
                 embedding,
                 rnn_type,
                 num_layers,
                 bidirectional,
                 hidden_size=512,
                 attn_hops=10,
                 mlp_input_size=256,
                 mlp_output_size=512,
                 dropout=0.5):

        super(SelfAttentive, self).__init__()

        self.embedding = embedding
        self.embedding_sizes = embedding.embedding_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.bidirection_num = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.bidirection_num

        self.attn_hops = attn_hops
        self.mlp_input_size = mlp_input_size

        # dropout
        self.dropout = nn.Dropout(dropout)

        # rnn
        self.rnn = rnn_factory(
            rnn_type,
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

        if rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        self.fc1 = nn.Linear(attn_hops * self.bidirection_num * self.hidden_size, mlp_output_size)

        self.Ws1 = nn.Parameter(torch.Tensor(1, mlp_input_size, self.bidirection_num * self.hidden_size))
        self.Ws2 = nn.Parameter(torch.Tensor(1, attn_hops, mlp_input_size))

        init_wt_normal(self.Ws1)
        init_wt_normal(self.Ws2)

    def forward(self, inputs, lengths):
        """
        Args:
            inputs: [max_len, batch_size]
        return:
            outputs: [1, batch_size, mlp_output_size]
        """
        max_len, batch_size = inputs.size()

        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        outputs, hidden_state = self.rnn(inputs) # outputs: [max_len, batch_size, hidden_size]

        A = F.tanh(torch.bmm(self.Ws1.repeat(batch_size, 1, 1), outputs.permute((1, 2, 0)).contiguous())) # [batch_size, mlp_input_size, max_len]
        A = torch.bmm(self.Ws2.repeat(batch_size, 1, 1), A) # [batch_size, attn_hops, max_len]

        # mask
        if lengths is not None:
            mask = sequence_mask(lengths, max_len=A.size(-1)) #mask: [batch_size, max_len)
            mask = mask.unsqueeze(1)  # Make it broadcastable. # [batch_size, 1, max_len]
            A.data.masked_fill_(1 - mask, -float('inf')) # [batch_size, 1, max_len]

        A = F.softmax(A, dim=2) # [batch_size, attn_hops, max_len]

        M = torch.bmm(A, outputs.transpose(0, 1)) # [batch_size, attn_hops, hidden_size]
        M = M.view(batch_size, -1) # [batch_size, attn_hops * hidden_size]

        outputs = F.relu(self.fc1(M)) # [batch_size, mlp_output_size]

        outputs = outputs.unsqueeze(0).contiguous()

        return outputs, hidden_state
