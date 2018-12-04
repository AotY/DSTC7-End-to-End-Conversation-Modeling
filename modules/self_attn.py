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
from modules.utils import init_wt_unif, init_linear_wt
from modules.utils import init_gru_orth, init_lstm_orth
from modules.utils import sequence_mask


class SelfAttentive(nn.Module):
    def __init__(self,
                 config,
                 embedding,
                 attn_hops=4,
                 mlp_input_size=256,
                 mlp_output_size=512,
                 dropout=0.2):

        super(SelfAttentive, self).__init__()

        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.bidirection_num = 2 if config.bidirectional else 1
        self.hidden_size = config.hidden_size // self.bidirection_num

        self.attn_hops = attn_hops
        self.mlp_input_size = mlp_input_size

        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # rnn
        self.rnn = rnn_factory(
            config.rnn_type,
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout
        )

        if config.rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)

        self.Ws1 = nn.Parameter(torch.Tensor(1, mlp_input_size, self.bidirection_num * self.hidden_size))
        self.Ws2 = nn.Parameter(torch.Tensor(1, attn_hops, mlp_input_size))
        init_wt_unif(self.Ws1)
        init_wt_unif(self.Ws2)

        self.fc1 = nn.Linear(attn_hops * self.bidirection_num * self.hidden_size, mlp_output_size)
        init_linear_wt(self.fc1)


    def forward(self, inputs, lengths=None):
        """
        Args:
            inputs: [max_len, batch_size]
            length: [batch_size]
        return:
            outputs: [batch_size, mlp_output_size]
        """
        max_len, batch_size = inputs.size()

        if lengths is not None:
            lengths, sorted_indexes = torch.sort(lengths, dim=0, descending=True)
            _, restore_indexes = torch.sort(sorted_indexes, dim=0)
            inputs = inputs.transpose(0, 1)[sorted_indexes].transpose(0, 1)

        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        outputs, hidden_state = self.rnn(embedded) # outputs: [max_len, batch_size, hidden_size]

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = outputs.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()
            hidden_state = hidden_state.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        A = torch.tanh(torch.bmm(self.Ws1.repeat(batch_size, 1, 1), outputs.permute((1, 2, 0)).contiguous())) # [batch_size, mlp_input_size, max_len]
        A = torch.bmm(self.Ws2.repeat(batch_size, 1, 1), A) # [batch_size, attn_hops, max_len]

        # mask
        if lengths is not None:
            mask = sequence_mask(lengths, max_len=A.size(-1)) #mask: [batch_size, max_len]
            mask = mask.unsqueeze(1).repeat(1, self.attn_hops, 1)  # Make it broadcastable.
            A.data.masked_fill_(1 - mask, -float('inf'))

        A = F.softmax(A, dim=2) # [batch_size, attn_hops, max_len]

        M = torch.bmm(A, outputs.transpose(0, 1)) # [batch_size, attn_hops, hidden_size]
        M = M.view(batch_size, -1) # [batch_size, attn_hops * hidden_size]

        output = torch.relu(self.fc1(M)) # [batch_size, mlp_output_size]

        return output, hidden_state
