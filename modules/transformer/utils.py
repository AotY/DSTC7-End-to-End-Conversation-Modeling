#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Utils for transformer model.
"""
import torch
import numpy as np


def get_non_pad_mask(input, padid):
    """
    input: [b, l] ?
    """
    assert input.dim() == 2
    return input.ne(padid).type(torch.float).unsqueeze(-1) # [max_len, batch_size, 1]

def get_sinusoid_encoding_table(n_position, embedding_size, padid=None):
    """Sinusoid position encoding table."""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / embedding_size)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_idx) for hid_idx in range(embedding_size)]

    sinusoid_table = np.array([get_posi_angle_vec(position) for position in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i + 1

    if padid is not None:
        # zero vector for padding dimension
        sinusoid_table[padid] = 0.

    return torch.tensor(sinusoid_table, dtype=torch.float)

def get_attn_key_pad_mask(k, q, padid):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = q.size(1)
    padding_mask = k.eq(padid)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(input):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = input.size()
    # Returns the upper triangular part of the matri
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=input.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
