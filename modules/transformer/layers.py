#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
From Attention is All You Need.
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Layers.py
"""

import torch
import torch.nn as nn
from modules.transformer.sub_layers import MultiHeadAttention
from modules.transformer.sub_layers import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """compose with two layer."""
    def __init__(self,
                 model_dim=512,
                 inner_dim=1024,
                 n_head=8,
                 k_dim=64,
                 v_dim=64,
                 dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mh_attn = MultiHeadAttention(
            n_head,
            model_dim,
            k_dim,
            v_dim,
            dropout=dropout
        )

        self.pos_ffn = PositionwiseFeedForward(
            model_dim,
            inner_dim,
            dropout=dropout
        )

    def forward(self,
                enc_input,
                non_pad_mask=None,
                attn_mask=None):
        """
        Args:
            enc_input: []
            non_pad_mask: []
            attn_mask: []
        """
        #  print('enc_input: ', enc_input.shape)

        enc_output, enc_attn = self.mh_attn(
            enc_input,
            enc_input,
            enc_input,
            mask=attn_mask
        )

        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        enc_output *= non_pad_mask

        return enc_output, enc_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, model_dim, inner_dim, n_head, k_dim, v_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, model_dim, k_dim, v_dim, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, model_dim, k_dim, v_dim, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(model_dim, inner_dim, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn




