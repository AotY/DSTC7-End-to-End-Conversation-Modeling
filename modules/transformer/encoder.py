#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Define the transformer model.
"""
import torch
import torch.nn as nn
import numpy as np
from modules.transformer.layers import EncoderLayer
from modules.transformer.utils import get_sinusoid_encoding_table
from modules.transformer.utils import get_attn_key_pad_mask
from modules.transformer.utils import get_non_pad_mask

from misc.vocab import PAD_ID

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            config,
            embedding):

        super(Encoder, self).__init__()

        n_position = config.c_max_len + 1

        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.pos_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position,
                                        config.embedding_size,
                                        padid=PAD_ID),
            freeze=True
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, enc_inputs, enc_inptus_pos, return_attns=False):
        """
        Args:
            enc_inputs: [batch_size, max_len]
            enc_inptus_pos: [batch_size, max_len]
        return:
            enc_output: [batch_size, max_len, transformer_size]
        """
        enc_attn_list = list()

        # -- Prepare masks
        attn_mask = get_attn_key_pad_mask(k=enc_inputs, q=enc_inputs, padid=self.padid)

        non_pad_mask = get_non_pad_mask(enc_inputs, self.padid)

        embedded = self.embedding(enc_inputs) # [batch_size, max_len, embedding_size]

        pos_embedded = self.pos_embedding(enc_inptus_pos).to(enc_inputs.device)

        print('embedded: ', embedded.shape)
        print('pos_embedded: ', pos_embedded.shape)
        enc_embedded = embedded + pos_embedded
        print('enc_embedded shape: ', enc_embedded.shape) # [b, max_len, embedding_size]

        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(
                enc_embedded,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask)

            if return_attns:
                enc_attn_list.append(enc_attn)

        print('enc_output shape: ', enc_output.shape)
        if return_attns:
            return enc_output, enc_attn_list

        # [batch_size, max_len, transformer_size]
        return enc_output

