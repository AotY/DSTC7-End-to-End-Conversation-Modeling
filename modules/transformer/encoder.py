#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Define the transformer model.
"""
import torch.nn as nn
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
            embedding,
            has_position=True):

        super(Encoder, self).__init__()

        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim
        self.has_position = has_position

        self.dropout = nn.Dropout(config.dropout)

        if has_position:
            n_position = config.c_max_len + 1
            self.pos_embedding = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(n_position,
                                            self.embedding_size,
                                            padid=PAD_ID),
                freeze=True
            )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.t_num_layers)]
        )

    def forward(self,
                enc_inputs,
                enc_inputs_pos=None,
                return_attns=False):
        """
        Args:
            enc_inputs: [batch_size, max_len]
            enc_inputs_pos: [batch_size, max_len]
        return:
            enc_output: [batch_size, max_len, transformer_size]
        """
        #  print('enc_inputs: ', enc_inputs.shape)
        #  print('enc_inputs_pos: ', enc_inputs_pos.shape)
        if return_attns:
            enc_slf_attn_list = list()

        # -- Prepare masks
        attn_mask = get_attn_key_pad_mask(k=enc_inputs, q=enc_inputs, padid=PAD_ID)
        #  print('attn_mask: ', attn_mask)

        non_pad_mask = get_non_pad_mask(enc_inputs, PAD_ID)
        #  print('non_pad_mask: ', non_pad_mask)

        enc_embedded = self.embedding(enc_inputs) # [batch_size, max_len, embedding_size]

        # dropout, default: 0.1
        enc_embedded = self.dropout(enc_embedded)

        if self.has_position:
            pos_embedded = self.pos_embedding(enc_inputs_pos).to(enc_inputs.device)
            #  print('embedded: ', embedded.shape)
            #  print('pos_embedded: ', pos_embedded.shape)
            enc_embedded = enc_embedded + pos_embedded

            #  print('enc_embedded shape: ', enc_embedded.shape) # [b, max_len, embedding_size]

        enc_output = enc_embedded
        for layer in self.layer_stack:
            enc_output, en_slf_attn = layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask
            )

            if return_attns:
                enc_slf_attn_list.append(en_slf_attn)

        #  print('enc_output shape: ', enc_output.shape)
        if return_attns:
            return enc_output, enc_slf_attn_list

        # [batch_size, max_len, transformer_size]
        return enc_output
