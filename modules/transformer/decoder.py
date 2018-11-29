#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from modules.transformer.layers import DecoderLayer

from modules.transformer.utils import get_sinusoid_encoding_table
from modules.transformer.utils import get_attn_key_pad_mask
from modules.transformer.utils import get_non_pad_mask
from modules.transformer.utils import get_subsequent_mask

from misc.vocab import PAD_ID



class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            config,
            embedding):
        super().__init__()

        n_position = config.r_max_len + 1

        self.embedding = embedding

        self.pos_embedding = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(n_position, config.embedding_size, padding_idx=PAD_ID),
                freeze=True
            )

        self.layer_stack = nn.ModuleList([
            DecoderLayer(config) for _ in range(num_layers)]
        )

    def forward(self, 
            dec_inputs,
            dec_inputs_pos,
            enc_inputs,
            enc_outputs,
            return_attns=False):
        """
        args:
            dec_inputs: [batch_size, r_max_len]
            dec_inputs_pos: [batch_size, r_max_len]
            enc_inputs: [batch_size, c_max_len]
            enc_outputs: [batch_size, c_max_len, transformer_size]
        return: [batch_size, r_max_len, transformer_size]
        """

        dec_slf_attn_list, dec_enc_attn_list = list(), list()

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(dec_inputs)

        slf_attn_mask_subseq = get_subsequent_mask(dec_inputs)

        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=dec_inputs, seq_q=dec_inputs)

        attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=enc_inputs, seq_q=dec_inputs)

        # -- Forward
        dec_embedded = self.embedding(dec_inputs) + self.pos_embedding(dec_inputs_pos).to(dec_inputs_pos.device)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_embedded, enc_outputs,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list.append(dec_slf_attn)
                dec_enc_attn_list.append(dec_enc_attn)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return dec_output,



