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
from modules.transformer.layers import DecoderLayer
from modules.transformer.utils import get_sinusoid_encoding_table
from modules.transformer.utils import get_attn_key_pad_mask
from modules.transformer.utils import get_non_pad_mask
from modules.transformer.utils import get_subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            max_len,
            embedding,
            num_layers,
            num_head,
            k_dim,
            v_dim,
            model_dim,
            inner_dim,
            padid,
            dropout=0.1):

        super(Encoder, self).__init__()

        self.padid = padid
        n_position = max_len + 1

        self.embedding = embedding
        self.embedding_size = embedding.embedding_dim

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position,
                                        self.embedding_size,
                                        padid=padid),
            freeze=True
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(model_dim, inner_dim, num_head, k_dim, v_dim, dropout=dropout)
            for _ in range(num_layers)]
        )

    def forward(self, input, input_position, return_attns=False):
        """
        Args:
            input: [batch_size, max_len]
            input_position: [batch_size, max_len]
        return:
            enc_output: []
        """
        enc_attn_list = []

        # -- Prepare masks
        attn_mask = get_attn_key_pad_mask(k=input, q=input, padid=self.padid)

        non_pad_mask = get_non_pad_mask(input, self.padid)

        # -- Forward
        embedded = self.embedding(input)
        position_embedded = self.position_enc(input_position).to(input.device)

        enc_output = embedded + position_embedded
        #  print('enc_output shape: ', enc_output.shape) # [b, max_len, embedding_size]

        for enc_layer in self.layer_stack:
            enc_output, enc_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask)

            if return_attns:
                enc_attn_list += [enc_attn]

        #  print('enc_output shape: ', enc_output.shape)

        if return_attns:
            return enc_output, enc_attn_list

        return enc_output



class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            vocab_size,
            max_len,
            embedding_size,
            num_layers,
            num_head,
            k_dim,
            v_dim,
            model_dim,
            inner_dim,
            padid,
            dropout=0.1):

        super().__init__()
        n_position = max_len + 1

        self.embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=padid)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, embedding_size, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(model_dim, inner_dim, num_head, k_dim, v_dim, dropout=dropout)
            for _ in range(num_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.embedding(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                attn_mask=attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            vocab_size,
            max_len,
            embedding_size=512,
            model_dim=512,
            inner_dim=2048,
            num_layers=6,
            num_head=8,
            k_dim=64,
            v_dim=64,
            dropout=0.1,
            tied=True,
            embedding_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_size=embedding_size,
            model_dim=model_dim,
            inner_dim=inner_dim,
            num_layers=num_layers,
            num_head=num_head,
            k_dim=k_dim,
            v_dim=v_dim,
            dropout=dropout)

        self.decoder = Decoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embedding_size=embedding_size,
            model_dim=model_dim,
            inner_dim=inner_dim,
            num_layers=num_layers,
            num_head=num_head,
            k_dim=k_dim,
            v_dim=v_dim,
            dropout=dropout)

        self.output_linear = nn.Linear(model_dim, vocab_size, bias=False)
        nn.init.xavier_normal_(self.output_linear.weight)

        assert model_dim == embedding_size, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
        if tied:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.output_linear.weight = self.decoder.embedding.weight
            self.x_logit_scale = (model_dim ** -0.5)
        else:
            self.x_logit_scale = 1.

        if embedding_sharing:
            # Share the weight matrix between source & target word embeddings
            self.encoder.embedding.weight = self.decoder.embedding.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, _ = self.encoder(src_seq, src_pos)
        dec_output, _ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.output_linear(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
