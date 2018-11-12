#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Define the transformer model.
"""
import torch
import troch.nn as nn
import numpy as np
from modules.transformer.layers import EncoderLayer
from modules.transformer.layers import DecoderLayer
from modules.transformer.ultils import get_sinusoid_encoding_table
from modules.transformer.ultils import get_attn_key_pad_mask
from modules.transformer.ultils import get_non_pad_mask
from modules.transformer.ultils import get_subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            vocab_size,
            max_len,
            word_vec_dim,
            n_layers,
            n_head,
            k_dim,
            v_dim,
            model_dim,
            inner_dim,
            padid,
            dropout=0.1):

        super(Encoder, self).__init__()

        n_position = max_len + 1

        self.embedding = nn.Embedding(
            vocab_size,
            word_vec_dim,
            padding_idx=padid
        )

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, word_vec_dim, padding_idx=0),
            freeze=True
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(model_dim, inner_dim, n_head, k_dim, v_dim, dropout=dropout)
            for _ in range(n_layers)]
        )

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.embedding(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output



class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            vocab_size,
            max_len,
            word_vec_dim,
            n_layers,
            n_head,
            k_dim,
            v_dim,
            model_dim,
            inner_dim,
            padid,
            dropout=0.1):

        super().__init__()
        n_position = max_len + 1

        self.embedding = nn.Embedding(
            vocab_size, word_vec_dim, padding_idx=padid)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, word_vec_dim, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(model_dim, inner_dim, n_head, k_dim, v_dim, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.embedding(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
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
            word_vec_dim=512,
            model_dim=512,
            inner_dim=2048,
            n_layers=6,
            n_head=8,
            k_dim=64,
            v_dim=64,
            dropout=0.1,
            tied=True,
            embedding_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            max_len=max_len,
            word_vec_dim=word_vec_dim,
            model_dim=model_dim,
            inner_dim=inner_dim,
            n_layers=n_layers,
            n_head=n_head,
            k_dim=k_dim,
            v_dim=v_dim,
            dropout=dropout)

        self.decoder = Decoder(
            vocab_size=vocab_size,
            max_len=max_len,
            word_vec_dim=word_vec_dim,
            model_dim=model_dim,
            inner_dim=inner_dim,
            n_layers=n_layers,
            n_head=n_head,
            k_dim=k_dim,
            v_dim=v_dim,
            dropout=dropout)

        self.output_linear = nn.Linear(model_dim, vocab_size, bias=False)
        nn.init.xavier_normal_(self.output_linear.weight)

        assert model_dim == word_vec_dim, \
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

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.output_linear(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
