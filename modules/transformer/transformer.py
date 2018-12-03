#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Transformer
"""
import torch
import torch.nn as nn

from modules.transformer.encoder import Encoder
from modules.transformer.decoder import Decoder


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            config,
            encoder_embedding,
            decoder_embedding):

        super().__init__()

        self.encoder = Encoder(
            config,
            encoder_embedding
        )

        self.decoder = Decoder(
            config,
            decoder_embedding
        )

        self.output_linear = nn.Linear(
            config.transformer_size,
            config.vocab_size,
            bias=False
        )
        nn.init.xavier_normal_(self.output_linear.weight)

        assert config.transformer_size == config.embedding_size, \
            'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if config.tied:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.output_linear.weight = self.decoder.embedding.weight
            self.x_logit_scale = (transformer_size ** -0.5)
        else:
            self.x_logit_scale = 1.

        if config.share_embedding:
            # Share the weight matrix between source & target word embeddings
            self.encoder.embedding.weight = self.decoder.embedding.weight

    def forward(self,
                enc_inputs,
                enc_inputs_pos,
                dec_inputs,
                dec_inputs_pos):
        """
        Args:
            enc_inputs: [batch_size, max_len]
            enc_inputs_pos: [batch_size, max_len]

            dec_inputs: [batch_size, max_len]
            dec_inputs_pos: [batch_size, max_len]

        return: [batch_size * max_len, vocab_size]
        """
        # [batch_size, max_len, transformer_size]
        enc_output, _ = self.encoder(enc_inputs, enc_inputs_pos)

        dec_inputs, dec_inputs_pos = dec_inputs[:, :-1], dec_inputs_pos[:, :-1]

        dec_output, _ = self.decoder(
            dec_inputs,
            dec_inputs_pos,
            enc_inputs,
            enc_output
        )

        # [batch_size, max_len, vocab_size]
        output = self.output_linear(dec_output) * self.x_logit_scale
        # [batch_size * max_len, vocab_size]
        return output.view(-1, output.size(2))
