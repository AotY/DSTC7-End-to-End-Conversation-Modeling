#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
HRED Model
"""
import torch
import torch.nn as nn
from modules.utils import to_device
from modules.encoder import EncoderRNN, ContextRNN
from modules.decoder import DecoderRNN
from modules.feedforward import FeedForward
from modules.rnn_cells import StackedLSTMCell, StackedGRUCell

from misc.pad import pad


class HRED(nn.Module):
    def __init__(self, config):
        super(HRED, self).__init__()

        self.config = config
        self.encoder = EncoderRNN(config.vocab_size,
                                  config.embedding_size,
                                  config.encoder_hidden_size,
                                  nn.LSTM if config.rnn == 'LSTM' else nn.GRU,
                                  config.encoder_num_layers,
                                  config.bidirectional,
                                  config.dropout)

        context_input_size = (config.encoder_num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)

        self.context_encoder = ContextRNN(context_input_size,
                                          config.context_size,
                                          nn.LSTM if config.rnn == 'LSTM' else nn.GRU,
                                          config.encoder_num_layers,
                                          config.dropout)

        self.decoder = DecoderRNN(config.vocab_size,
                                  config.embedding_size,
                                  config.decoder_hidden_size,
                                  StackedLSTMCell if config.rnn == 'LSTM' else StackedGRUCell,
                                  config.decoder_num_layers,
                                  config.dropout,
                                  config.word_drop,
                                  config.max_unroll,
                                  config.sample,
                                  config.temperature,
                                  config.beam_size)

        self.context2decoder = FeedForward(config.context_size,
                                           config.decoder_num_layers * config.decoder_hidden_size,
                                           num_layers=1,
                                           activation=config.activation)

        if config.share_embedding:
            self.decoder.embedding = self.encoder.embedding

    def forward(self,
                input_sentences,
                input_sentence_length,
                input_conversation_length,
                target_sentences,
                decode=False):
        """
        Args:
            input_sentences: [num_sentences, seq_len]
            target_sentences: [num_sentences, seq_len]
            input_sentence_length: [num_sentences]
            input_conversation_length: [batch_size]
        Return:
            decoder_outputs:
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        #  print(input_sentences.shape)
        #  print(input_sentence_length)
        #  print(input_conversation_length)

        num_sentences = input_sentences.size(0)
        c_max_len = input_conversation_length.data.max().item()

        # encoder_outputs: [num_sentences, max_source_length, hidden_size * direction]
        # encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input_sentences,
                                                       input_sentence_length)
        #  print(encoder_outputs.shape)
        #  print(encoder_hidden.shape)

        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_sentences, -1)
        #  print(encoder_hidden.shape)

        # pad and pack encoder_hidden
        tmp_lengths = torch.cat((to_device(input_conversation_length.data.new(1).zero_()), input_conversation_length[:-1]))
        start = torch.cumsum(tmp_lengths, 0) # [batch_size]

        # encoder_hidden: [batch_size, c_max_len, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l), c_max_len)
                                      for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0)
        #  print(encoder_hidden.shape)

        # context_outputs: [batch_size, c_max_len, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden,
                                                                    input_conversation_length)

        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])

        # project context_outputs to decoder init state
        decoder_init = self.context2decoder(context_outputs)

        # [num_layers, batch_size, hidden_size]
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:
            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)
            return decoder_outputs

        else:
            # decoder_outputs = self.decoder(target_sentences,
            #                                init_h=decoder_init,
            #                                decode=decode)
            # return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(
                init_h=decoder_init)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return prediction

    def generate(self, context, sentence_length, n_context):
        # context: [batch_size, n_context, seq_len]
        batch_size = context.size(0)
        # n_context = context.size(1)
        samples = []

        # Run for context
        context_hidden = None
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :],
                                                           sentence_length[:, i])

            encoder_hidden = encoder_hidden.transpose(
                1, 0).contiguous().view(batch_size, -1)
            # context_outputs: [batch_size, 1, context_hidden_size * direction]
            # context_hidden: [num_layers * direction, batch_size, context_hidden_size]
            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden,
                                                                        context_hidden)

        # Run for generation
        for j in range(self.config.n_sample_step):
            # context_outputs: [batch_size, context_hidden_size * direction]
            context_outputs = context_outputs.squeeze(1)
            decoder_init = self.context2decoder(context_outputs)
            decoder_init = decoder_init.view(
                self.decoder.num_layers, -1, self.decoder.hidden_size)

            prediction, final_score, length = self.decoder.beam_decode(
                init_h=decoder_init)
            # prediction: [batch_size, seq_len]
            prediction = prediction[:, 0, :]
            # length: [batch_size]
            length = [l[0] for l in length]
            length = to_device(torch.LongTensor(length))
            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction,
                                                           length)

            encoder_hidden = encoder_hidden.transpose(
                1, 0).contiguous().view(batch_size, -1)

            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden,
                                                                        context_hidden)

        samples = torch.stack(samples, 1)
        return samples
