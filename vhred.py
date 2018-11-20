#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues.
https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/master/model/models.py
"""

import torch
import torch.nn as nn
import numpy as np

from modules.utils import to_device
from modules.encoder import EncoderRNN, ContextRNN
from modules.decoder import DecoderRNN
from modules.feedforward import FeedForward
from modules.rnn_cells import StackedLSTMCell, StackedGRUCell

from misc.bow import to_bow, bag_of_words_loss
from misc.probability import normal_logpdf, normal_kl_div
from misc.pad import pad
from misc.vocab import EOS_ID


class VHRED(nn.Module):
    def __init__(self, config):
        super(VHRED, self).__init__()

        self.config = config
        self.encoder = EncoderRNN(config.vocab_size,
                                  config.embedding_size,
                                  config.encoder_hidden_size,
                                  nn.LSTM if config.rnn == 'LSTM' else nn.GRU,
                                  config.num_layers,
                                  config.bidirectional,
                                  config.dropout)

        context_input_size = (config.num_layers
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

        self.context2decoder = FeedForward(config.context_size + config.z_sent_size,
                                           config.num_layers * config.decoder_hidden_size,
                                           num_layers=1,
                                           activation=config.activation)

        self.softplus = nn.Softplus()

        self.prior_h = FeedForward(config.context_size,
                                   config.context_size,
                                   num_layers=2,
                                   hidden_size=config.context_size,
                                   activation=config.activation)

        self.prior_mu = nn.Linear(config.context_size,
                                  config.z_sent_size)

        self.prior_var = nn.Linear(config.context_size,
                                   config.z_sent_size)

        self.posterior_h = FeedForward(
            config.encoder_hidden_size * self.encoder.num_directions * \
            config.num_layers + config.context_size,
            config.context_size,
            num_layers=2,
            hidden_size=config.context_size,
            activation=config.activation
        )
        self.posterior_mu = nn.Linear(config.context_size,
                                      config.z_sent_size)

        self.posterior_var = nn.Linear(config.context_size,
                                       config.z_sent_size)
        if config.share_embedding:
            self.decoder.embedding = self.encoder.embedding

        if config.bow:
            self.bow_h = FeedForward(config.z_sent_size,
                                            config.decoder_hidden_size,
                                            num_layers=1,
                                            hidden_size=config.decoder_hidden_size,
                                            activation=config.activation)

            self.bow_predict = nn.Linear(
                config.decoder_hidden_size, config.vocab_size)

    def prior(self, context_outputs):
        # Context dependent prior
        h_prior = self.prior_h(context_outputs)
        mu_prior = self.prior_mu(h_prior)
        var_prior = self.softplus(self.prior_var(h_prior))
        return mu_prior, var_prior

    def posterior(self, context_outputs, encoder_hidden):
        h_posterior = self.posterior_h(
            torch.cat([context_outputs, encoder_hidden], 1))
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = self.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def compute_bow_loss(self, target_conversations):
        target_bow = np.stack([to_bow(sent, self.config.vocab_size)
                               for conv in target_conversations for sent in conv], axis=0)
        target_bow = to_device(torch.FloatTensor(target_bow))
        bow_logits = self.bow_predict(self.bow_h(self.z_sent))
        bow_loss = bag_of_words_loss(bow_logits, target_bow)
        return bow_loss

    def forward(self,
                input_sentences,
                input_sentence_length,
                input_conversation_length,
                target_sentences,
                decode=False):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences + batch_size, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        batch_size = input_conversation_length.size(0)
        num_sentences = input_sentences.size(0) - batch_size
        c_max_len = input_conversation_length.data.max().item()

        # encoder_outputs: [num_sentences + batch_size, max_source_length, hidden_size]
        # encoder_hidden: [num_layers * direction, num_sentences + batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input_sentences,
                                                       input_sentence_length)

        # encoder_hidden: [num_sentences + batch_size, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(
            1, 0).contiguous().view(num_sentences + batch_size, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_device(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1] + 1)), 0)
        # encoder_hidden: [batch_size, c_max_len + 1, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l + 1), c_max_len + 1)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        # encoder_hidden_inference: [batch_size, c_max_len, num_layers * direction * hidden_size]
        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat(
            [encoder_hidden_inference[i, :l, :] for i, l in enumerate(input_conversation_length.data)])

        # encoder_hidden_input: [batch_size, c_max_len, num_layers * direction * hidden_size]
        encoder_hidden_input = encoder_hidden[:, :-1, :]

        # context_outputs: [batch_size, c_max_len, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden_input,
                                                                    input_conversation_length)
        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])

        mu_prior, var_prior = self.prior(context_outputs)
        eps = to_device(torch.randn((num_sentences, self.config.z_sent_size)))
        if not decode:
            mu_posterior, var_posterior = self.posterior(
                context_outputs, encoder_hidden_inference_flat)
            z_sent = mu_posterior + torch.sqrt(var_posterior) * eps
            log_q_zx = normal_logpdf(z_sent, mu_posterior, var_posterior).sum()

            log_p_z = normal_logpdf(z_sent, mu_prior, var_prior).sum()
            # kl_div: [num_sentences]
            kl_div = normal_kl_div(mu_posterior, var_posterior,
                                   mu_prior, var_prior)
            kl_div = torch.sum(kl_div)
        else:
            z_sent = mu_prior + torch.sqrt(var_prior) * eps
            kl_div = None
            log_p_z = normal_logpdf(z_sent, mu_prior, var_prior).sum()
            log_q_zx = None

        self.z_sent = z_sent
        latent_context = torch.cat([context_outputs, z_sent], 1)
        decoder_init = self.context2decoder(latent_context)
        decoder_init = decoder_init.view(-1,
                                         self.decoder.num_layers,
                                         self.decoder.hidden_size)
        decoder_init = decoder_init.transpose(1, 0).contiguous()

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:

            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)

            return decoder_outputs, kl_div, log_p_z, log_q_zx

        else:
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(
                init_h=decoder_init)

            return prediction, kl_div, log_p_z, log_q_zx

    def generate(self, context, input_sentence_length, n_context):
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
                                                           input_sentence_length[:, i])

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

            mu_prior, var_prior = self.prior(context_outputs)
            eps = to_device(torch.randn((batch_size, self.config.z_sent_size)))
            z_sent = mu_prior + torch.sqrt(var_prior) * eps

            latent_context = torch.cat([context_outputs, z_sent], 1)
            decoder_init = self.context2decoder(latent_context)
            decoder_init = decoder_init.view(
                self.decoder.num_layers, -1, self.decoder.hidden_size)

            if self.config.sample:
                prediction = self.decoder(None, decoder_init)
                p = prediction.data.cpu().numpy()
                length = torch.from_numpy(np.where(p == EOS_ID)[1])
            else:
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
