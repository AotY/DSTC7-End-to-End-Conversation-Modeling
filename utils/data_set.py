# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import logging
import numpy as np

import os
import sys
import logging
import argparse

import torch

from vocab import Vocab
from vocab import PAD, SOS, EOS, UNK


class Seq2seqDataSet:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 path_conversations,
                 path_responses,

                 dialog_encoder_vocab_size=8e4 + 4,
                 dialog_encoder_max_length=50,
                 dialog_encoder_vocab=None,

                 dialog_decoder_vocab_size=8e4 + 4,
                 dialog_decoder_max_length=50,
                 dialog_decoder_vocab=None,

                 test_split=0.2,  # how many hold out as vali data
                 device=None,
                 logger=None
                 ):

        self.dialog_encoder_vocab_size = dialog_encoder_vocab_size
        self.dialog_encoder_max_length = dialog_encoder_max_length
        self.dialog_encoder_vocab = dialog_encoder_vocab

        self.dialog_decoder_vocab_size = dialog_decoder_vocab_size
        self.dialog_decoder_max_length = dialog_decoder_max_length
        self.dialog_decoder_vocab = dialog_decoder_vocab

        self.device = device
        self.logger = logger
        self.read_txt(path_conversations, path_responses, test_split)

    def read_txt(self, path_conversations, path_responses, test_split):
        print('loading data from txt files...')
        # load source-target pairs, tokenized

        seqs = dict()
        for k, path in [('conversation', path_conversations), ('response', path_responses)]:
            seqs[k] = []

            with open(path, encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                seq = []
                for c in line.strip('\n').strip().split(' '):
                    i = int(c)
                    seq.append(i)

                if k == 'conversation':
                    seqs[k].append(seq[-min(self.dialog_encoder_max_length - 2, len(seq)):])
                elif k == 'response':
                    seqs[k].append(seq[-min(self.dialog_decoder_max_length - 2, len(seq)):])

        self.pairs = list(zip(seqs['conversation'], seqs['response']))

        # train-test split
        np.random.shuffle(self.pairs)
        self.n_train = int(len(self.pairs) * (1. - test_split))

        self.i_sample_range = {
            'train': (0, self.n_train),
            'test': (self.n_train, len(self.pairs)),
        }
        self.i_sample = dict()

        self.reset()

    def reset(self):
        for task in self.i_sample_range:
            self.i_sample[task] = self.i_sample_range[task][0]

    def all_loaded(self, task):
        return self.i_sample[task] == self.i_sample_range[task][1]

    def load_data(self, task, max_num_sample_loaded=None):
        '''

        :param task: train or test
        :param max_num_sample_loaded: similar to
        :return: encoder_input_data, decoder_input_data, decoder_target_data, source_texts, target_texts
        '''

        i_sample = self.i_sample[task]

        if max_num_sample_loaded is None:
            max_num_sample_loaded = self.i_sample_range[task][1] - i_sample

        i_sample_next = min(i_sample + max_num_sample_loaded, self.i_sample_range[task][1])

        num_samples = i_sample_next - i_sample

        self.i_sample[task] = i_sample_next

        self.logger.info('building %s data from %i to %i' % (task, i_sample, i_sample_next))

        encoder_input_data = np.zeros((num_samples, self.dialog_encoder_max_length))
        encoder_input_lengths = np.zeros((num_samples, ))

        decoder_input_data = np.zeros((num_samples, self.dialog_decoder_max_length))
        decoder_input_lengths = np.zeros((num_samples, ))

        # decoder_target_data = np.zeros((num_samples, self.max_seq_len, self.vocab_size + 1))  # +1 as mask_zero
        decoder_target_data = np.zeros((num_samples, self.dialog_encoder_max_length, self.dialog_decoder_vocab_size))

        conversation_texts = []
        response_texts = []

        for i in range(num_samples):

            seq_conversation, seq_response = self.pairs[i_sample + i]

            if not bool(seq_response) or not bool(seq_conversation):
                continue

            if seq_response[-1] != EOS:
                seq_response.append(EOS)

            conversation_texts.append(' '.join([self.dialog_encoder_vocab.words_to_id(j) for j in seq_conversation]))
            response_texts.append(' '.join([self.dialog_decoder_vocab.words_to_id(j) for j in seq_response]))

            for t, token_index in enumerate(seq_conversation):
                encoder_input_data[i, t] = token_index

            decoder_input_data[i, 0] = SOS
            for t, token_index in enumerate(seq_response):
                decoder_input_data[i, t + 1] = token_index
                decoder_target_data[i, t, token_index] = 1.

        encoder_input_data = torch.tensor(encoder_input_data, dtype=torch.long, device=self.device)
        decoder_input_data = torch.tensor(decoder_input_data, dtype=torch.long, device=self.device)
        decoder_target_data = torch.tensor(decoder_target_data, dtype=torch.long, device=self.device)

        return encoder_input_data, decoder_input_data, decoder_target_data, conversation_texts, response_texts


class KDataSet:
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
