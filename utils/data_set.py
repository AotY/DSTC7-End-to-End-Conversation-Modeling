# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import logging
import numpy as np

import os
import sys
import logging
import argparse

from vocab import Vocab
from vocab import PAD, SOS, EOS, UNK


class Seq2seqDataSet:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 path_conversations, path_responses, path_vocab,
                 max_seq_len=50,
                 test_split=0.2,  # how many hold out as vali data
                 vocab_size=8e4 + 4,
                 vocab=None
                 ):

        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.vocab_size = vocab_size
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

                seqs[k].append(seq[-min(self.max_seq_len - 2, len(seq)):])

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

        print('building %s data from %i to %i' % (task, i_sample, i_sample_next))

        encoder_input_data = np.zeros((num_samples, self.max_seq_len))
        decoder_input_data = np.zeros((num_samples, self.max_seq_len))
        decoder_target_data = np.zeros((num_samples, self.max_seq_len, self.vocab_size + 1))  # +1 as mask_zero

        conversation_texts = []
        response_texts = []

        for i in range(num_samples):

            seq_source, seq_target = self.pairs[i_sample + i]
            if not bool(seq_target) or not bool(seq_source):
                continue

            if seq_target[-1] != EOS:
                seq_target.append(EOS)

            conversation_texts.append(' '.join([self.vocab.words_to_id(j) for j in seq_source]))
            response_texts.append(' '.join([self.vocab.words_to_id(j) for j in seq_target]))

            for t, token_index in enumerate(seq_source):
                encoder_input_data[i, t] = token_index

            decoder_input_data[i, 0] = SOS
            for t, token_index in enumerate(seq_target):
                decoder_input_data[i, t + 1] = token_index
                decoder_target_data[i, t, token_index] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data, conversation_texts, response_texts


class KDataSet:
    def __init__(self):
        pass


if __name__ == '__main__':
    # train_dataset = TextDataSet(
    #     train_file, vocab, category_vocab, hps.num_timesteps)
    # val_dataset = TextDataSet(
    #     val_file, vocab, category_vocab, hps.num_timesteps)
    # test_dataset = TextDataSet(
    #     test_file, vocab, category_vocab, hps.num_timesteps)

    pass
