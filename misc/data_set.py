# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import torch
import numpy as np


class Seq2seqDataSet:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 path_conversations_responses_pair,
                 dialog_encoder_max_length=50,
                 dialog_encoder_vocab=None,
                 dialog_decoder_max_length=50,
                 dialog_decoder_vocab=None,
                 test_split=0.2,  # how many hold out as vali data
                 device=None,
                 logger=None
                 ):

        self.dialog_encoder_vocab_size = dialog_encoder_vocab.get_vocab_size()
        self.dialog_encoder_max_length = dialog_encoder_max_length
        self.dialog_encoder_vocab = dialog_encoder_vocab

        self.dialog_decoder_vocab_size = dialog_decoder_vocab.get_vocab_size()
        self.dialog_decoder_max_length = dialog_decoder_max_length
        self.dialog_decoder_vocab = dialog_decoder_vocab

        self.device = device
        self.logger = logger
        self.read_txt(path_conversations_responses_pair, test_split)

    def read_txt(self, path_conversations_responses_pair, test_split):
        self.logger.info('loading data from txt files: {}'.format(
            path_conversations_responses_pair))
        # load source-target pairs, tokenized

        seqs = dict()
        '''
        for k, path in [('conversation', path_conversations), ('response', path_responses)]:
            seqs[k] = []

            with open(path, 'r', encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                seq = []
                for c in line.strip('\n').strip().split(' '):
                    if k == 'conversation':
                        i = self.dialog_encoder_vocab.word_to_id(c)
                    elif k == 'response':
                        i = self.dialog_decoder_vocab.word_to_id(c)
                    seq.append(i)

                if k == 'conversation':
                    seqs[k].append(seq[-min(self.dialog_encoder_max_length - 2, len(seq)):])
                elif k == 'response':
                    seqs[k].append(seq[-min(self.dialog_decoder_max_length - 2, len(seq)):])
        '''
        seqs['conversation'] = []
        seqs['response'] = []
        with open(path_conversations_responses_pair, 'r', encoding='utf-8') as f:
            for line in f:
                conversation, response, hash_value = line.rstrip().split('\t')

                ids = self.dialog_encoder_vocab.words_to_id(conversation.split())
                seqs['conversation'].append(ids[-min(self.dialog_encoder_max_length - 2, len(ids)):])

                ids = self.dialog_decoder_vocab.words_to_id(response.split())
                seqs['response'].append(ids[-min(self.dialog_encoder_max_length - 2, len(ids)):])

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

        i_sample_next = min(i_sample + max_num_sample_loaded,
                            self.i_sample_range[task][1])

        num_samples = i_sample_next - i_sample

        self.i_sample[task] = i_sample_next

        self.logger.info('building %s data from %i to %i' %
                         (task, i_sample, i_sample_next))

        encoder_input_data = torch.zeros(
            (self.dialog_encoder_max_length, num_samples))
        encoder_input_lengths = []

        decoder_input_data = torch.zeros(
            (self.dialog_decoder_max_length, num_samples))
        decoder_target_data = torch.zeros(
            (self.dialog_decoder_max_length, num_samples))
        decoder_input_lengths = []

        conversation_texts = []
        response_texts = []

        for i in range(num_samples):

            seq_conversation, seq_response = self.pairs[i_sample + i]

            if not bool(seq_response) or not bool(seq_conversation):
                continue

            if seq_response[-1] != self.dialog_encoder_vocab.eosid:
                seq_response.append(self.dialog_encoder_vocab.eosid)

            conversation_texts.append(
                ' '.join([str(self.dialog_encoder_vocab.id_to_word(j)) for j in seq_conversation]))
            response_texts.append(
                ' '.join([str(self.dialog_decoder_vocab.id_to_word(j)) for j in seq_response]))

            for t, token_id in enumerate(seq_conversation):
                encoder_input_data[t, i] = token_id

            encoder_input_lengths.append(len(seq_conversation))

            decoder_input_data[0, i] = self.dialog_decoder_vocab.sosid
            for t, token_id in enumerate(seq_response):
                decoder_input_data[t + 1, i] = token_id
                decoder_target_data[t, i] = token_id

            decoder_input_lengths.append(len(seq_response))

        # To long tensor
        encoder_input_data = torch.tensor(
            encoder_input_data, dtype=torch.long, device=self.device)
        decoder_input_data = torch.tensor(
            decoder_input_data, dtype=torch.long, device=self.device)
        decoder_target_data = torch.tensor(
            decoder_target_data, dtype=torch.long, device=self.device)

        encoder_input_lengths = torch.tensor(
            encoder_input_lengths, dtype=torch.long)
        decoder_input_lengths = torch.tensor(
            decoder_input_lengths, dtype=torch.long)

        return num_samples, \
            encoder_input_data, \
            decoder_input_data, \
            decoder_target_data, \
            encoder_input_lengths, decoder_input_lengths, \
            conversation_texts, response_texts


class KDataSet:
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
