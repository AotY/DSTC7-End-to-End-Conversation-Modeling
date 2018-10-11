# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import pickle

import torch
import numpy as np

from misc import es_helper
from embedding.embedding_score import get_top_k_fact_average


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
                 eval_split=0.2,  # how many hold out as eval data
                 device=None,
                 logger=None):

        self.dialog_encoder_vocab_size = dialog_encoder_vocab.get_vocab_size()
        self.dialog_encoder_max_length = dialog_encoder_max_length
        self.dialog_encoder_vocab = dialog_encoder_vocab

        self.dialog_decoder_vocab_size = dialog_decoder_vocab.get_vocab_size()
        self.dialog_decoder_max_length = dialog_decoder_max_length
        self.dialog_decoder_vocab = dialog_decoder_vocab

        self.device = device
        self.logger = logger
        self._data_dict = {}
        self._indicator_dict = {}
        self.read_txt(path_conversations_responses_pair, eval_split)

    def read_txt(self, path_conversations_responses_pair, eval_split):
        self.logger.info('loading data from txt files: {}'.format(
            path_conversations_responses_pair))
        # load source-target pairs, tokenized
        datas = []
        with open(path_conversations_responses_pair, 'r', encoding='utf-8') as f:
            for line in f:
                conversation, response, hash_value = line.rstrip().split('\t')

                conversation_ids = self.dialog_encoder_vocab.words_to_id(
                    conversation.split())
                conversation_ids = conversation_ids[0: min(
                    self.dialog_encoder_max_length - 2, len(conversation_ids))]

                response_ids = self.dialog_decoder_vocab.words_to_id(
                    response.split())
                response_ids = response_ids[0: min(
                    self.dialog_decoder_max_length - 2, len(response_ids))]

                datas.append((conversation_ids, response_ids))

        np.random.shuffle(datas)

        # train-eval split
        self.n_train = int(len(datas) * (1. - eval_split))
        self.n_eval = len(datas) - self.n_train

        self._data_dict = {
            'train': datas[0: self.n_train],
            'eval': datas[self.n_train:]
        }
        self._indicator_dict = {
            'train': 0,
            'eval': 0
        }

    def reset_data(self, task):
        np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def all_loaded(self, task):
        return self._data_dict[task]

    def load_data(self, task, batch_size):
        # if batch > len(self._data_dict[task] raise error
        task_len = len(self._data_dict[task])
        if batch_size > task_len:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > task_len:
            self.reset_data(task)
            cur_indicator = batch_size

        self.logger.info('building %s data from %d to %d' %
                         (task, self._indicator_dict[task], cur_indicator))

        encoder_inputs = torch.zeros((self.dialog_encoder_max_length, batch_size),
                                     dtype=torch.long,
                                     device=self.device)
        encoder_inputs_length = []

        decoder_inputs = torch.zeros((self.dialog_decoder_max_length, batch_size),
                                     dtype=torch.long,
                                     device=self.device)
        decoder_targets = torch.zeros((self.dialog_decoder_max_length, batch_size),
                                      dtype=torch.long,
                                      device=self.device)

        conversation_texts = []
        response_texts = []

        batch_data = self._data_dict[task][self._indicator_dict[task]: cur_indicator]
        for i, (conversation_ids, response_ids) in enumerate(batch_data):
            if not bool(response_ids) or not bool(conversation_ids):
                continue

            # append length
            encoder_inputs_length.append(len(conversation_ids))

            if response_ids[-1] != self.dialog_decoder_vocab.eosid:
                response_ids.append(self.dialog_decoder_vocab.eosid)
            # ids to word
            conversation_texts.append(
                ' '.join(self.dialog_encoder_vocab.ids_to_word(conversation_ids)))
            response_texts.append(
                ' '.join(self.dialog_decoder_vocab.ids_to_word(response_ids)))

            # encoder_inputs
            for t, token_id in enumerate(conversation_ids):
                encoder_inputs[t, i] = token_id

            # decoder_inputs
            decoder_inputs[0, i] = self.dialog_decoder_vocab.sosid
            for t, token_id in enumerate(response_ids):
                decoder_inputs[t + 1, i] = token_id
                decoder_targets[t, i] = token_id

        # To long tensor
        encoder_inputs_length = torch.tensor(
            encoder_inputs_length, dtype=torch.long)

        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator

        return encoder_inputs, encoder_inputs_length, \
            decoder_inputs, decoder_targets, \
            conversation_texts, response_texts

    def generating_texts(self, decoder_outputs_argmax, batch_size):
        """
        decoder_outputs_argmax: [max_length, batch_size]
        return: [text * batch_size]
        """
        texts = []
        decoder_outputs_argmax.transpose_(0, 1)
        for bi in range(batch_size):
            text_ids = decoder_outputs_argmax[bi]
            words = self.dialog_encoder_vocab.ids_to_word(text_ids)
            text = ' '.join(words)
            texts.append(text)

        return texts

    def save_generated_texts(self, conversation_texts, response_texts, generated_texts, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for conversation, response, generated_text in zip(conversation_texts, response_texts, generated_texts):
                # conversation, true response, generated_text
                f.write('Conversation: %s\n' % conversation)
                f.write('Response: %s\n' % response)
                f.write('Generated: %s\n' % generated_text)
                f.write('---------------------------------\n')


class KnowledgeGroundedDataSet:
    """
    KnowledgeGroundedDataSet
        conversations_responses_pair (Conversation, response, hash_value)
        dialog_encoder_vocab
        dialog_encoder_max_length
        fact_vocab
        fact_max_length
        dialog_decoder_vocab
        dialog_decoder_max_length
        device
        logger
    """

    def __init__(self,
                 path_conversations_responses_pair=None,
                 dialog_encoder_max_length=50,
                 dialog_encoder_vocab=None,
                 fact_vocab=None,
                 fact_max_length=50,
                 dialog_decoder_max_length=50,
                 dialog_decoder_vocab=None,
                 eval_split=0.2,  # how many hold out as eval data
                 device=None,
                 logger=None):

        self.dialog_encoder_vocab_size = dialog_encoder_vocab.get_vocab_size()
        self.dialog_encoder_max_length = dialog_encoder_max_length
        self.dialog_encoder_vocab = dialog_encoder_vocab

        self.fact_max_length = fact_max_length
        self.fact_vocab = fact_vocab

        self.dialog_decoder_vocab_size = dialog_decoder_vocab.get_vocab_size()
        self.dialog_decoder_max_length = dialog_decoder_max_length
        self.dialog_decoder_vocab = dialog_decoder_vocab

        self.device = device
        self.logger = logger
        self._data_dict = {}
        self._indicator_dict = {}

        # es
        self.es = es_helper.get_connection()

        # facts dict
        self.top_k_facts_embedded_mean_dict = None

        # read text, prepare data
        self.read_txt(path_conversations_responses_pair, eval_split)

    def read_txt(self, path_conversations_responses_pair, eval_split):
        self.logger.info('loading data from txt files: {}'.format(
            path_conversations_responses_pair))
        # load source-target pairs, tokenized
        datas = []
        with open(path_conversations_responses_pair, 'r', encoding='utf-8') as f:
            for line in f:
                conversation, response, hash_value = line.rstrip().split('\t')

                conversation_ids = self.dialog_encoder_vocab.words_to_id(
                    conversation.split())

                conversation_ids = conversation_ids[0: min(
                    self.dialog_encoder_max_length - 2, len(conversation_ids))]

                response_ids = self.dialog_decoder_vocab.words_to_id(
                    response.split())
                response_ids = response_ids[0: min(
                    self.dialog_decoder_max_length - 2, len(response_ids))]

                datas.append((conversation_ids, response_ids, hash_value))

        np.random.shuffle(datas)

        # train-eval split
        self.n_train = int(len(datas) * (1. - eval_split))
        self.n_eval = len(datas) - self.n_train

        self._data_dict = {
            'train': datas[0: self.n_train],
            'eval': datas[self.n_train:]
        }
        self._indicator_dict = {
            'train': 0,
            'eval': 0
        }

    def reset_data(self, task):
        np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def all_loaded(self, task):
        return self._data_dict[task]

    def load_data(self, task, batch_size, top_k, fact_embedding_size):
        # if batch > len(self._data_dict[task] raise error
        task_len = len(self._data_dict[task])
        if batch_size > task_len:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > task_len:
            self.reset_data(task)
            cur_indicator = batch_size

        self.logger.info('building %s data from %d to %d' %
                         (task, self._indicator_dict[task], cur_indicator))

        encoder_inputs = torch.zeros((self.dialog_encoder_max_length, batch_size),
                                     dtype=torch.long,
                                     device=self.device)
        encoder_inputs_length = []

        decoder_inputs = torch.zeros((self.dialog_decoder_max_length, batch_size),
                                     dtype=torch.long,
                                     device=self.device)
        decoder_targets = torch.zeros((self.dialog_decoder_max_length, batch_size),
                                      dtype=torch.long,
                                      device=self.device)

        conversation_texts = []
        response_texts = []

        # facts
        facts_inputs = torch.zeros((batch_size, top_k, fact_embedding_size))
        facts_texts = []

        batch_data = self._data_dict[task][self._indicator_dict[task]                                           : cur_indicator]
        for i, (conversation_ids, response_ids, hash_value) in enumerate(batch_data):
            if not bool(response_ids) or not bool(conversation_ids) or not bool(hash_value):
                continue
            # append length
            encoder_inputs_length.append(len(conversation_ids))

            if response_ids[-1] != self.dialog_decoder_vocab.eosid:
                response_ids.append(self.dialog_decoder_vocab.eosid)

            # ids to word
            conversation_texts.append(
                ' '.join(self.dialog_encoder_vocab.ids_to_word(conversation_ids)))
            response_texts.append(
                ' '.join(self.dialog_decoder_vocab.ids_to_word(response_ids)))

            # encoder_inputs
            for t, token_id in enumerate(conversation_ids):
                encoder_inputs[t, i] = token_id

            # decoder_inputs
            decoder_inputs[0, i] = self.dialog_decoder_vocab.sosid
            for t, token_id in enumerate(response_ids):
                decoder_inputs[t + 1, i] = token_id
                decoder_targets[t, i] = token_id

            # load top_k facts
            top_k_facts_embedded_mean, top_k_fact_texts, top_k_indices_list = self.top_k_facts_embedded_mean_dict(hash_value)

            facts_inputs[i] = top_k_facts_embedded_mean
            facts_texts.append(top_k_fact_texts)

        # To long tensor
        encoder_inputs_length = torch.tensor(encoder_inputs_length,
                                             dtype=torch.long,
                                             device=self.device)
        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator

        return encoder_inputs, encoder_inputs_length, \
            facts_inputs, decoder_inputs, decoder_targets, \
            conversation_texts, response_texts, facts_texts

    def computing_similarity_offline(self, encoder_embedding, fact_embedding,
                                     encoder_embedding_size, fact_embedding_size,
                                     top_k, device, filename, logger):
        if os.path.exists(filename):
            logger.info('Loading top_k_facts_embedded_mean_dict')
            self.top_k_facts_embedded_mean_dict = pickle.load(
                open(filename, 'rb'))
        else:
            logger.info('Computing top_k_facts_embedded_mean_dict')

            top_k_facts_embedded_mean_dict = {}
            with torch.no_grad():
                """ computing similarity between conversation and facts, then saving to dict"""
                for task, datas in self._data_dict.items():
                    logger.info('computing similarity: %s ' % task)

                    for conversation_ids, _, hash_value in datas:
                        if not bool(conversation_ids) or not bool(hash_value):
                            continue
                        # search facts ?
                        hit_count, facts, domains, conversation_ids = es_helper.search_facts_by_conversation_hash_value(
                            self.es, hash_value)

                        # facts to id
                        facts_ids = [self.fact_vocab.words_to_id(
                            fact) for fact in facts]
                        facts_ids = facts_ids[0: min(
                            self.fact_max_length - 2, len(facts_ids))]

                        fact_texts = [' '.join(fact) for fact in facts]

                        # score top_k_facts_embedded -> [top_k, embedding_size]
                        top_k_facts_embedded_mean, top_k_indices = get_top_k_fact_average(encoder_embedding, fact_embedding,
                                                                                          encoder_embedding_size, fact_embedding_size,
                                                                                          conversation_ids, facts_ids, top_k, device)
                        top_k_fact_texts = []
                        top_k_indices_list = top_k_indices.tolist()
                        for topi in top_k_indices_list:
                            top_k_fact_texts.append(fact_texts[topi])

                        top_k_facts_embedded_mean_dict[hash_value] = (
                            top_k_facts_embedded_mean, top_k_fact_texts, top_k_indices_list)

            # save top_k_facts_embedded_mean_dict
            pickle.dum(top_k_facts_embedded_mean_dict, open(filename, 'wb'))
            self.top_k_facts_embedded_mean_dict = top_k_facts_embedded_mean_dict

    def generating_texts(self, decoder_outputs_argmax, batch_size):
        """
        decoder_outputs_argmax: [max_length, batch_size]
        return: [text * batch_size]
        """
        texts = []
        decoder_outputs_argmax.transpose_(0, 1)
        for bi in range(batch_size):
            text_ids = decoder_outputs_argmax[bi]
            words = self.dialog_encoder_vocab.ids_to_word(text_ids)
            text = ' '.join(words)
            texts.append(text)

        return texts

    def save_generated_texts(self, conversation_texts, response_texts, generated_texts, top_k_fact_texts, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for conversation, response, generated_text in zip(conversation_texts, response_texts, generated_texts):
                # conversation, true response, generated_text
                f.write('Conversation: %s\n' % conversation)
                f.write('Response: %s\n' % response)
                f.write('Generated: %s\n' % generated_text)
                for fi, fact_text in enumerate(top_k_fact_texts):
                    f.write('Facts %d: %s\n' % (fi, fact_text))
                f.write('---------------------------------\n')


if __name__ == '__main__':
    pass
