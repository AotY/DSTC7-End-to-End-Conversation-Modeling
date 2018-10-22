# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import pickle

import torch
import numpy as np

from misc import es_helper
from embedding.embedding_score import get_topk_facts

from misc.url_tags_weight import tags_weight_dict


class DataSet:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 model_type,
                 pair_path,
                 max_len,
                 fact_max_len,
                 fact_topk,
                 vocab,
                 save_path,
                 turn_num,
                 eval_split,  # how many hold out as eval data
                 device,
                 logger):

        self.model_type = model_type
        self.max_len = max_len
        self.fact_max_len = fact_max_len
        self.fact_topk = fact_topk
        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size()

        self.device = device
        self.logger = logger

        # es
        self.es = es_helper.get_connection()

        self._data_dict = {}
        self._indicator_dict = {}

        self.read_txt(save_path,
                      pair_path,
                      turn_num,
                      eval_split)

    def read_txt(self, save_path, pair_path, turn_num, eval_split):
        _data_dict_path = os.path.join(save_path, '{}_data_dict.pkl')
        if not os.path.exists(_data_dict_path):
            # load source-target pairs, tokenized
            datas = []
            with open(pair_path, 'r', encoding='utf-8') as f:
                for line in f:
                    conversation, response, hash_value = line.rstrip().split('SPLITTOKEN')

                    if not bool(conversation) or not bool(response):
                        continue

                    # conversation split by EOS, START
                    if conversation.startswith('start eos'):
                        # START: special symbol indicating the start of the
                        # conversation
                        conversation = conversation[10:]
                        history_dialogues = self.parser_dialogues(
                            conversation, turn_num, 'conversation')
                    elif conversation.startswith('eos'):
                        # EOS: special symbol indicating a turn transition
                        conversation = conversation[4:]
                        history_dialogues = self.parser_dialogues(
                            conversation, turn_num, 'turn')
                    else:
                        history_dialogues = self.parser_dialogues(
                            conversation, turn_num, 'other')

                    if history_dialogues is None:
                        continue

                    #  conversation_context = ' '.join(history_dialogues)
                    conversation = history_dialogues[0]
                    response = history_dialogues[1]

                    conversation_ids = self.vocab.words_to_id(
                        conversation.split(' '))
                    if len(conversation_ids) <= 3:
                            continue

                    conversation_ids = conversation_ids[-min(
                        self.max_len - 1, len(conversation_ids)):]

                    response_ids = self.vocab.words_to_id(response.split(' '))
                    if len(response_ids) <= 3:
                            continue

                    response_ids = response_ids[-min(
                        self.max_len - 1, len(response_ids)):]

                    datas.append((conversation_ids, response_ids, hash_value))

            np.random.shuffle(datas)
            # train-eval split
            self.n_train = int(len(datas) * (1. - eval_split))
            self.n_eval = len(datas) - self.n_train

            self._data_dict = {
                'train': datas[0: self.n_train],
                'eval': datas[self.n_train:]
            }

            pickle.dump(self._data_dict, open(_data_dict_path, 'wb'))
        else:
            self._data_dict = pickle.load(_data_dict_path, 'rb'))
            self.n_train=len(self._data_dict['train'])
            self.n_eval=len(self._data_dict['eval'])

        self._indicator_dict={
            'train': 0,
            'eval': 0
        }

    def parser_dialogues(self, conversation, turn_num=1, symbol='conversation'):
        """
        parser conversation context by dialogue turn (default 1)
        """
        history_dialogues = conversation.split('eos')
        history_dialogues = [history_dialogue for history_dialogue in history_dialogues if len(
            history_dialogue.split()) > 3]

        if len(history_dialogues) < turn_num:
            return None

        history_dialogues = history_dialogues[-min(
            turn_num, len(history_dialogues)):]

        return history_dialogues

    def reset_data(self, task):
        np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def load_data(self, task, batch_size):
        task_len = len(self._data_dict[task])
        if batch_size > task_len:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > task_len:
            self.reset_data(task)
            cur_indicator = batch_size

        encoder_inputs = torch.zeros((self.max_len, batch_size),
                                    dtype = torch.long,
                                     device = self.device)
        encoder_inputs_length=[]

        decoder_inputs=torch.zeros((self.max_len, batch_size),
                                     dtype = torch.long,
                                     device = self.device)

        decoder_targets=torch.zeros((self.max_len, batch_size),
                                      dtype = torch.long,
                                      device = self.device)

        conversation_texts=[]
        response_texts=[]

        # facts
        facts_inputs = []
        facts_texts=[]

        batch_data=self._data_dict[task][self._indicator_dict[task]: cur_indicator]
        for i, (conversation_ids, response_ids, hash_value) in enumerate(batch_data):
            # append length
            encoder_inputs_length.append(len(conversation_ids))

            # ids to word
            conversation_text=' '.join(self.vocab.ids_to_word(conversation_ids))
            response_text=' '.join(self.vocab.ids_to_word(response_ids))

            conversation_texts.append(conversation_text)
            response_texts.append(response_text)

            # encoder_inputs
            for c, token_id in enumerate(conversation_ids):
                encoder_inputs[c, i]=token_id
            encoder_inputs[len(conversation_ids), i]=self.vocab.eosid

            # decoder_inputs
            decoder_inputs[0, i]=self.vocab.sosid
            for r, token_id in enumerate(response_ids):
                decoder_inputs[r + 1, i]=token_id
                decoder_targets[r, i]=token_id
            decoder_targets[len(response_ids), i]=self.vocab.eosid

            if self.model_type == 'kg':
                topk_facts_embedded, topk_facts=self.assembel_facts(hash_value)
                facts_inputs.append(topk_facts_embedded)
                facts_texts.append(topk_facts_text)

        # To long tensor
        encoder_inputs_length = torch.tensor(encoder_inputs_length, dtype = torch.long, device =self.device)
        # update _indicator_dict[task]
        self._indicator_dict[task]=cur_indicator

        if self.model_type == 'kg':
            facts_inputs = torch.cat(facts_inputs, dim=0)

        return encoder_inputs, encoder_inputs_length, \
            decoder_inputs, decoder_targets, \
            conversation_texts, response_texts \
            facts_inputs, facts_texts


    def assembel_facts(self, hash_value):

        # load top_k facts
        topk_facts_embedded, topk_facts = self.topk_facts_embedded_dict.get(hash_value,
                                                                             (None, None))
        if topk_facts_embedded is None:
            return (torch.zeros((self.fact_topk, self.embedding_size), device=self.device), [])

        return topk_facts_embedded, topk_facts

    def computing_similarity_facts_offline(self,
                                           embedding_size,
                                           embedding,
                                           topk,
                                           filename):

        self.embedding_size = embedding_size

        if os.path.exists(filename):
            self.logger.info('Loading topk_facts_embedded_dict')
            self.topk_facts_embedded_dict = pickle.load(open(filename, 'rb'))
        else:
            self.logger.info('Computing topk_facts_embedded_dict..........')

            topk_facts_embedded_dict = {}
            with torch.no_grad():
                """ computing similarity between conversation and facts, then saving to dict"""
                for task, datas in self._data_dict.items():
                    self.logger.info('computing similarity: %s ' % task)

                    for conversation_ids, _, hash_value in datas:
                        if not bool(conversation_ids) or not bool(hash_value):
                            continue

                        # search facts ?
                        hit_count, facts = es_helper.search_facts(self.es, hash_value)

                        # parser html tags, <h1-6> <title> <p> etc.
                        # facts to id
                        facts, facts_weight = get_facts_weight(facts)

                        facts_ids = [self.vocab.words_to_id(fact.split(' ')) for fact in facts]
                        if len(facts_ids) == 0:
                            continue

                        facts_weight = torch.tensor(facts_weight, dtype=torch.float, device=self.device)

                        facts_ids = [fact_ids[-min(self.fact_max_len, len(facts_ids)): ] for fact_ids in facts_ids]

                        # score top_k_facts_embedded -> [top_k, embedding_size]
                        topk_facts_embedded, topk_indexes = get_topk_facts(embedding_size, 
                                                                            embedding,
                                                                            conversation_ids,
                                                                            facts_ids,
                                                                            topk,
                                                                            facts_weight,
                                                                            device)
                        topk_indexes_list = topk_indexes.tolist()
                        topk_facts_text = [facts[topi] for topi in topk_indexes_list]

                        topk_facts_embedded_dict[hash_value] = (topk_facts_embedded, topk_facts_text)

            # save topk_facts_embedded_dict
            pickle.dump(topk_facts_embedded_dict, open(filename, 'wb'))
            self.topk_facts_embedded_dict = topk_facts_embedded_dict

    def get_facts_weight(facts):
        """ facts: [[w_n] * size]"""
        facts_wight = []
        new_facts = []
        for fact in facts:
            new_fact = ''
            fact_weight = tags_weight_dict['default']
            fact = ' '.join(fact)
            for tag, weight in tags_weight_dict.items():
                if not bool(fact):
                    break
                soup =  BeautifulSoup(fact, 'lxml')
                tag = soup.find(tag)
                if tag is not None:
                    new_fact += tag.text
                    fact_weight = max(fact_weight, weight)
                    fact = fact.replace(str(tag), '')

            new_facts.append(new_fact)
            facts_wight.append(fact_weight)

        return new_facts,  facts_weight


    def save_generated_texts(self,
                             conversation_texts,
                             response_texts,
                             batch_generated_texts,
                             filename,
                             decode_type='greed',
                             topk_facts=None):

        with open(filename, 'w', encoding='utf-8') as f:
            for conversation, response, generated_text in zip(conversation_texts, response_texts, batch_generated_texts):
                # conversation, true response, generated_text
                f.write('Conversation: %s\n' % conversation)
                f.write('Response: %s\n' % response)
                if decode_type == 'greedy':
                    f.write('Generated: %s\n' % generated_text)
                elif decode_type == 'beam_search':
                    for i, topk_text in enumerate(batch_generated_texts):
                        f.write('Generated %d: %s\n' % (i, topk_text))

                if topk_facts is not None:
                    for fi, fact_text in enumerate(topk_facts):
                        f.write('Facts %d: %s\n' % (fi, fact_text))

                f.write('---------------------------------\n')


    def assemble_texts(self, batch_utterances, batch_size, decode_type='greedy'):
        """
        decode_type == greedy:
            batch_utterances: [batch_size, max_length]
            return: [batch_size]
        decode_type == 'beam_search':
            batch_utterances: [batch_size, topk, len]
            return: [batch_size, topk]
        """

        batch_generated_texts = []
        if decode_type == 'greedy':
            for bi in range(batch_size):
                text = self.ids_to_text(batch_utterances[bi].tolist())
                batch_generated_texts.append(text)
        elif decode_type == 'beam_search':
            for bi in range(batch_size):
                topk_text_ids = batch_utterances[bi]
                topk_texts = []
                for ids in topk_text_ids:
                    text = self.ids_to_text(ids)
                    topk_texts.append(text)
                batch_generated_texts.append(topk_texts)

        return batch_generated_texts


