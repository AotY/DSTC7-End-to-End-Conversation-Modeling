# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import pickle

import torch
import numpy as np

from summa import keywords # for keyword extraction

from misc import es_helper
#  from embedding.embedding_score import get_topk_facts

from misc.url_tags_weight import tag_weight_dict
from misc.url_tags_weight import default_weight


class Dataset:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 model_type,
                 pair_path,
                 max_len,
                 h_max_len,
                 c_max_len,
                 r_max_len,
                 min_len,
                 fact_max_len,
                 f_topk,
                 vocab,
                 save_path,
                 turn_num,
                 turn_type,
                 eval_split,  # how many hold out as eval data
                 device,
                 logger):

        self.model_type = model_type
        self.max_len = max_len
        self.h_max_len = h_max_len
        self.c_max_len = c_max_len
        self.r_max_len = r_max_len
        self.min_len = min_len
        self.fact_max_len = fact_max_len
        self.f_topk = f_topk
        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size()
        self.turn_num = turn_num
        self.turn_type = turn_type

        self.device = device
        self.logger = logger

        # es
        self.es = es_helper.get_connection()

        self._data_dict = {}
        self._indicator_dict = {}

        self.read_txt(save_path,
                      pair_path,
                      eval_split)

    def read_txt(self, save_path, pair_path, eval_split):
        _data_dict_path = os.path.join(save_path, '_data_dict_%s.pkl' % self.turn_type)
        if not os.path.exists(_data_dict_path):
            # load source-target pairs, tokenized
            datas = []
            with open(pair_path, 'r', encoding='utf-8') as f:
                for line in f:
                    conversation, response, hash_value = line.rstrip().split('SPLITTOKEN')

                    if not bool(conversation) or not bool(response):
                        continue

                    response_ids = self.vocab.words_to_id(response.split(' '))
                    if len(response_ids) < self.min_len:
                            continue
                    response_ids = response_ids[-min(self.r_max_len - 1, len(response_ids)):]

                    # conversation split by EOS, START
                    if conversation.startswith('start eos'):
                        # START: special symbol indicating the start of the
                        # conversation
                        conversation = conversation[10:]
                        history_conversations = self.parser_conversations(conversation, 'conversation')
                    elif conversation.startswith('eos'):
                        # EOS: special symbol indicating a turn transition
                        conversation = conversation[4:]
                        history_conversations = self.parser_conversations(conversation, 'turn')
                    else:
                        history_conversations = self.parser_conversations(
                            conversation, 'other')

                    if history_conversations is None:
                        continue

                    raw_conversation = conversation

                    # len(history_conversations) <= self.turn_num
                    if self.turn_type == 'concat':
                        conversation = ' '.join(history_conversations)
                        history_conversations = []
                    elif self.turn_type == 'dcgm':
                        conversation = history_conversations[-1]
                        history_conversations = [' '.join(history_conversations[:-1])]
                    elif self.turn_type == 'hred':
                        conversation = history_conversations[-1]
                        history_conversations = history_conversations[:-1]
                    elif self.turn_type == 'none':
                        conversation = history_conversations[-1]
                        history_conversations = []

                    conversation_ids = self.vocab.words_to_id(conversation.split(' '))
                    conversation_ids = conversation_ids[-min(self.c_max_len, len(conversation_ids)):]

                    history_conversations_ids = []
                    for history in history_conversations:
                        history_ids = self.vocab.words_to_id(history.split(' '))
                        history_ids = history_ids[-min(self.h_max_len, len(history_ids))]
                        history_conversations_ids.append(history_ids)

                    datas.append((raw_conversation, history_conversations_ids, conversation_ids, response_ids, hash_value))

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
            self._data_dict = pickle.load(open(_data_dict_path, 'rb'))
            self.n_train = len(self._data_dict['train'])
            self.n_eval = len(self._data_dict['eval'])

        self._indicator_dict = {
            'train': 0,
            'eval': 0
        }

    def parser_conversations(self, conversation, symbol='conversation'):
        """
        parser conversation context by dialogue turn (default 1)
        """
        history_conversations = conversation.split('eos')
        history_conversations = [history_dialogue for history_dialogue in history_conversations if len(
            history_dialogue.split()) >= self.min_len]

        if len(history_conversations) < self.turn_num:
            return None

        history_conversations = history_conversations[-min(
            self.turn_num, len(history_conversations)):]

        return history_conversations

    def reset_data(self, task):
        np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def load_data(self, task, batch_size):
        task_len=len(self._data_dict[task])
        if batch_size > task_len:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator=self._indicator_dict[task] + batch_size
        if cur_indicator > task_len:
            self.reset_data(task)
            cur_indicator=batch_size

        h_encoder_inputs = []
        #  h_encoder_inputs_length = []

        c_encoder_inputs=torch.zeros((self.c_max_len, batch_size),
                                     dtype=torch.long,
                                     device=self.device)
        c_encoder_inputs_lengths=[]

        decoder_inputs=torch.zeros((self.r_max_len, batch_size),
                                     dtype=torch.long,
                                     device=self.device)

        decoder_targets=torch.zeros((self.r_max_len, batch_size),
                                      dtype=torch.long,
                                      device=self.device)

        conversation_texts=[]
        response_texts=[]

        # facts
        f_encoder_inputs=[]
        facts_texts=[]

        batch_data=self._data_dict[task][self._indicator_dict[task]: cur_indicator]
        for i, (conversation, history_conversations_ids, conversation_ids, response_ids, hash_value) in enumerate(batch_data):

            # append length
            c_encoder_inputs_lengths.append(len(conversation_ids))

            # ids to word
            conversation_text = ' '.join(self.vocab.ids_to_word(conversation_ids))
            response_text = ' '.join(self.vocab.ids_to_word(response_ids))
            conversation_texts.append(conversation_text)
            response_texts.append(response_text)

			# history inputs
            if len(history_conversations_ids) > 0:
                history_input=[]
                #  history_input_length = []
                for ids in history_conversations_ids:
                    #  slot_ids = torch.zeros(self.h_max_len, dtype=torch.long, device=self.device)
                    #  slot_ids[:len(ids)] = ids[:len(ids)]
                    ids = torch.tensor(ids, dtype=torch.long, device=self.device)
                    history_input.append(ids)
                    #  history_input_length.append(len(ids))
                history_input = torch.stack(history_input, dim=0) #[num, ids]
                h_encoder_inputs.append(history_input)

            # c_encoder_inputs
            for c, token_id in enumerate(conversation_ids):
                c_encoder_inputs[c, i] = token_id
            #  c_encoder_inputs[len(conversation_ids), i]=self.vocab.eosid

            # decoder_inputs
            decoder_inputs[0, i]=self.vocab.sosid
            for r, token_id in enumerate(response_ids):
                decoder_inputs[r + 1, i]=token_id
                decoder_targets[r, i]=token_id
            decoder_targets[len(response_ids), i]=self.vocab.eosid

            if self.model_type == 'kg':
                topk_facts_embedded, topk_facts_text=self.assembel_facts(hash_value)
                if topk_facts_embedded.size(0) < self.f_topk:
                    tmp_tensor = torch.zeros((self.f_topk, topk_facts_embedded.size(1)))
                    tmp_tensor[:topk_facts_embedded.size(0)] = topk_facts_embedded
                    topk_facts_embedded = tmp_tensor
                elif topk_facts_embedded.size(0) > self.f_topk:
                    topk_facts_embedded = topk_facts_embedded[:self.f_topk]

                topk_facts_embedded = topk_facts_embedded.to(self.device)
                f_encoder_inputs.append(topk_facts_embedded)
                facts_texts.append(topk_facts_text)

        # To long tensor
        c_encoder_inputs_lengths = torch.tensor(c_encoder_inputs_lengths, dtype=torch.long, device=self.device)
        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator

        if self.model_type == 'kg':
            f_encoder_inputs=torch.stack(f_encoder_inputs, dim=0) #[batch_size, topk, embedding_size]

        return c_encoder_inputs, c_encoder_inputs_lengths, \
            decoder_inputs, decoder_targets, \
            conversation_texts, response_texts, \
            f_encoder_inputs, facts_texts, h_encoder_inputs


    def assembel_facts(self, hash_value):
        # load top_k facts
        topk_facts_embedded, topk_facts=self.topk_facts_embedded_dict.get(hash_value, (None, None))
        if topk_facts_embedded is None:
            return (torch.zeros((self.f_topk, self.embedding_size), device=self.device), [])

        return topk_facts_embedded, topk_facts

    def computing_similarity_facts_offline(self,
                                           fasttext,
                                           embedding_size,
                                           f_topk,
                                           filename):

        self.embedding_size=embedding_size

        if os.path.exists(filename):
            self.logger.info('Loading topk_facts_embedded_dict')
            self.topk_facts_embedded_dict=pickle.load(open(filename, 'rb'))
        else:
            self.logger.info('Computing topk_facts_embedded_dict..........')

            topk_facts_embedded_dict={}
            with torch.no_grad():
                """ computing similarity between conversation and facts, then saving to dict"""
                for task, datas in self._data_dict.items():
                    self.logger.info('computing similarity: %s ' % task)
                    for conversation, _, conversation_ids, _, hash_value in datas:
                        if not bool(conversation_ids) or not bool(hash_value):
                            continue

                        # search facts ?
                        hit_count, facts = es_helper.search_facts(self.es, hash_value)
                        if hit_count == 0:
                            continue

                        # parser html tags, <h1-6> <title> <p> etc.
                        # facts to id
                        facts_text, facts_weight = self.get_facts_weight(facts)
                        if len(facts_text) == 0:
                            continue

                        #  conversation keywords
                        conversation_keywords = keywords.keywords(conversation, ratio=0.5, split=True)

                        # extraction keywords
                        distances = []
                        for text, weight in zip(facts_text, facts_weight):
                            fact_keywords = keywords.keywords(text, ratio=0.5, split=True)
                            if len(fact_keywords) != 0:
                                distance = fasttext.wmdistance(' '.join(conversation_keywords), ' '.join(fact_keywords))
                                distances.append(distance / weight)
                            else:
                                distances.append(np.inf)

                        # sorted
                        sorted_indexes = np.argsort(distances)
                        topk_indexes = sorted_indexes[:f_topk]

                        topk_indexes_list = topk_indexes.tolist()
                        topk_facts_text = [facts_text[topi] for topi in topk_indexes_list]

                        topk_facts_embedded = []
                        for text in topk_facts_text:
                            fact_embedded = torch.zeros(embedding_size)
                            count = 0
                            text_keywords = keywords.keywords(text, ratio=0.5, split=True)
                            for word in ' '.join(text_keywords).split():
                                try:
                                    worde_embedded = torch.tensor(fasttext.get_vector(word)).view(-1)
                                    fact_embedded.add_(worde_embedded)
                                    count += 1
                                except KeyError:
                                    continue
                            if count != 0:
                                mean_fact_embedded = torch.div(fact_embedded, count * 1.0)
                            topk_facts_embedded.append(mean_fact_embedded)
                        topk_facts_embedded = torch.stack(topk_facts_embedded, dim=0) # [f_topk, embedding_size]

                        topk_facts_embedded_dict[hash_value]=(topk_facts_embedded, topk_facts_text)

                        """
                        facts_weight = torch.tensor(facts_weight, dtype=torch.float, device=self.device)
                        facts_ids=[fact_ids[-min(self.fact_max_len, len(facts_ids)):] for fact_ids in facts_ids]
                        # score top_k_facts_embedded -> [top_k, embedding_size]
                        topk_facts_embedded, topk_indexes = get_topk_facts(embedding_size,
                                                                            embedding,
                                                                            conversation_ids,
                                                                            facts_ids,
                                                                            topk,
                                                                            facts_weight,
                                                                            self.device)
                        """

            # save topk_facts_embedded_dict
            pickle.dump(topk_facts_embedded_dict, open(filename, 'wb'))
            self.topk_facts_embedded_dict=topk_facts_embedded_dict

    def get_facts_weight(self, facts):
        """ facts: [[w_n] * size]"""
        facts_weight=[]
        new_facts=[]
        for fact in facts:
            if len(fact) < self.min_len:
                continue
            fact_str = " ".join(fact)
            fact_weight=default_weight
            for tag, weight in tag_weight_dict.items():
                if fact_str.find(tag) != -1:
                    fact_weight=max(fact_weight, weight)
                    fact_str=fact_str.replace(tag, '')
                    fact_str=fact_str.replace(tag[0] + '/' + tag[1:], '')

            if len(fact_str.split(" ")) >= self.min_len:
                new_facts.append(fact_str)
                facts_weight.append(fact_weight)

        return new_facts,  facts_weight


    def save_generated_texts(self,
                             conversation_texts,
                             response_texts,
                             batch_generated_texts,
                             filename,
                             decode_type='greed',
                             topk_facts=None):

        with open(filename, 'a', encoding='utf-8') as f:
            for conversation, response, generated_text in zip(conversation_texts, response_texts, batch_generated_texts):
                # conversation, true response, generated_text
                f.write('Conversation: %s\n' % conversation)
                f.write('Response: %s\n' % response)
                if decode_type == 'greedy':
                    f.write('Generated: %s\n' % generated_text)
                elif decode_type == 'beam_search':
                    for i, topk_text in enumerate(generated_text):
                        f.write('Generated best_n: %d: %s\n' % (i, topk_text))

                if topk_facts is not None:
                    for fi, fact_text in enumerate(topk_facts):
                        f.write('Facts %d: %s\n' % (fi, fact_text))

                f.write('---------------------------------\n')


    def generating_texts(self, batch_utterances, batch_size, decode_type='greedy'):
        """
        decode_type == greedy:
            batch_utterances: [batch_size, max_length]
            return: [batch_size]
        decode_type == 'beam_search':
            batch_utterances: [batch_size, topk, len]
            return: [batch_size, topk]
        """

        batch_generated_texts=[]
        if decode_type == 'greedy':
            for bi in range(batch_size):
                text=self.vocab.ids_to_text(batch_utterances[bi].tolist())
                batch_generated_texts.append(text)
        elif decode_type == 'beam_search':
            for bi in range(batch_size):
                best_n_ids = batch_utterances[bi]
                best_n_texts=[]
                for ids in best_n_ids:
                    text = self.vocab.ids_to_text(ids)
                    best_n_texts.append(text)
                batch_generated_texts.append(best_n_texts)

        return batch_generated_texts
