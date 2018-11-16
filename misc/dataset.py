# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np

#  from summa import keywords # for keyword extraction
#  from rake_nltk import Rake
from misc import es_helper
#  from embedding.embedding_score import get_topk_facts

from misc.url_tags_weight import tag_weight_dict
from misc.url_tags_weight import default_weight
from misc.utils import Tokenizer


class Dataset:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 model_type,
                 pair_path,
                 c_max_len,
                 r_max_len,
                 min_len,
                 f_max_len,
                 f_topk,
                 vocab,
                 save_path,
                 turn_num,
                 min_turn,
                 turn_type,
                 eval_split,  # how many hold out as eval data
                 test_split,
                 batch_size,
                 device,
                 logger):

        self.model_type = model_type
        self.c_max_len = c_max_len
        self.r_max_len = r_max_len
        self.min_len = min_len
        self.f_max_len = f_max_len
        self.f_topk = f_topk
        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size()
        self.turn_num = turn_num
        self.min_turn = min_turn
        self.turn_type = turn_type

        self.device = device
        self.logger = logger

        # es
        self.es = es_helper.get_connection()

        self._data_dict = {}
        self._indicator_dict = {}

        self.read_txt(save_path,
                      pair_path,
                      eval_split,
                      test_split,
                      batch_size)

    def read_txt(self, save_path, pair_path, eval_split, test_split, batch_size):
        _data_dict_path = os.path.join(save_path, '_data_dict.%s.pkl' % self.turn_type)
        if not os.path.exists(_data_dict_path):
            # load source-target pairs, tokenized
            datas = []
            with open(pair_path, 'r', encoding='utf-8') as f:
                for line in f:
                    conversation_id, conversation, response, hash_value = line.rstrip().split('SPLITTOKEN')
                    if not bool(conversation) or not bool(response):
                        continue

                    response_ids = self.vocab.words_to_id(response.split(' '))
                    if len(response_ids) < self.min_len:
                        continue

                    response_ids = response_ids[-min(self.r_max_len - 1, len(response_ids)):]

                    # conversation split by EOS, START
                    if conversation.startswith('start eos'):
                        conversation = conversation[10:]
                        history_conversations = self.parser_conversations(conversation)
                    elif conversation.startswith('eos'):
                        conversation = conversation[4:]
                        history_conversations = self.parser_conversations(conversation)
                    else:
                        history_conversations = self.parser_conversations(conversation)

                    if history_conversations is None:
                        continue

                    raw_conversation = conversation

                    if self.turn_type == 'concat':
                        conversation = ' '.join(history_conversations)
                        history_conversations = [conversation]
                    elif self.turn_type == 'none':
                        conversation = history_conversations[-1]
                        history_conversations = [conversation]

                    h_conversations_ids = []
                    for history in history_conversations:
                        history_ids = self.vocab.words_to_id(history.split(' '))
                        history_ids = history_ids[-min(self.c_max_len, len(history_ids)):]
                        h_conversations_ids.append(history_ids)

                    datas.append((conversation_id, raw_conversation, h_conversations_ids, response_ids, hash_value))

            np.random.shuffle(datas)
            # train-eval split
            self.n_train = int(len(datas) * (1. - eval_split - test_split))
            self.n_eval = max(int(len(datas) * eval_split), batch_size)
            self.n_test = len(datas) - self.n_train - self.n_eval

            self._data_dict = {
                'train': datas[0: self.n_train],
                'eval': datas[self.n_train: (self.n_train + self.n_eval)],
                'test': datas[self.n_train + self.n_eval: ]
            }
            pickle.dump(self._data_dict, open(_data_dict_path, 'wb'))
        else:
            self._data_dict = pickle.load(open(_data_dict_path, 'rb'))
            self.n_train = len(self._data_dict['train'])
            self.n_test = len(self._data_dict['test'])
            self.n_eval = len(self._data_dict['eval'])

        self._indicator_dict = {
            'train': 0,
            'eval': 0,
            'test': 0
        }

    def parser_conversations(self, conversation):
        history_conversations = conversation.split('eos')
        history_conversations = [history for history in history_conversations if len(history.split()) >= self.min_len]

        if len(history_conversations) < self.min_turn:
            return None

        history_conversations = history_conversations[-min(self.turn_num, len(history_conversations)):]

        return history_conversations

    def reset_data(self, task):
        np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def load_data(self, task, batch_size):
        task_len=len(self._data_dict[task])
        if batch_size > task_len:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > task_len:
            self.reset_data(task)
            cur_indicator=batch_size

        h_inputs = list()
        h_inputs_position = list()
        h_turns_length = list()
        h_inputs_lenght = list()

        decoder_inputs = torch.zeros((self.r_max_len, batch_size),
                                     dtype=torch.long,
                                     device=self.device)

        decoder_targets = torch.zeros((self.r_max_len, batch_size),
                                      dtype=torch.long,
                                      device=self.device)

        decoder_inputs_length = list()

        conversation_texts = list()
        response_texts = list()

        # facts
        f_embedded_inputs = list()
        f_embedded_inputs_length = list()
        facts_texts = list()

        f_ids_inputs = list()
        f_ids_inputs_length = list()
        f_topks_length = list()

        batch_data = self._data_dict[task][self._indicator_dict[task]: cur_indicator]
        """sort batch_data, by turn num"""
        batch_data = sorted(batch_data, key=lambda item: len(item[2]), reverse=True)
        for i, (_, conversation_text, h_conversations_ids, response_ids, hash_value) in enumerate(batch_data):
            # ids to word
            response_text = ' '.join(self.vocab.ids_to_word(response_ids))
            response_texts.append(response_text)
            conversation_texts.append(conversation_text)

			# history inputs
            h_inputs_lenght.append(list([1]) * self.turn_num)
            h_turns_length.append(len(h_conversations_ids))

            if len(h_conversations_ids) > 0:
                h_input = torch.zeros((self.c_max_len, self.turn_num), dtype=torch.long).to(self.device) #[max_len, turn_num]
                if self.turn_type == 'transformer':
                    h_position = torch.zeros((self.c_max_len, self.turn_num), dtype=torch.long).to(self.device) #[max_len, turn_num]

                for j, ids in enumerate(h_conversations_ids):
                    h_inputs_lenght[i][j] = len(ids)

                    tmp_i = torch.zeros(self.c_max_len, dtype=torch.long, device=self.device)
                    tmp_h = torch.zeros(self.c_max_len, dtype=torch.long, device=self.device)
                    for k, id in enumerate(ids):
                        tmp_i[k] = id
                        if self.turn_type == 'transformer':
                            h_position[k] = k + 1

                    h_input[:, j] = tmp_i
                    if self.turn_type == 'transformer':
                        h_position[:, j] = tmp_h

                h_inputs.append(h_input)
                if self.turn_type == 'transformer':
                    h_inputs_position.append(h_position)
                    #  print(h_position)
                #  print(h_input)

            # decoder_inputs
            decoder_inputs[0, i] = self.vocab.sosid
            for r, token_id in enumerate(response_ids):
                decoder_inputs[r + 1, i] = token_id
                decoder_targets[r, i] = token_id

            decoder_targets[len(response_ids), i] = self.vocab.eosid
            decoder_inputs_length.append(len(response_ids) + 1)

            if self.model_type == 'kg':
                topk_facts_embedded, topk_facts_text = self.assembel_facts(hash_value)

                f_ids_input = torch.zeros((self.f_max_len, self.f_topk),
                                            dtype=torch.long,
                                            device=self.device)

                f_ids_input_length = torch.ones(self.f_topk,
                                                 dtype=torch.long,
                                                 device=self.device)

                if topk_facts_text is not None:
                    topk_facts_ids = [self.vocab.words_to_id(text.split(' ')) for text in topk_facts_text]
                    f_topks_length.append(min(len(topk_facts_ids), self.f_topk))
                    for fi, ids in enumerate(topk_facts_ids[:self.f_topk]):
                        ids = ids[-min(self.f_max_len, len(ids)):]
                        f_ids_input_length[fi] = len(ids)
                        for fj, id in enumerate(ids):
                            f_ids_input[fj, fi] = id
                else:
                    f_topks_length.append(1)

                f_ids_inputs.append(f_ids_input)
                f_ids_inputs_length.append(f_ids_input_length)

                facts_texts.append(topk_facts_text)

                if topk_facts_embedded is not None:
                    topk_facts_embedded = topk_facts_embedded.squeeze(1)
                    if topk_facts_embedded.size(0) < self.f_topk:
                        tmp_tensor = torch.zeros((self.f_topk, self.pre_embedding_size))
                        tmp_tensor[:topk_facts_embedded.size(0)] = topk_facts_embedded
                    elif topk_facts_embedded.size(0) >= self.f_topk:
                        tmp_tensor = topk_facts_embedded[:self.f_topk]

                    tmp_tensor = torch.zeros((self.f_topk, self.pre_embedding_size))

                    tmp_tensor = tmp_tensor.to(self.device)
                    f_embedded_inputs.append(tmp_tensor)

                    f_embedded_inputs_length.append(topk_facts_embedded.size(0))


        h_inputs = torch.stack(h_inputs, dim=1) # [max_len, batch_size, turn_num]
        if self.turn_type == 'transformer':
            h_inputs_position = torch.stack(h_inputs_position, dim=1) # [max_len, batch_size, turn_num]

        h_turns_length = torch.tensor(h_turns_length, dtype=torch.long, device=self.device)
        h_inputs_lenght = torch.tensor(h_inputs_lenght, dtype=torch.long, device=self.device) #[batch_size, turn_num]

        decoder_inputs_length = torch.tensor(decoder_inputs_length, dtype=torch.long, device=self.device) #[batch_size]

        if self.model_type == 'kg':
            if len(f_embedded_inputs) > 0:
                f_embedded_inputs = torch.stack(f_embedded_inputs, dim=0) #[batch_size, topk, pre_embedding_size]
                f_embedded_inputs_length = torch.tensor(f_embedded_inputs_length, dtype=torch.long, device=self.device)

            if len(f_ids_inputs) > 0:
                f_ids_inputs = torch.stack(f_ids_inputs, dim=1) # [f_max_len, batch_size, f_topk]
                f_ids_inputs_length = torch.stack(f_ids_inputs_length, dim=0) # [batch_size, f_topk]

            f_topks_length = torch.tensor(f_topks_length, dtype=torch.long, device=self.device)

        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator

        return decoder_inputs, decoder_targets, decoder_inputs_length, \
            conversation_texts, response_texts, \
            f_embedded_inputs, f_embedded_inputs_length, \
            f_ids_inputs, f_ids_inputs_length, f_topks_length, facts_texts, \
            h_inputs, h_turns_length, h_inputs_lenght, h_inputs_position


    def assembel_facts(self, hash_value):
        # load top_k facts
        topk_facts_embedded, topk_facts = self.topk_facts_embedded_dict.get(hash_value, (None, None))
        if topk_facts_embedded is None:
            #  return (torch.zeros((self.f_topk, self.pre_embedding_size), device=self.device), [])
            return None, None

        return topk_facts_embedded, topk_facts

    def build_similarity_facts_offline(self,
                                      facts_dict,
                                      fasttext,
                                      pre_embedding_size,
                                      f_topk,
                                      filename):

        tokenizer = Tokenizer()
        #  rake = Rake()

        self.pre_embedding_size = pre_embedding_size

        if os.path.exists(filename):
            self.topk_facts_embedded_dict=pickle.load(open(filename, 'rb'))
        else:
            topk_facts_embedded_dict={}
            with torch.no_grad():
                for task, datas in self._data_dict.items():
                    self.logger.info('computing similarity: %s ' % task)
                    for conversation_id, conversation, _, _, hash_value in datas:
                        if not bool(conversation) or not bool(hash_value):
                            continue

                        facts = facts_dict.get(conversation_id, None)

                        if facts is None or len(facts) == 0:
                            continue

                        #  conversation_keywords = keywords.keywords(conversation, ratio=0.7, split=True)
                        #  rake.extract_keywords_from_text(conversation)
                        #  conversation_keywords = rake.get_ranked_phrases()
                        #  print('conversation_keywords: {}'.format(conversation_keywords))
                        conversation_embedded = self.get_sentence_embedded(conversation.split(), fasttext)

                        #  distances = []

                        # tokenizer
                        facts_words = [tokenizer.tokenize(fact) for fact in facts]

                        facts_embedded = list()
                        for fact_words in facts_words:
                            fact_embedded = self.get_sentence_embedded(fact_words, fasttext)
                            facts_embedded.append(fact_embedded)

                            """
                            if len(fact_words) != 0:
                                distance = fasttext.wmdistance(' '.join(conversation_keywords), ' '.join(fact_words))
                                distances.append(distance)
                            else:
                                distances.append(np.inf)
                            """
                        facts_embedded = torch.stack(facts_embedded, dim=0)
                        similarities = F.cosine_similarity(conversation_embedded.view(1, -1), facts_embedded) # [len]

                        #  distances = np.array(distances)
                        #  sorted_indexes = np.argsort(distances)

                        _, sorted_indexes = similarities.sort(dim=0, descending=True)

                        topk_indexes = sorted_indexes[:f_topk]
                        #  topk_distances = distances[topk_indexes]

                        topk_facts_words = [facts_words[topi] for topi in topk_indexes.tolist()]
                        topk_facts_text = [' '.join(fact_words) for fact_words in topk_facts_words]

                        topk_facts_embedded_dict[hash_value]=(None, topk_facts_text)

                        """
                        topk_facts_embedded = []
                        for words, distance in zip(topk_facts_words, topk_distances):
                            #  fact_embedded = torch.zeros(pre_embedding_size)
                            fact_embedded = list()
                            fact_distance = []

                            count = 0.0
                            for word in words:
                                try:
                                    word_embedded = torch.tensor(fasttext.get_vector(word), device=self.device).view(-1)
                                    #  fact_embedded.add_(word_embedded)
                                    fact_embedded.append(word_embedded)
                                    fact_distance.append(-distance)
                                    count += 1.0
                                except KeyError:
                                    continue

                            if count > 0:
                                #  fact_embedded = torch.div(fact_embedded, count)
                                fact_embedded = torch.stack(fact_embedded, dim=0) # [topk, embedding_size]
                                fact_distance = torch.tensor(fact_distance, device=self.device).view(1, -1)

                                fact_weight = torch.softmax(fact_distance, dim=1)
                                fact_embedded = torch.mm(fact_weight, fact_embedded) # [1, embedding_size]

                                topk_facts_embedded.append(fact_embedded)
                        if len(topk_facts_embedded) > 0:
                            topk_facts_embedded = torch.stack(topk_facts_embedded, dim=0) # [f_topk, pre_embedding_size]
                            topk_facts_embedded_dict[hash_value]=(topk_facts_embedded, topk_facts_text)

                        """


            # save topk_facts_embedded_dict
            pickle.dump(topk_facts_embedded_dict, open(filename, 'wb'))
            self.topk_facts_embedded_dict=topk_facts_embedded_dict

    def get_sentence_embedded(self, words, fasttext):
        sentence_embedded = list()
        for word in words:
            try:
                word_embedded = torch.tensor(fasttext.get_vector(word), device=self.device).view(-1)
                sentence_embedded.append(word_embedded)
            except KeyError:
                continue

        if len(sentence_embedded) > 0:
            sentence_embedded = torch.stack(sentence_embedded, dim=0) # [len, pre_embedding_size]
            mean_sentence_embedded = sentence_embedded.mean(dim=0) # [pre_embedding_size]
        else:
            mean_sentence_embedded = torch.zeros(self.pre_embedding_size, device=self.device)
        return mean_sentence_embedded

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
                             greedy_texts,
                             beam_texts,
                             filename,
                             decode_type='greed',
                             facts_texts=None):

        #  print(facts_texts)
        with open(filename, 'a', encoding='utf-8') as f:
            for i, (conversation, response, greedy_text, beam_text) in enumerate(zip(conversation_texts, response_texts, greedy_texts, beam_texts)):
                f.write('Conversation:\t %s\n' % conversation)
                f.write('Response:\t %s\n' % response)

                f.write('greedy:\t %s\n' % greedy_text)

                for i, best_text in enumerate(beam_text):
                    #  print('best_text: {}'.format(best_text))
                    f.write('beam %d:\t %s\n' % (i, best_text))

                if facts_texts is not None and len(facts_texts) > 0:
                    topk_facts = facts_texts[i]
                    if topk_facts is not None:
                        for fi, fact_text in enumerate(topk_facts):
                            f.write('Fact %d:\t %s\n' % (fi, fact_text))

                f.write('---------------------------------\n')

    def generating_texts(self, batch_utterances, batch_size, decode_type='greedy'):
        """
        decode_type == greedy:
            batch_utterances: [batch_size, max_len]
            return: [batch_size]
        decode_type == 'beam_search':
            batch_utterances: [batch_size, topk, max_len]
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
                #  print('best_n_ids: {}'.format(best_n_ids))
                best_n_texts=[]
                for ids in best_n_ids:
                    #  print('ids: {}'.format(ids))
                    text = self.vocab.ids_to_text(ids)
                    #  print('text: {}'.format(text))
                    best_n_texts.append(text)
                batch_generated_texts.append(best_n_texts)

        return batch_generated_texts
