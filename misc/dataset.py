# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from misc import es_helper

from misc.url_tags_weight import tag_weight_dict
from misc.url_tags_weight import default_weight
from misc.utils import Tokenizer
from misc.vocab import EOS_ID, PAD_ID
from modules.utils import to_device


class Dataset:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 opt,
                 vocab,
                 device,
                 logger):

        self.opt = opt
        self.vocab = vocab
        self.vocab_size = vocab.size

        self.device = device
        self.logger = logger

        # es
        self.es = es_helper.get_connection()

        self._data_dict = {}
        self._indicator_dict = {}

        self.read_txt(opt)

    def read_txt(self, opt):
        _data_dict_path = os.path.join(opt.save_path, '_data_dict.pkl')
        if not os.path.exists(_data_dict_path):
            datas = []
            with open(opt.pair_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    conversation_id, context, response, hash_value = line.rstrip().split('SPLITTOKEN')
                    if not bool(context) or not bool(response):
                        continue

                    response_ids = self.vocab.words_to_id(response.split(' '))
                    if len(response_ids) < opt.min_unroll or len(response_ids) > opt.max_unroll + 50:
                        continue

                    response_ids = response_ids[:min(opt.max_unroll - 1, len(response_ids))]
                    #  response_ids.insert(0, SOS_ID)
                    response_ids.append(EOS_ID)
                    response_len = len(response_ids)
                    response_ids.extend(list([PAD_ID]) * (opt.max_unroll - response_len))

                    # context split by EOS, START
                    if context.startswith('start eos'):
                        context = context[10:]
                        sentences = self.parser_conversation(context)
                        if len(sentences) > 2:
                            sentences = sentences[1:]
                    elif context.startswith('eos'):
                        context = context[4:]
                        sentences = self.parser_conversation(context)
                    elif context.startswith('... eos'):
                        context = context[7:]
                        sentences = self.parser_conversation(context)
                    elif context.startswith('... '):
                        context = context[4:]
                        sentences = self.parser_conversation(context)
                    else:
                        sentences = self.parser_conversation(context)

                    if sentences is None or len(sentences) < opt.min_turn:
                        continue

                    sentences = sentences[-min(opt.turn_num, len(sentences)):]

                    sentence_texts = []
                    sentence_ids = []
                    sentence_lengths = []
                    for s in sentences:
                        ids = self.vocab.words_to_id(s.split(' '))
                        ids = ids[:min(opt.max_unroll - 1, len(ids))]
                        #  ids.insert(0, SOS_ID)
                        ids.append(EOS_ID)
                        sentence_lengths.append(len(ids))

                        ids.extend(list([PAD_ID]) * (opt.max_unroll - len(ids)))

                        sentence_ids.append(ids)
                        sentence_texts.append(' '.join(self.vocab.ids_to_word(ids)))

                    sentence_ids.append(response_ids)
                    sentence_lengths.append(response_len)
                    sentence_texts.append(' '.join(self.vocab.ids_to_word(response_ids)))
                    #  print(sentence_lengths)

                    datas.append((conversation_id, sentence_texts, sentence_ids, sentence_lengths, hash_value))

            np.random.shuffle(datas)

            # train-eval split
            self.n_train = int(len(datas) * (1. - opt.eval_split - opt.test_split))
            self.n_eval = max(int(len(datas) * opt.eval_split), opt.batch_size)
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

    def parser_conversation(self, context):
        sentences = context.split('eos')
        sentences = [s for s in sentences if len(s.split()) > self.opt.min_unroll]
        return sentences

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
            cur_indicator=batch_size

        input_sentences = list()
        target_sentences = list()

        input_sentence_length = list()
        target_sentence_length = list()

        input_conversation_length = list()
        conversation_texts = list()

        batch_data = self._data_dict[task][self._indicator_dict[task]: cur_indicator]
        for i, (_, sentence_texts, sentence_ids, sentence_lengths, hash_value) in enumerate(batch_data):
            input_sentences.extend(sentence_ids[:-1])
            target_sentences.extend(sentence_ids[1:])

            input_sentence_length.extend(sentence_lengths[:-1])
            target_sentence_length.extend(sentence_lengths[1:])

            input_conversation_length.append(len(sentence_ids) - 1)

            conversation_texts.append(sentence_texts)

        input_sentences = to_device(torch.LongTensor(input_sentences))
        target_sentences = to_device(torch.LongTensor(target_sentences))
        input_sentence_length = to_device(torch.LongTensor(input_sentence_length))
        target_sentence_length = to_device(torch.LongTensor(target_sentence_length))
        input_conversation_length = to_device(torch.LongTensor(input_conversation_length))

        return input_sentences, target_sentences, \
                input_sentence_length, target_sentence_length, \
                input_conversation_length, conversation_texts

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
            if len(fact) < self.opt.min_unroll:
                continue
            fact_str = " ".join(fact)
            fact_weight=default_weight
            for tag, weight in tag_weight_dict.items():
                if fact_str.find(tag) != -1:
                    fact_weight=max(fact_weight, weight)
                    fact_str=fact_str.replace(tag, '')
                    fact_str=fact_str.replace(tag[0] + '/' + tag[1:], '')

            if len(fact_str.split(" ")) >= self.opt.min_unroll:
                new_facts.append(fact_str)
                facts_weight.append(fact_weight)

        return new_facts,  facts_weight

    def save_generated_texts(self,
                             conversation_texts=[],
                             beam_texts=[],
                             decode_type='greedy',
                             filename=None,
                             facts_texts=None):

        #  print(facts_texts)
        with open(filename, 'a', encoding='utf-8') as f:
            for i, (sentence_texts, beam_text) in enumerate(zip(conversation_texts, beam_texts)):
                for sentence in sentence_texts:
                    f.write('> %s\n' % sentence)

                for i, text in enumerate(beam_text):
                    f.write('beam %d > %s\n' % (i, text))

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
                best_n_texts=[]
                for ids in best_n_ids:
                    text = self.vocab.ids_to_text(ids)
                    best_n_texts.append(text)
                batch_generated_texts.append(best_n_texts)

        return batch_generated_texts
