# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import re
import os
import pickle

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from rake_nltk import Rake
from misc import es_helper
from misc.utils import remove_stop_words

from misc.url_tags_weight import tag_weight_dict
from misc.url_tags_weight import default_weight
from misc.vocab import PAD_ID, SOS_ID, EOS_ID


class Dataset:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 config,
                 vocab,
                 device,
                 logger):

        self.config = config
        self.vocab = vocab
        self.device = device
        self.logger = logger

        self._data_dict = {}
        self._indicator_dict = {}

        self.read_data()

    def read_data(self):
        self.logger.info('read data...')
        _data_dict_path = os.path.join(
            self.config.save_path, '_data_dict.%s_%s.pkl' % (self.config.turn_min, self.config.turn_num))
        if not os.path.exists(_data_dict_path):
            datas = []
            batch_size = self.config.batch_size
            min_len = self.config.min_len
            c_max_len, q_max_len, r_max_len = self.config.c_max_len, self.config.q_max_len, self.config.r_max_len
            with open(self.config.pair_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    line = line.rstrip()
                    if not bool(line):
                        continue

                    subreddit_name, conversation_id, context, query, \
                        response, hash_value, score, turn = line.split(
                            ' SPLIT ')

                    if not bool(query) or not bool(response) \
                            or len(query.split()) < min_len \
                            or len(response.split()) < min_len:
                        continue

                    # query
                    words = [word for word in query.split()
                             if len(word.split()) > 0]
                    query_ids = self.vocab.words_to_id(words)
                    if len(query_ids) < min_len:
                        continue
                    query_ids = query_ids[-min(q_max_len, len(query_ids)):]

                    # response
                    words = [word for word in response.split()
                             if len(word.split()) > 0]
                    response_ids = self.vocab.words_to_id(words)
                    if len(response_ids) < min_len:
                        continue
                    response_ids = response_ids[:min(
                        r_max_len, len(response_ids))]

                    # context split by EOS
                    context_sentences = self.parser_context(context)
                    if context_sentences is None or len(context_sentences) < self.config.turn_min:
                        continue

                    context_sentences = context_sentences[-min(
                        self.config.turn_num, len(context_sentences)):]
                    #  print('context: ', context)
                    #  print('context_sentences: ', context_sentences)

                    context_ids = []
                    for si, sentence in enumerate(context_sentences):
                        words = [word for word in sentence.split()
                                 if len(word.split()) > 0]
                        ids = self.vocab.words_to_id(words)
                        if si % 2 == 0:  # start
                            ids = ids[-min(c_max_len, len(ids)):]
                        else:  # reply
                            ids = ids[:min(c_max_len, len(ids))]

                        context_ids.append(ids)

                    datas.append((subreddit_name, conversation_id,
                                  context_ids, query_ids, response_ids, hash_value))

            np.random.shuffle(datas)
            # train-eval split
            n_eval = self.config.eval_batch * batch_size
            n_train = int(len(datas) * (1. - self.config.test_split) - n_eval)
            n_test = len(datas) - n_train - n_eval
            self._size_dict = {
                'train': n_train,
                'eval': n_eval,
                'test': n_test
            }

            self._data_dict = {
                'train': datas[0: n_train],
                'eval': datas[n_train: (n_train + n_eval)],
                'test': datas[n_train + n_eval:]
            }
            pickle.dump(self._data_dict, open(_data_dict_path, 'wb'))
        else:
            self._data_dict = pickle.load(open(_data_dict_path, 'rb'))
            self._size_dict = {
                'train': len(self._data_dict['train']),
                'eval': len(self._data_dict['eval']),
                'test': len(self._data_dict['test'])
            }

        self._indicator_dict = {
            'train': 0,
            'eval': 0,
            'test': 0
        }

    def parser_context(self, context):
        sentences = context.split(' EOS ')
        context_sentences = list()
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < self.config.min_len:
                if i != len(sentences) - 1:
                    context_sentences = list()
                    continue
                else:
                    continue
            else:
                context_sentences.append(sentence)
        return context_sentences

    def reset_data(self, task, shuffle=True):
        if shuffle:
            np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def load_data(self, task):
        batch_size = self.config.batch_size
        c_max_len, q_max_len, r_max_len = self.config.c_max_len, self.config.q_max_len, self.config.r_max_len
        f_max_len = self.config.f_max_len
        if batch_size > self._size_dict[task]:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > self._size_dict[task]:
            if task == 'train' or task = 'test':
                self.reset_data(task, True)
                cur_indicator = batch_size
            else:
                self.reset_data(task, False)
                
        q_inputs = list()
        q_inputs_length = list()

        c_inputs = list()
        c_inputs_length = list()
        c_turn_length = list()

        dec_inputs = list()

        subreddit_names = list()
        conversation_ids = list()
        hash_values = list()

        f_inputs = list()
        f_inputs_length = list()
        f_topk_length = list()

        batch_data = self._data_dict[task][self._indicator_dict[task]: cur_indicator]

        """sort batch_data, by turn num"""
        batch_data = sorted(
            batch_data, key=lambda item: len(item[3]), reverse=True)
        for i, (subreddit_name, conversation_id, context_ids, query_ids, response_ids, hash_value) in enumerate(batch_data):
            subreddit_names.append(subreddit_name)
            conversation_ids.append(conversation_id)
            hash_values.append(hash_value)

            if self.config.turn_type not in ['none', 'concat']:
                # c inputs
                c_turn_length.append(len(context_ids))
                c_inputs_length.append(list([1]) * (self.config.turn_num))

                context_ids = torch.LongTensor([
                    ids + [PAD_ID] * (c_max_len - len(ids))
                    for ids in context_ids
                ]).to(self.device)
                c_input = torch.zeros((self.config.turn_num, c_max_len),
                                      dtype=torch.long).to(self.device)
                c_input[:len(context_ids), :] = context_ids
                c_inputs.append(c_input)

            if self.config.turn_type == 'concat':
                concat_ids = []
                for ids in context_ids:
                    concat_ids.extend(ids)
                concat_ids.extend(query_ids)
                query_ids = concat_ids[-min(q_max_len, len(concat_ids)):]

            # q inputs
            q_inputs_length.append(len(query_ids))
            #  print('query_ids: ', query_ids)
            q_input = torch.LongTensor(
                query_ids + [PAD_ID] * (q_max_len - len(query_ids))).to(self.device)
            q_inputs.append(q_input)

            # dec_inputs
            dec_input = torch.LongTensor([SOS_ID] + response_ids + [EOS_ID]
                                         + [PAD_ID] * (r_max_len - len(response_ids))).to(self.device)
            dec_inputs.append(dec_input)

            if self.config.model_type == 'kg':
                topk_facts_text = self.facts_topk_phrases.get(hash_value, None)
                f_input = torch.zeros((self.config.f_topk, f_max_len),
                                      dtype=torch.long,
                                      device=self.device)

                f_input_length = torch.ones(self.config.f_topk,
                                            dtype=torch.long,
                                            device=self.device)

                if topk_facts_text is not None:
                    topk_facts_ids = [self.vocab.words_to_id(
                        text.split()) for text in topk_facts_text]
                    f_topk_length.append(
                        min(len(topk_facts_ids), self.config.f_topk))
                    for fi, ids in enumerate(topk_facts_ids[:self.config.f_topk]):
                        ids = ids[:min(f_max_len, len(ids))]
                        f_input_length[fi] = len(ids)
                        for fj, id in enumerate(ids):
                            f_input[fi, fj] = id
                else:
                    f_topk_length.append(1)

                f_inputs.append(f_input)
                f_inputs_length.append(f_input_length)

        # q [max_len, batch_size]
        q_inputs = torch.stack(q_inputs, dim=1)
        q_inputs_length = torch.tensor(
            q_inputs_length, dtype=torch.long, device=self.device)

        if self.config.turn_type not in ['none', 'concat']:
            # [turn_num, max_len, batch_size]
            c_inputs = torch.stack(c_inputs, dim=2)
            c_turn_length = torch.tensor(
                c_turn_length, dtype=torch.long, device=self.device)  # [batch_size]
            # [turn_num, batch_size]
            c_inputs_length = torch.tensor(
                c_inputs_length, dtype=torch.long, device=self.device).transpose(0, 1)
        # decoder [max_len, batch_size]
        dec_inputs = torch.stack(dec_inputs, dim=1)

        if self.config.model_type == 'kg':
            # [topk, batch_size, max_len]
            f_inputs = torch.stack(f_inputs, dim=1)
            f_inputs_length = torch.stack(
                f_inputs_length, dim=1)  # [f_topk, batch_size]

            f_topk_length = torch.tensor(
                f_topk_length, dtype=torch.long, device=self.device)

        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator

        return dec_inputs, \
            q_inputs, q_inputs_length, \
            c_inputs, c_inputs_length, c_turn_length, \
            f_inputs, f_inputs_length, f_topk_length, \
            subreddit_names, conversation_ids, hash_values

    def load_similarity_facts(self, offline_filename):
        self.facts_topk_phrases = pickle.load(open(offline_filename, 'rb'))

    def retrieval_similarity_facts(self, offline_filename):
        es = es_helper.get_connection()
        r = Rake(
            min_length=1,
            max_length=self.config.f_max_len
        )
        facts_topk_phrases = {}
        for task, task_datas in self._data_dict.items():
            self.logger.info('retrieval similarity facts for: %s data.' % task)
            for data in tqdm(task_datas):
                conversation_id, sentences_text, _, _, hash_value = data
                #  sentences_text = [' '.join(remove_stop_words(sentence.split())) for sentence in sentences_text]
                sentences_text = [re.sub(r'<number>|<url>|<unk>', '', sentence)
                                  for sentence in sentences_text]
                #  print('sentences_text: ', sentences_text)
                if len(sentences_text) >= 2:
                    r.extract_keywords_from_sentences(sentences_text[:-1])
                    ranked_text = ' '.join(r.get_ranked_phrases())
                    query_text = ranked_text + ' ' + sentences_text[-1]
                else:
                    query_text = ' '.join(sentences_text)

                #  print('query_text: ', query_text)
                query_body = es_helper.assemble_search_fact_body(
                    conversation_id, query_text)
                hits, hit_count = es_helper.search(
                    es, es_helper.index, es_helper.fact_type, query_body)
                if hit_count == 0:
                    continue

                phrases = list()
                for hit in hits[:self.config.f_topk]:
                    hit_conversation_id = hit['_source']['conversation_id']
                    assert (conversation_id == hit_conversation_id), "%s != %s" % (
                        conversation_id, hit_conversation_id)
                    phrase = hit['_source']['text']
                    if len(phrase) <= self.config.f_max_len:
                        phrases.append(phrase)
                        continue

                    parts = [part for part in phrase.split(
                        '.') if len(part.split()) > 2]
                    if len(parts) == 0:
                        phrases.append(phrase)
                    elif len(parts) == 1:
                        phrases.append(parts[0] + ' .')
                    else:
                        #  phrases.append(parts[-1])
                        phrases.append('.'.join(parts))

                #  print('conversation_id: ', conversation_id)
                #  print('phrases: {}'.format(phrases))

                facts_topk_phrases[hash_value] = phrases

        pickle.dump(facts_topk_phrases, open(offline_filename, 'wb'))
        self.facts_topk_phrases = facts_topk_phrases

    def build_similarity_facts_offline(self,
                                       facts_dict=None,
                                       offline_filename=None,
                                       embedding=None):
        if os.path.exists(offline_filename):
            self.load_similarity_facts(offline_filename)
            return

        offline_type = self.config.offline_type
        if offline_type in ['elastic', 'elastic_tag']:
            self.retrieval_similarity_facts(offline_filename)
            return

        ranked_phrase_dict_path = self.config.save_path + \
            'ranked_phrase_dict.%s.pkl' % offline_type
        ranked_phrase_embedded_dict_path = self.config.save_path + \
            'ranked_phrase_embedded_dict.%s.pkl' % offline_type
        try:
            self.logger.info('load ranked phrase...')
            ranked_phrase_dict = pickle.load(
                open(ranked_phrase_dict_path, 'rb'))
            ranked_phrase_embedded_dict = pickle.load(
                open(ranked_phrase_embedded_dict_path, 'rb'))
        except FileNotFoundError as e:
            r = Rake(
                min_length=1,
                max_length=self.config.f_max_len
            )
            ranked_phrase_dict = {}
            ranked_phrase_embedded_dict = {}
            for conversation_id, ps in tqdm(facts_dict.items()):
                if len(ps) == 0:
                    continue
                r.extract_keywords_from_sentences(ps)
                phrases = r.get_ranked_phrases()
                if len(phrases) == 0:
                    continue
                ranked_phrase_dict[conversation_id] = phrases
                phrase_embeddeds = list()
                for phrase in phrases:
                    if offline_type == 'fasttext':
                        ids = self.vocab.words_to_id(phrase.split())
                        mean_embedded = self.get_sentence_embedded(
                            ids, embedding, offline_type)
                    elif offline_type == 'elmo':
                        mean_embedded = self.get_sentence_embedded(
                            phrase.split(), embedding, offline_type)

                    phrase_embeddeds.append(mean_embedded)
                #  phrase_embeddeds = torch.stack(phrase_embeddeds, dim=0) # [len(phrase), pre_embedding_size]
                ranked_phrase_embedded_dict[conversation_id] = phrase_embeddeds

            pickle.dump(ranked_phrase_dict, open(
                ranked_phrase_dict_path, 'wb'))
            pickle.dump(ranked_phrase_embedded_dict, open(
                ranked_phrase_embedded_dict_path, 'wb'))

        # embedding match
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        facts_topk_phrases = {}
        for task, task_datas in self._data_dict.items():
            self.logger.info('build similarity facts for: %s data.' % task)
            for data in tqdm(task_datas):
                conversation_id, sentences_text, _, _, hash_value = data
                if not bool(hash_value):
                    continue

                # [len(phrases), pre_embedding_size]
                phrase_embeddeds = ranked_phrase_embedded_dict.get(
                    conversation_id, None)
                if phrase_embeddeds is None or len(phrase_embeddeds) == 0:
                    continue
                phrases = ranked_phrase_dict.get(conversation_id, None)
                if phrases is None or len(phrases) == 0:
                    continue

                sum_scores = list()
                for sentence in sentences_text:
                    sentence_words = remove_stop_words(sentence.split())
                    sentence_ids = self.vocab.words_to_id(sentence_words)
                    mean_embedded = self.get_sentence_embedded(
                        sentence_ids, embedding, offline_type)
                    scores = list()
                    for phrase_embedded in phrase_embeddeds:
                        #  print(phrase_embedded.shape)
                        #  print(mean_embedded.shape)
                        scores.append(cos(mean_embedded.view(-1),
                                          phrase_embedded.view(-1)))
                    scores = torch.stack(scores, dim=0)
                    sum_scores.append(scores)
                # [len(sentences), len(phrase_embeddeds)]
                sum_scores = torch.stack(sum_scores)
                #  print('sum_scores: ', sum_scores.shape)
                # [len(phrase_embeddeds)]
                sum_score = sum_scores.sum(dim=0)
                _, indexes = sum_score.topk(
                    min(self.config.f_topk, sum_score.size(0)), dim=0)

                topk_phrases = list()
                #  print('phrases: %d' % len(phrases))
                #  print('indexes: %d' % indexes.size(0))
                for index in indexes.tolist():
                    if index >= len(phrases):
                        continue
                    topk_phrases.append(phrases[index])

                facts_topk_phrases[hash_value] = phrases

                del phrase_embeddeds
                del phrases

        # save topk_facts_embedded_dict
        pickle.dump(facts_topk_phrases, open(offline_filename, 'wb'))
        self.facts_topk_phrases = facts_topk_phrases

    def get_sentence_embedded(self, ids, embedding, offline_type):
        if offline_type == 'fasttext':
            ids = torch.LongTensor(ids).to(self.device)
            embeddeds = embedding(ids)  # [len(ids), pre_embedding_size]
            mean_embedded = embeddeds.mean(dim=0)  # [pre_embedding_size]
            return mean_embedded
        elif offline_type == 'elmo':
            tokens = ids  # if offline_type is elmo, ids = words
            vectors = embedding.embed_sentence(tokens)
            assert(len(vectors) == 3)
            assert(len(vectors[0]) == len(tokens))
            # vectors: [3, len(tokens), 1024]
            mean_embedded = vectors[2].mean(dim=0)
            return mean_embedded

    def get_facts_weight(self, facts):
        """ facts: [[w_n] * size]"""
        facts_weight = []
        new_facts = []
        for fact in facts:
            if len(fact) < self.config.min_len:
                continue
            fact_str = " ".join(fact)
            fact_weight = default_weight
            for tag, weight in tag_weight_dict.items():
                if fact_str.find(tag) != -1:
                    fact_weight = max(fact_weight, weight)
                    fact_str = fact_str.replace(tag, '')
                    fact_str = fact_str.replace(tag[0] + '/' + tag[1:], '')

            if len(fact_str.split(" ")) >= self.config.min_len:
                new_facts.append(fact_str)
                facts_weight.append(fact_weight)

        return new_facts, facts_weight

    def save_generated_texts(self,
                             epoch,
                             names,
                             ids,
                             hash_values,
                             q_inputs,
                             c_inputs,
                             f_inputs,
                             r_inputs,
                             greedy_texts,
                             beam_texts,
                             topk_texts,
                             filename,
                             time_str):

        q_inputs = q_inputs.transpose(0, 1).tolist()  # [batch_size, max_len]
        r_inputs = r_inputs.transpose(0, 1).tolist()  # [batch_size, max_len]

        if len(c_inputs) == 0:
            c_inputs = [None] * self.config.batch_size
        else:
            # c [batch_size, turn_num, max_len]
            c_inputs = c_inputs.permute(2, 0, 1).tolist()

        if len(f_inputs) == 0:
            f_inputs = [None] * self.config.batch_size
        else:
            # [batch_size, topk, max_len]
            f_inputs = f_inputs.transpose(0, 1).tolist()

        if topk_texts is None:
            topk_texts = [None] * self.config.batch_size

        ground_truth_path = os.path.join(self.config.save_path, 'ground_truth/%s_%s.txt' % (
            self.config.turn_min, self.config.turn_num
        ))

        predicted_path = os.path.join(self.config.save_path, 'predicted/%s_%s_%s_%s_%s.txt' % (
            self.config.turn_type, epoch, self.config.turn_min, self.config.turn_num, time_str
        ))

        ground_truth_f = open(ground_truth_path, 'w')
        predicted_f = open(predicted_path, 'w')

        with open(filename, 'w', encoding='utf-8') as f:
            for subreddit_name, conversation_id, hash_value, \
                q_ids, c_ids, f_ids, r_ids, g_text, b_text, t_text in \
                    zip(names,
                        ids,
                        hash_values,
                        q_inputs,
                        c_inputs,
                        f_inputs,
                        r_inputs,
                        greedy_texts,
                        beam_texts,
                        topk_texts):

                f.write('subreddit_name: %s\n' % subreddit_name)
                f.write('conversation_id: %s\n' % conversation_id)
                f.write('hash_value: %s\n' % hash_value)

                if c_ids is not None:
                    for ci, ids in enumerate(c_ids):
                        text = self.vocab.ids_to_text(ids)
                        f.write('c %d: %s\n' % (ci, text))

                query_text = self.vocab.ids_to_text(q_ids)
                f.write('query: %s\n' % query_text)

                response_text = self.vocab.ids_to_text(r_ids)
                f.write('\n')
                f.write('response: %s\n' % response_text)

                f.write('greedy: %s\n' % g_text)

                # for embedding metrics
                ground_truth_f.write('%s\n' % response_text)
                predicted_f.write('%s\n' % g_text)

                for i, text in enumerate(b_text):
                    f.write('beam %d: %s\n' % (i, text))

                if t_text is not None:
                    for i, text in enumerate(t_text):
                        f.write('topk %d: %s\n' % (i, text))

                if f_ids is not None:
                    for fi, ids in enumerate(f_ids):
                        text = self.vocab.ids_to_text(ids)
                        f.write('fact %d: %s\n' % (fi, text))

                f.write('-' * 70 + '\n')

        ground_truth_f.close()
        predicted_f.close()

    def generating_texts(self, outputs, outputs_length=None, decode_type='greedy'):
        """ decode_type == greedy:
            outputs: [batch_size, max_len]
            return: [batch_size]
        decode_type == 'beam_search':
            outputs: [batch_size, topk, max_len]
            outputs_length: [batch_size, topk]
            return: [batch_size, topk]
        """

        batch_generated_texts = []
        if decode_type == 'greedy':
            for bi in range(self.config.batch_size):
                text = self.vocab.ids_to_text(outputs[bi].tolist())
                batch_generated_texts.append(text)
        elif decode_type == 'beam_search':
            for bi in range(self.config.batch_size):
                topk_ids = outputs[bi]
                topk_texts = []
                if outputs_length is not None:
                    topk_length = outputs_length[bi]
                    for ids, length in zip(topk_ids, topk_length):
                        text = self.vocab.ids_to_text(ids[:length])
                        topk_texts.append(text)
                else:
                    for ids in topk_ids:
                        text = self.vocab.ids_to_text(ids)
                        topk_texts.append(text)

                batch_generated_texts.append(topk_texts)

        return batch_generated_texts
