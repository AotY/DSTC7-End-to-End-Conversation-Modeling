# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import pickle

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

#  from summa import keywords # for keyword extraction
from rake_nltk import Rake
from misc import es_helper
from misc.utils import remove_stop_words

from misc.url_tags_weight import tag_weight_dict
from misc.url_tags_weight import default_weight


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

        self.read_txt()

    def read_txt(self):
        self.logger.info('read data...')
        _data_dict_path = os.path.join(self.config.save_path, '_data_dict.%s.%s.pkl' % (
            self.config.turn_type, self.config.turn_num))
        if not os.path.exists(_data_dict_path):
            datas = []
            with open(self.config.pair_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    line = line.rstrip()
                    if not bool(line):
                        continue

                    conversation_id, context, response, hash_value, score, turn = line.split('\t')

                    if not bool(context) or not bool(response):
                        continue

                    response_ids = self.vocab.words_to_id(response.split(' '))
                    if len(response_ids) < self.config.min_len or len(response_ids) > self.config.r_max_len + 50:
                        continue

                    #  response_ids = response_ids[-min(self.r_max_len - 1, len(response_ids)):]
                    response_ids = response_ids[:min(self.config.r_max_len - 1, len(response_ids))]

                    # context split by EOS, START
                    if context.startswith('start eos'):
                        context = context[10:]
                        sentences = self.parser_conversations(context)
                        if len(sentences) > 2:
                            sentences = sentences[1:]
                    elif context.startswith('eos'):
                        context = context[4:]
                        sentences = self.parser_conversations(context)
                    elif context.startswith('... eos'):
                        context = context[7:]
                        sentences = self.parser_conversations(context)
                    elif context.startswith('... '):
                        context = context[4:]
                        sentences = self.parser_conversations(context)
                    else:
                        sentences = self.parser_conversations(context)

                    if sentences is None or len(sentences) < self.config.min_turn:
                        continue

                    sentences = sentences[-min(self.config.turn_num,
                                               len(sentences)):]

                    if self.config.turn_type == 'concat':
                        context = ' '.join(sentences)
                        sentences = [context]
                    elif self.config.turn_type == 'none':
                        context = sentences[-1]
                        sentences = [context]

                    sentences_text = []

                    sentences_ids = []
                    for sentence in sentences:
                        sentence_ids = self.vocab.words_to_id(
                            sentence.split(' '))
                        sentence_ids = sentence_ids[-min(
                            self.config.c_max_len, len(sentence_ids)):]
                        sentences_ids.append(sentence_ids)
                        sentences_text.append(
                            ' '.join(self.vocab.ids_to_word(sentence_ids)))

                    datas.append((conversation_id, sentences_text,
                                  sentences_ids, response_ids, hash_value))

            np.random.shuffle(datas)
            # train-eval split
            n_train = int(
                len(datas) * (1. - self.config.eval_split - self.config.test_split))
            n_eval = max(int(len(datas) * self.config.eval_split),
                         self.config.batch_size)
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

    def parser_conversations(self, context):
        sentences = context.split('eos')
        sentences = [sentence for sentence in sentences if len(
            sentence.split()) >= self.config.min_len]
        return sentences

    def reset_data(self, task, shuffle=True):
        if shuffle:
            np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def load_data(self, task, batch_size):
        task_len = len(self._data_dict[task])
        if batch_size > task_len:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > self._size_dict[task]:
            self.reset_data(task)
            cur_indicator = batch_size

        h_inputs = list()
        h_inputs_position = list()
        h_inputs_lenght = list()
        h_turns_length = list()

        decoder_inputs = torch.zeros((self.config.r_max_len, batch_size),
                                     dtype=torch.long,
                                     device=self.device)

        decoder_targets = torch.zeros((self.config.r_max_len, batch_size),
                                      dtype=torch.long,
                                      device=self.device)

        decoder_inputs_length = list()

        context_texts = list()
        response_texts = list()

        # facts
        facts_texts = list()

        f_inputs = list()
        f_inputs_length = list()
        f_topks_length = list()

        batch_data = self._data_dict[task][self._indicator_dict[task]: cur_indicator]
        """sort batch_data, by turn num"""
        batch_data = sorted(batch_data, key=lambda item: len(item[2]), reverse=True)
        for i, (conversation_id, sentences_text, sentences_ids, response_ids, hash_value) in enumerate(batch_data):

            response_text = ' '.join(self.vocab.ids_to_word(response_ids))
            response_texts.append(response_text)
            context_texts.append(sentences_text)

            # h inputs
            h_inputs_lenght.append(list([1]) * self.config.turn_num)
            h_turns_length.append(len(sentences_ids))

            h_input = torch.zeros((self.config.turn_num, self.config.c_max_len), dtype=torch.long).to(
                self.device)  # [turn_num, max_len]
            if self.config.turn_type == 'transformer':
                h_position = torch.zeros((self.config.turn_num, self.config.c_max_len), dtype=torch.long).to(
                    self.device)  # [turn_nu, max_len]

            for j, ids in enumerate(sentences_ids):
                h_inputs_lenght[i][j] = len(ids)

                tmp_i = torch.zeros(self.config.c_max_len,
                                    dtype=torch.long, device=self.device)
                if self.config.turn_type == 'transformer':
                    tmp_p = torch.zeros(
                        self.config.c_max_len, dtype=torch.long, device=self.device)

                for k, id in enumerate(ids):
                    tmp_i[k] = id
                    if self.config.turn_type == 'transformer':
                        tmp_p[k] = k + 1

                h_input[j, :] = tmp_i
                if self.config.turn_type == 'transformer':
                    h_position[j, :] = tmp_p

            h_inputs.append(h_input)
            if self.config.turn_type == 'transformer':
                h_inputs_position.append(h_position)

            # decoder_inputs
            decoder_inputs[0, i] = self.vocab.sosid
            for r, token_id in enumerate(response_ids):
                decoder_inputs[r + 1, i] = token_id
                decoder_targets[r, i] = token_id
            decoder_targets[len(response_ids), i] = self.vocab.eosid
            decoder_inputs_length.append(len(response_ids) + 1)

            if self.config.model_type == 'kg':
                topk_facts_text = self.facts_topk_phrases.get(hash_value, None)
                f_input = torch.zeros((self.config.f_topk, self.config.f_max_len),
                                      dtype=torch.long,
                                      device=self.device)

                f_input_length = torch.ones(self.config.f_topk,
                                            dtype=torch.long,
                                            device=self.device)

                if topk_facts_text is not None:
                    topk_facts_ids = [self.vocab.words_to_id(
                        text.split(' ')) for text in topk_facts_text]
                    f_topks_length.append(
                        min(len(topk_facts_ids), self.config.f_topk))
                    for fi, ids in enumerate(topk_facts_ids[:self.config.f_topk]):
                        ids = ids[:min(self.config.f_max_len, len(ids))]
                        f_input_length[fi] = len(ids)
                        for fj, id in enumerate(ids):
                            f_input[fi, fj] = id
                else:
                    f_topks_length.append(1)

                f_inputs.append(f_input)
                f_inputs_length.append(f_input_length)

                facts_texts.append(topk_facts_text)

        # [turn_num, max_len, batch_size]
        h_inputs = torch.stack(h_inputs, dim=2)
        if self.config.turn_type == 'transformer':
            # [turn_num, max_len, batch_size]
            h_inputs_position = torch.stack(h_inputs_position, dim=2)

        h_turns_length = torch.tensor(
            h_turns_length, dtype=torch.long, device=self.device) # [batch_size]
        h_inputs_lenght = torch.tensor(
            h_inputs_lenght, dtype=torch.long, device=self.device).transpose(0, 1)  # [turn_num, batch_size]

        decoder_inputs_length = torch.tensor(
            decoder_inputs_length, dtype=torch.long, device=self.device)  # [batch_size]

        if self.config.model_type == 'kg':
            # [topk, max_len, batch_size]
            f_inputs = torch.stack(f_inputs, dim=2)
            f_inputs_length = torch.stack(
                f_inputs_length, dim=1)  # [f_topk, batch_size]

            f_topks_length = torch.tensor(
                f_topks_length, dtype=torch.long, device=self.device)

        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator

        return decoder_inputs, decoder_targets, decoder_inputs_length, \
            context_texts, response_texts, \
            f_inputs, f_inputs_length, f_topks_length, facts_texts, \
            h_inputs, h_turns_length, h_inputs_lenght, h_inputs_position

    def load_similarity_facts(self, offline_filename):
        self.facts_topk_phrases = pickle.load(open(offline_filename, 'rb'))

    def retrieval_similarity_facts(self, offline_filename):
        es = es_helper.get_connection()
        r = Rake()
        facts_topk_phrases = {}
        for task, task_datas in self._data_dict.items():
            self.logger.info('retrieval similarity facts for: %s data.' % task)
            for data in tqdm(task_datas):
                conversation_id, sentences_text, _, _, hash_value = data
                sentences_text = [' '.join(remove_stop_words(sentence.split())) for sentence in sentences_text]
                if len(sentences_text) >= 2 or len(sentences_text[0]) > self.config.f_max_len + 20:
                    r.extract_keywords_from_sentences(sentences_text)
                    texts = r.get_ranked_phrases()[:self.config.f_topk]
                    query_text = ' '.join(texts)
                else:
                    query_text = ' '.join(sentences_text)

                query_body = es_helper.assemble_search_fact_body(conversation_id, query_text)
                hits, hit_count = es_helper.search(es, es_helper.fact_type, query_body)
                if hit_count == 0:
                    continue
                phrases = []
                for hit in hits[:self.config.topk]:
                    phrase = hit['text']
                    phrases.append(phrase)
                r.extract_keywords_from_sentences(phrases)
                topk_phrase = r.get_ranked_phrases()[:self.config.f_topk]

                facts_topk_phrases[hash_value] = topk_phrase

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
        if offline_type == 'elastic':
            self.retrieval_similarity_facts(offline_filename)

        ranked_phrase_dict_path = self.config.save_path + 'ranked_phrase_dict.%s.pkl' % offline_type
        ranked_phrase_embedded_dict_path = self.config.save_path + 'ranked_phrase_embedded_dict.%s.pkl' % offline_type
        try:
            self.logger.info('load ranked phrase...')
            ranked_phrase_dict = pickle.load(open(ranked_phrase_dict_path, 'rb'))
            ranked_phrase_embedded_dict = pickle.load(open(ranked_phrase_embedded_dict_path, 'rb'))
        except FileNotFoundError as e:
            r = Rake(
                min_length=1,
                max_length=self.config.f_max_len + 15
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
                        mean_embedded = self.get_sentence_embedded(ids, embedding, offline_type)
                    elif offline_type == 'elmo':
                        mean_embedded = self.get_sentence_embedded(phrase.split(), embedding, offline_type)

                    phrase_embeddeds.append(mean_embedded)
                #  phrase_embeddeds = torch.stack(phrase_embeddeds, dim=0) # [len(phrase), pre_embedding_size]
                ranked_phrase_embedded_dict[conversation_id] = phrase_embeddeds

            pickle.dump(ranked_phrase_dict, open(ranked_phrase_dict_path, 'wb'))
            pickle.dump(ranked_phrase_embedded_dict, open(ranked_phrase_embedded_dict_path, 'wb'))

        # embedding match
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        facts_topk_phrases = {}
        for task, task_datas in self._data_dict.items():
            self.logger.info('build similarity facts for: %s data.' % task)
            for data in tqdm(task_datas):
                conversation_id, sentences_text, _, _, hash_value = data
                # [len(phrases), pre_embedding_size]
                phrase_embeddeds = ranked_phrase_embedded_dict.get(conversation_id, None)
                if phrase_embeddeds is None or len(phrase_embeddeds) == 0:
                    continue
                phrases = ranked_phrase_dict.get(conversation_id, None)
                if phrases is None or len(phrases) == 0:
                    continue

                sum_scores = list()
                for sentence in sentences_text:
                    sentence_words = remove_stop_words(sentence.split())
                    sentence_ids = self.vocab.words_to_id(sentence_words)
                    mean_embedded = self.get_sentence_embedded(sentence_ids, embedding, offline_type)
                    scores = list()
                    for phrase_embedded in phrase_embeddeds:
                        #  print(phrase_embedded.shape)
                        #  print(mean_embedded.shape)
                        scores.append(cos(mean_embedded.view(-1), phrase_embedded.view(-1)))
                    scores = torch.stack(scores, dim=0)
                    sum_scores.append(scores)
                # [len(sentences), len(phrase_embeddeds)]
                sum_scores = torch.stack(sum_scores)
                #  print('sum_scores: ', sum_scores.shape)
                # [len(phrase_embeddeds)]
                sum_score = sum_scores.sum(dim=0)
                _, indexes = sum_score.topk(min(self.config.f_topk, sum_score.size(0)), dim=0)

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
            tokens = ids # if offline_type is elmo, ids = words
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
                             context_texts,
                             response_texts,
                             greedy_texts,
                             beam_texts,
                             filename,
                             decode_type='greed',
                             facts_texts=None):

        #  print(facts_texts)
        with open(filename, 'a', encoding='utf-8') as f:
            for i, (sentences, response, greedy_text, beam_text) in enumerate(zip(context_texts, response_texts, greedy_texts, beam_texts)):
                for sentence in sentences:
                    f.write('> %s\n' % sentence)

                #  f.write('> %s\n' % sentences)
                f.write('gold: %s\n' % response)

                f.write('greedy: %s\n' % greedy_text)

                for i, best_text in enumerate(beam_text):
                    f.write('beam %d: %s\n' % (i, best_text))

                if facts_texts is not None and len(facts_texts) > 0:
                    topk_facts = facts_texts[i]
                    if topk_facts is not None:
                        for fi, fact_text in enumerate(topk_facts):
                            f.write('fact %d: %s\n' % (fi, fact_text))
                f.write('---------------------------------\n')

    def generating_texts(self, outputs, outputs_length=None, decode_type='greedy'):
        """
        decode_type == greedy:
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
                topk_length = outputs_length[bi]
                topk_texts = []
                for ids, length in zip(topk_ids, topk_length):
                    text = self.vocab.ids_to_text(ids[:length])
                    topk_texts.append(text)
                batch_generated_texts.append(topk_texts)

        return batch_generated_texts
