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

from misc.vocab import PAD_ID, SOS_ID, EOS_ID, UNK_ID


class Dataset:
    """
    Read data, and
    Load data
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
        """split to train, dev, valid, test"""
        self.logger.info('read data...')
        _data_dict_path = os.path.join(self.config.save_path, '_data_dict.3.%s_%s.pkl' % (self.config.c_min, self.config.c_max))
        if not os.path.exists(_data_dict_path):
            min_len = self.config.min_len
            c_max_len, q_max_len, r_max_len = self.config.c_max_len, self.config.q_max_len, self.config.r_max_len
            with open(self.config.convos_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    line = line.rstrip()
                    if not bool(line):
                        continue

                    data_type, subreddit_name, conversation_id, context, query, \
                        response, hash_value, score, turn = line.split(' SPLIT ')

                    if not bool(query) or not bool(response):
                        continue

                    # query
                    q_words = [word for word in query.split() if len(word.split()) > 0]
                    q_ids = self.vocab.words_to_id(q_words)
                    if len(q_ids) < min_len:
                        continue

                    # response
                    if data_type == 'TEST':
                        r_ids = [UNK_ID]
                    else:
                        r_words = [word for word in response.split() if len(word.split()) > 0]
                        r_ids = self.vocab.words_to_id(r_words)
                        if len(r_ids) < min_len:
                            continue

                    # context split by EOS
                    c_sentences = self.parser_context(context)
                    if c_sentences is None or len(c_sentences) < self.config.c_min:
                        continue

                    c_sentences = c_sentences[-min(self.config.c_max, len(c_sentences)):]

                    c_ids = []
                    for si, sentence in enumerate(c_sentences):
                        words = [word for word in sentence.split() if len(word.split()) > 0]
                        ids = self.vocab.words_to_id(words)
                        c_ids.append(ids)

                    if self._data_dict.get(data_type) is None:
                        self._data_dict[data_type] = list()

                    self._data_dict[data_type].append((hash_value, subreddit_name, conversation_id, c_ids, q_ids, r_ids))

            pickle.dump(self._data_dict, open(_data_dict_path, 'wb'))
        else:
            self._data_dict = pickle.load(open(_data_dict_path, 'rb'))

        for key in self._data_dict.keys():
            self._indicator_dict[key] = 0
            self.logger.info('%s: %d' % (key, len(self._data_dict[key])))

    def parser_context(self, context):
        sentences = context.split(' EOS ')
        context_sentences = list()
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < self.config.min_len:
                if i != len(sentences) - 1:
                    context_sentences.clear()
                continue
            context_sentences.append(sentence)
        return context_sentences

    def reset_data(self, task, shuffle=True):
        if shuffle:
            np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def load_data(self, task):
        enc_type = self.config.enc_type
        model_type = self.config.model_type
        turn_num = self.config.turn_num
        batch_size = self.config.batch_size
        c_max_len, q_max_len, r_max_len, f_max_len = self.config.c_max_len, self.config.q_max_len, \
            self.config.r_max_len, self.config.f_max_len

        if batch_size > len(self._data_dict[task]):
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > len(self._data_dict[task]):
            if task in ['TRAIN', 'DEV', 'VALID', 'TEST']:
                self.reset_data(task, True)
                cur_indicator = batch_size

        batch_data = self._data_dict[task][self._indicator_dict[task]: cur_indicator]

        padded_batch_data = list()
        for hash_value, subreddit_name, conversation_id, c_ids, q_ids, r_ids in batch_data:
            enc_len = None
            turn_len = None
            enc_ids = None
            if enc_type in ['q', 'q_attn']:
                enc_ids = q_ids[-min(q_max_len, len(q_ids)):]
                enc_len = len(enc_ids)
                enc_ids = enc_ids + [PAD_ID] * (q_max_len - enc_len)
            elif enc_type  in ['qc', 'qc_attn']:
                enc_ids = list()
                for ids in c_ids:
                    enc_ids.extend(ids)
                enc_ids.extend(q_ids)
                enc_ids = enc_ids[-min(q_max_len, len(enc_ids)):]
                enc_len = len(enc_ids)
                enc_ids = enc_ids + [PAD_ID] * (q_max_len - enc_len)
            else:
                enc_ids = list()
                for ids in c_ids:
                    enc_ids.append(ids)
                enc_ids.append(q_ids)

                turn_len = len(enc_ids)
                enc_len = [1] * turn_num

                padded_enc_ids = list()
                for i, ids in enumerate(enc_ids):
                    ids = ids[-min(q_max_len, len(ids)):]
                    enc_len[i] = len(ids)
                    ids = ids + [PAD_ID] * (q_max_len - len(ids))
                    padded_enc_ids.append(ids)

                if len(padded_enc_ids) < turn_num:
                    for _ in range(turn_num - len(padded_enc_ids)):
                        padded_enc_ids.append([EOS_ID] + [PAD_ID] * (q_max_len - 1))

                enc_ids = padded_enc_ids

            r_ids = r_ids[:min(r_max_len, len(r_ids))]
            dec_len = len(r_ids)

            #  r_ids = r_ids[-min(r_max_len, len(r_ids)):]
            dec_ids = [SOS_ID] + r_ids + [EOS_ID] + [PAD_ID] * (r_max_len - len(r_ids))

            f_ids = None
            f_len = None
            if model_type == 'kg':
                words = self.facts_topk_phrases.get(hash_value, None)
                if words is None:
                    f_len = 1
                    #  f_ids = [EOS_ID]  + [PAD_ID] * (f_max_len - 1)
                    f_ids = [PAD_ID] * f_max_len
                else:
                    f_ids = self.vocab.words_to_id(words)
                    f_ids = f_ids[:min(f_max_len, len(f_ids))]
                    f_len = len(f_ids)

                    f_ids = f_ids + [PAD_ID] * (f_max_len - len(f_ids))

            padded_batch_data.append((hash_value, subreddit_name, conversation_id, \
                                      enc_ids, enc_len, turn_len, dec_ids, dec_len, \
                                      f_ids, f_len))

        # To Tensor
        enc_inputs = list()
        enc_inputs_length = list()
        enc_turn_length = list()

        dec_inputs = list()
        dec_inputs_length = list()

        subreddit_names = list()
        conversation_ids = list()
        hash_values = list()

        f_inputs = list()
        f_inputs_length = list()
        f_topk_length = list()

        """sort batch_data"""
        if enc_type in ['q', 'q_attn', 'qc', 'qc_attn']:
            sorted_batch_data = sorted(padded_batch_data, key=lambda item: item[4], reverse=True)
        else:
            sorted_batch_data = sorted(padded_batch_data, key=lambda item: item[5], reverse=True)

        for hash_value, subreddit_name, conversation_id, \
                enc_ids, enc_len, turn_len, dec_ids, dec_len, \
                f_ids, f_len in sorted_batch_data:

            hash_values.append(hash_value)
            subreddit_names.append(subreddit_name)
            conversation_ids.append(conversation_id)

            enc_inputs.append(enc_ids)
            enc_inputs_length.append(enc_len)
            enc_turn_length.append(turn_len)

            dec_inputs.append(dec_ids)

            if model_type == 'kg':
                f_inputs.append(f_ids)
                f_inputs_length.append(f_len)

        if enc_type in ['q', 'q_attn', 'qc', 'qc_attn']:
            # [max_len, batch_size]
            enc_inputs = torch.tensor(enc_inputs, dtype=torch.long, device=self.device).transpose(0, 1)
            enc_inputs_length = torch.tensor(enc_inputs_length, dtype=torch.long, device=self.device)
        else:
            # [turn_num, max_len, batch_size]
            enc_inputs = torch.tensor(enc_inputs, dtype=torch.long, device=self.device).permute(1, 2, 0)

            # [turn_num, batch_size]
            enc_inputs_length = torch.tensor(
                enc_inputs_length, dtype=torch.long, device=self.device).transpose(0, 1)

            # [batch_size]
            enc_turn_length = torch.tensor(
                enc_turn_length, dtype=torch.long, device=self.device)  # [batch_size]

        # decoder [max_len, batch_size]
        dec_inputs = torch.tensor(dec_inputs, dtype=torch.long, device=self.device).transpose(0, 1)

        if self.config.model_type == 'kg':
            # [batch_size, f_topk]
            f_inputs = torch.tensor(f_inputs, dtype=torch.long, device=self.device)
            # [batch_size]
            f_inputs_length = torch.tensor(f_inputs_length, dtype=torch.long, device=self.device)

        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator

        return dec_inputs, enc_inputs, \
            enc_inputs_length, enc_turn_length, \
            f_inputs, f_inputs_length, f_topk_length, \
            subreddit_names, conversation_ids, hash_values

    def load_similarity_facts(self, offline_filename):
        self.facts_topk_phrases = pickle.load(open(offline_filename, 'rb'))

    def retrieval_similarity_facts(self, offline_filename, facts_tfidf_dict):
        es = es_helper.get_connection()
        f_max_len = self.config.f_max_len
        turn_num = self.config.turn_num
        #  rake = Rake(
            #  min_length=1,
            #  max_length=self.config.f_max_len
        #  )
        facts_topk_phrases = {}
        for task, task_datas in self._data_dict.items():
            self.logger.info('retrieval similarity facts for: %s data.' % task)
            for data in tqdm(task_datas):
                hash_value, _, conversation_id, context_ids, query_ids, _ = data
                query = self.vocab.ids_to_text(query_ids)
                query = re.sub(r'__number__|__url__|__unk__', '', query)
                context_sentences = []
                for ids in context_ids:
                    sentence = self.vocab.ids_to_text(ids)
                    sentence = re.sub(r'__number__|__url__|__unk__', '', sentence)
                    context_sentences.append(sentence)

                query_text = query + ' ' + ' '.join(context_sentences)
                query_text = ' '.join(remove_stop_words(query_text.split()))

                #  print('query_text: ', query_text)
                query_body = es_helper.assemble_search_fact_body(conversation_id, query_text)
                _, total = es_helper.search(es, es_helper.index, es_helper.fact_type, query_body, size=0)

                words = set()
                if total == 0:
                    #  f_max_len = int(np.ceil(f_max_len / turn_num * (len(context_sentences) + 1)))
                    #  for word in query_text.split():
                        #  words.add(word)
                    continue

                else:
                    hits, _ = es_helper.search(es, es_helper.index, es_helper.fact_type, query_body, size=total)

                    for hit in hits[:self.config.f_topk]:
                        hit_conversation_id = hit['_source']['conversation_id']
                        assert (conversation_id == hit_conversation_id), "%s != %s" % (
                            conversation_id, hit_conversation_id)
                        text = hit['_source']['text']

                        for word in text.split():
                            words.add(word)

                    if total <= self.config.min_len:
                        for word in query_text.split():
                            words.add(word)

                words_tfidf = []
                for word in words:
                    try:
                        value = facts_tfidf_dict[conversation_id].get(word, 0.0)
                    except Exception as e:
                        value = 0.0
                        print('conversation_id: %s' % conversation_id)
                    words_tfidf.append((word, value))

                words_tfidf = sorted(words_tfidf, key=lambda item: item[1], reverse=True)
                words = [item[0] for item in words_tfidf[:f_max_len]]
                #  print('words: ', words)

                facts_topk_phrases[hash_value] = words

        pickle.dump(facts_topk_phrases, open(offline_filename, 'wb'))
        self.facts_topk_phrases = facts_topk_phrases

    def build_similarity_facts_offline(self,
                                       facts_dict=None,
                                       offline_filename=None,
                                       facts_tfidf_dict=None,
                                       embedding=None):
        if os.path.exists(offline_filename):
            self.load_similarity_facts(offline_filename)
            return

        offline_type = self.config.offline_type
        if offline_type in ['elastic', 'elastic_tag']:
            self.retrieval_similarity_facts(offline_filename, facts_tfidf_dict)
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
            ids = torch.tensor(ids, dtype=torch.long, device=self.device)
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

    def save_generated_texts(self,
                             epoch,
                             subreddit_names,
                             conversation_ids,
                             hash_values,
                             enc_inputs,
                             f_inputs,
                             dec_inputs,
                             greedy_texts,
                             beam_texts,
                             save_path,
                             time_str):

        save_mode = 'w'
        if os.path.exists(save_path):
            save_mode = 'a'

        dec_inputs = dec_inputs.transpose(0, 1).tolist()  # [batch_size, max_len]
        #  print('dec_inputs: ', dec_inputs)

        if self.config.enc_type == 'q' or \
            self.config.enc_type == 'qc':
            # [max_len, batch_size] -> [batch_size, max_len]
            enc_inputs = enc_inputs.transpose(0, 1).tolist()
        else:
            # [turn_num, max_len, batch_size] -> [batch_size, turn_num, max_len]
            enc_inputs = enc_inputs.permute(2, 0, 1).tolist()

        if f_inputs is None or len(f_inputs) == 0:
            f_inputs = [None] * self.config.batch_size
        else:
            # [batch_size, topk, max_len]
            #  f_inputs = f_inputs.transpose(0, 1).tolist()

            # [batch_size, f_topk]
            f_inputs = f_inputs.tolist()

        predicted_path = os.path.join(self.config.save_path, 'predicted/%s_%s_%s_%s_%s_%s.txt' % (
            self.config.model_type, self.config.enc_type, epoch, \
            self.config.c_min, self.config.c_max, time_str
        ))
        predicted_f = open(predicted_path, save_mode)

        submission_path = os.path.join(self.config.save_path, 'submission/%s_%s_%s_%s_%s_%s.txt' % (
            self.config.model_type, self.config.enc_type, epoch, \
            self.config.c_min, self.config.c_max, time_str
        ))
        submission_f = open(submission_path, save_mode)

        with open(save_path, save_mode, encoding='utf-8') as f:
            for subreddit_name, conversation_id, hash_value, \
                enc_ids, f_ids, dec_ids, g_text, b_text in \
                    zip(subreddit_names,
                        conversation_ids,
                        hash_values,
                        enc_inputs,
                        f_inputs,
                        dec_inputs,
                        greedy_texts,
                        beam_texts):

                f.write('subreddit_name: %s\n' % subreddit_name)
                f.write('conversation_id: %s\n' % conversation_id)
                f.write('hash_value: %s\n' % hash_value)

                if self.config.enc_type == 'q' or \
                    self.config.enc_type == 'qc':
                    text = self.vocab.ids_to_text(enc_ids)
                    f.write('> %s\n' % (text))
                else:
                    for ids in enc_ids:
                        text = self.vocab.ids_to_text(ids)
                        f.write('> %s\n' % (text))

                #  print('dec_ids: ', dec_ids)
                #  response_text = self.vocab.ids_to_text(dec_ids)
                #  f.write('gold: %s\n' % response_text)
                f.write('\n')

                f.write('greedy: %s\n' % g_text)

                # for embedding metrics
                predicted_f.write('%s\n' % g_text)

                # submission
                submission_f.write('%s\t-\t-\t-\t-\t-\t%s\n' % (hash_value, g_text))

                for i, text in enumerate(b_text):
                    f.write('beam %d: %s\n' % (i, text))

                if f_ids is not None:
                    f_text = self.vocab.ids_to_text(f_ids)
                    f.write('fact : %s\n' % f_text)

                f.write('-' * 70 + '\n')

        predicted_f.close()
        submission_f.close()

    def save_ground_truth(self, task):
        enc_type = self.config.enc_type
        ground_truth_path = os.path.join(self.config.save_path, 'ground_truth/%s.3.ground_truth.txt' % (
            task
        ))

        if os.path.exists(ground_truth_path):
            return
        ground_truth_f = open(ground_truth_path, 'w')

        test_hash_values = list()
        for hash_value, _, _, _, _, _ in self._data_dict[task]:
            test_hash_values.append(hash_value)

        refs_response_dict = {}
        for hash_value, _, _, _, _, r_ids in self._data_dict['REFS']:
            response_text = self.vocab.ids_to_text(r_ids)
            refs_response_dict[hash_value] = response_text

        for hash_value in test_hash_values:
            response_text = refs_response_dict.get(hash_value, '')
            ground_truth_f.write('%s\n' % response_text)

        ground_truth_f.close()
