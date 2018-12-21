#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
statistics sentence num distribution.
"""

import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--convos_path', type=str, help='', default='./data/cleaned.convos.txt')
parser.add_argument('--save_path', type=str, help='', default='./data')

args = parser.parse_args()

def stats():
    q_len_dict = {}
    q_sentence_count_dict = {}

    c_len_dict = {}
    c_sentence_count_dict = {}
    c_num_dict = {}

    r_len_dict = {}
    r_sentence_count_dict = {}

    with open(args.convos_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            data_type, subreddit_name, conversation_id, context, query, \
                response, hash_value, score, turn = line.split(' SPLIT ')

            if data_type in ['TEST']:
                continue

            if not bool(query) or not bool(response):
                continue

            context_sentences = context.split('EOS')
            if len(context_sentences) > 0:
                context_sentences = [s for s in context_sentences if len(s.split()) >= 3]
                context = ''.join(context_sentences)
            else:
                context = ''

            # sentences
            q_sentences = query.split('.')
            q_sentences = [sentence for sentence in q_sentences if len(sentence.split()) >= 3]

            c_sentences = context.split('.')
            c_sentences = [sentence for sentence in c_sentences if len(sentence.split()) >= 3]

            r_sentences = response.split('.')
            r_sentences = [sentence for sentence in r_sentences if len(sentence.split()) >= 3]

            # dict
            query_tokens = query.split()
            q_len_dict[len(query_tokens)] = q_len_dict.get(len(query_tokens), 0) + 1
            q_sentence_count_dict[len(q_sentences)] = q_sentence_count_dict.get(len(q_sentences), 0) + 1

            response_tokens = response.split()
            r_len_dict[len(response_tokens)] = r_len_dict.get(len(response_tokens), 0) + 1
            r_sentence_count_dict[len(r_sentences)] = r_sentence_count_dict.get(len(r_sentences), 0) + 1

            for sentence in context_sentences:
                tokens = sentence.split()
                c_len_dict[len(tokens)] = c_len_dict.get(len(tokens), 0) + 1

            c_sentence_count_dict[len(c_sentences)] = c_sentence_count_dict.get(len(c_sentences), 0) + 1
            c_num_dict[len(context_sentences)] = c_num_dict.get(len(context_sentences), 0) + 1

        save_distribution(q_len_dict, 'q_len')
        save_distribution(q_sentence_count_dict, 'q_sentence_count')

        save_distribution(r_len_dict, 'r_len')
        save_distribution(r_sentence_count_dict, 'r_sentence_count')

        save_distribution(c_len_dict, 'c_len')
        save_distribution(c_sentence_count_dict, 'c_sentence_count')
        save_distribution(c_num_dict, 'c_num')

def save_distribution(distribution, name):
    distribution_list = sorted(distribution.items(), key=lambda item: item[0], reverse=False)
    with open(os.path.join(args.save_path, name + '.dist.txt'), 'w', encoding="utf-8") as f:
        for i, j in distribution_list:
            f.write('%s\t%s\n' % (str(i), str(j)))


if __name__ == '__main__':
    stats()
