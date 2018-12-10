#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
statistics sentence num distribution.
"""

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--convos_path', type=str, help='')

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
            subreddit_name, conversation_id, context, query, \
                response, hash_value, score, turn = line.split(' SPLIT ')

            sentences = context.split('EOS')
            sentences = [s for s in sentences if len(s.split()) >= 3]

            if len(sentences) < 1:
                continue

            context_texts = sentences[:-1]
            context = ''.join(context_texts)

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

            for text in context_texts:
                tokens = text.split()
                c_len_dict[len(tokens)] = c_len_dict.get(len(tokens), 0) + 1

            c_sentence_count_dict[len(c_sentences)] = c_sentence_count_dict.get(len(c_sentences), 0) + 1
            c_num_dict[len(context_texts)] = c_num_dict.get(len(context_texts), 0) + 1

        save_distribution(q_len_dict, 'q_len')
        save_distribution(q_sentence_count_dict, 'q_sentence_count')

        save_distribution(r_len_dict, 'r_len')
        save_distribution(r_sentence_count_dict, 'r_sentence_count')

        save_distribution(c_len_dict, 'c_len')
        save_distribution(c_sentence_count_dict, 'c_sentence_count')
        save_distribution(c_num_dict, 'c_num')

def save_distribution(distribution, name):
    distribution_list = sorted(distribution.items(), key=lambda item: item[0], reverse=False)
    with open(name + '.distribution.txt', 'w', encoding="utf-8") as f:
        for i, j in distribution_list:
            f.write('%s\t%s\n' % (str(i), str(j)))


if __name__ == '__main__':
    stats()
