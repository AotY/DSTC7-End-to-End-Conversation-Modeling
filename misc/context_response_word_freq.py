#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
stats word frequency of context and response.
"""

from tqdm import tqdm
from collections import Counter


def stats_word_freq(pair_path):
    context_word_dict = Counter()
    response_word_dict = Counter()
    with open(pair_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            _, context, response, _, _, _ = line.split('\t')

            if not bool(context) or not bool(response):
                continue

            if context.startswith('start eos'):
                context = context[10:]
            elif context.startswith('eos'):
                context = context[4:]
            elif context.startswith('... eos'):
                context = context[7:]
            elif context.startswith('... '):
                context = context[4:]

            sentences = context.split('eos')
            context = ' '.join(sentences)

            context_word_dict.update(context.split(' '))
            response_word_dict.update(response.split(' '))

    save_distribution(context_word_dict, 'context_word.freq')
    save_distribution(response_word_dict, 'response_word.freq')


def save_distribution(distribution, name):
    distribution_list = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    with open(name + '.distribution.txt', 'w', encoding="utf-8") as f:
        for i, j in distribution_list:
            f.write('%s\t%s\n' % (str(i), str(j)))


if __name__ == '__main__':
    pair_path = './../data/conversations_responses.pair.txt'
    stats_word_freq(pair_path)
