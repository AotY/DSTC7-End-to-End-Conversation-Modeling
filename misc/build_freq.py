#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Build word freq from convos and facts.
"""

import argparse
from tqdm import tqdm
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--convos_path', type=str, default='./data/train.convos.txt')
parser.add_argument('--facts_path', type=str, default='./data/train.facts.txt')
parser.add_argument('--freq_path', type=str, default='./data/word.freq.txt')
args = parser.parse_args()

def main():
    freq_dict = Counter()
    # read pair
    with open(args.convos_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            subreddit_name, conversation_id, context, query, \
                response, hash_value, score, turn = line.split(' SPLIT ')

            if not bool(query) or not bool(response):
                continue

            context = context.replace('EOS', '')
            if len(context.split()) > 0:
                words = [word for word in context.split() if len(word.split()) > 0]
                freq_dict.update(words)

            if len(query.split()) > 0:
                words = [word for word in query.split() if len(word.split()) > 0]
                freq_dict.update(words)

            if len(response.split()) > 0:
                words = [word for word in response.split() if len(word.split()) > 0]
                freq_dict.update(words)

    with open(args.facts_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            _, _, _, fact = line.split('\t')

            if len(fact.split()) > 0:
                words = [word for word in fact.split() if len(word.split()) > 0]
                freq_dict.update(words)

    sorted_freq_list = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)
    with open(args.freq_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))

if __name__ == '__main__':
    main()
