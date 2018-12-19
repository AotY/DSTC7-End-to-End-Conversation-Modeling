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

            if len(context.split()) > 0:
                freq_dict.update(context.split())

            if len(query.split()) > 0:
                freq_dict.update(query.split())

            if len(response.split()) > 0:
                freq_dict.update(response.split())

    with open(args.facts_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            _, _, _, fact = line.split('\t')

            if len(fact.split()) > 0:
                freq_dict.update(fact.split())

    sorted_freq_list = sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)
    with open(args.freq_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))


if __name__ == '__main__':
    main()
