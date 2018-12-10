#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.


import re
import argparse
from tqdm import tqdm
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--pair_path', type=str, default='../data/train.pseudo_convos.txt')
parser.add_argument('--pair_save_path', type=str, default='../data/train.convos.txt')

parser.add_argument('--fact_path', type=str, default='../data/train.pseudo_facts.txt')
parser.add_argument('--fact_save_path', type=str, default='../data/train.facts.txt')

parser.add_argument('--vocab_freq_path', type=str, default='./vocab_freq.txt')
args = parser.parse_args()


def clean_number_url(text):
    text = text.replace('( __number__ )', '__number__')
    text = text.replace('( __url__ )', '__url__')

    text = text.replace('__url__ __url__', '__url__')
    text = text.replace('__number__ __number__', '__number__')
    text = text.replace('__number__ __url__', '__number__')
    text = text.replace('__url__ __number__', '__url__')

    text = re.sub(r'__number__\w+', '__number__', text)
    text = re.sub(r'\w+__number__', '__number__', text)
    text = re.sub(r'__url__\w+', '__url__', text)
    text = re.sub(r'\w+__url__', '__url__', text)

    text = re.sub(r'__number__ \S __number__', '__number__', text)
    text = re.sub(r'__url__ \S __number__', '__url__', text)
    text = re.sub(r'__number__ \S __url__', '__number__', text)

    text = text.replace('__number __', ' __number__ ')
    text = text.replace('__url __', ' __url__ ')
    return text

def main():
    freq_dict = Counter()
    # read pair
    pair_save_f = open(args.pair_save_path, 'w')
    with open(args.pair_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            subreddit_name, conversation_id, context, query, \
                response, hash_value, score, turn = line.split(' SPLIT ')

            #  print('context: ', context)
            #  print('query: ', query)
            #  print('response: ', response)

            if not bool(query) or not bool(response):
                continue

            if len(context.split()) != 0:
                context = clean_number_url(context)
                freq_dict.update(context.split())

            query = clean_number_url(query)
            freq_dict.update(query.split())

            response = clean_number_url(response)
            freq_dict.update(response.split())

            pair_save_f.write('%s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s\n' % \
                    (subreddit_name, conversation_id, context, query, response, hash_value, score, turn))

    pair_save_f.close()

    fact_save_f = open(args.fact_save_path, 'w')
    with open(args.fact_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue
            subreddit_name, conversation_id, domain, fact = line.split('\t')

            fact = clean_number_url(fact)
            freq_dict.update(fact.split())

            fact_save_f.write('%s\t%s\t%s\t%s\n' % (subreddit_name, conversation_id, domain, fact))
    fact_save_f.close()

    freq_list = list(freq_dict.items())
    sorted_freq_list = sorted(freq_list, key=lambda item: item[1], reverse=True)
    with open(args.vocab_freq_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))


if __name__ == '__main__':
    main()
