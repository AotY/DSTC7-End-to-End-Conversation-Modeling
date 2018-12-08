#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.


import argparse
from tqdm import tqdm
from utils import tokenizer

parser = argparse.ArgumentParser()

parser.add_argument('--pair_path', type=str, default='../data/train.pseudo_pair.txt')
parser.add_argument('--pair_save_path', type=str, default='../data/train.pair.txt')

parser.add_argument('--fact_path', type=str, default='../data/train.pseudo_facts.txt')
parser.add_argument('--fact_save_path', type=str, default='../data/train.facts.txt')

parser.add_argument('--vocab_freq_path', type=str, default='./vocab_freq.txt')
args = parser.parse_args()



def main():
    tokenizer = Tokenizer()
    freq_dict = Counter()
    # read pair
    pair_save_f = open(args.pair_save_path, 'w')
    with open(self.config.pair_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            subreddit_name, conversation_id, context, query, \
                response, hash_value, score, turn = line.split(' SPLIT ')

            if not bool(query) or not bool(response):
                continue
            
            if context != '':
                context_tokens = tokenizer.tokenize(context)
                freq_dict.update(context_tokens)
                context = ' '.join(context)

            query_tokens = tokenizer.tokenize(query)
            freq_dict.update(query_tokens)
            query = ' '.join(query_tokens)

            response_tokens = tokenizer.tokenize(response)
            freq_dict.update(response_tokens)
            response = ' '.join(response_tokens)

            pair_save_f.write('%s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s\n' % \
                    (subreddit_name, conversation_id, context, query, response, hash_value, score, turn))

    pair_save_f.close()
    
    fact_save_f = open(args.fact_save_path, 'w')
    with open(self.config.fact_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue
            subreddit_name, conversation_id, domain, fact = line.split('\t')

            fact_save_f.write('%s\t%s\t%s\t%s\n' %
                    (subreddit, conversation_id, domain, fact))
    fact_save_f.close()

    freq_dict = list(freq_dict.items())
    sorted_freq_list = sorted(freq_dict.items(), key=lambda d: d[1], reverse=True)
    with open(args.vocab_freq_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))



if __name__ == '__main__':
    main()
