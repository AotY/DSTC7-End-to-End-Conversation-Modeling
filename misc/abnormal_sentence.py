#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Abnormal sentences.
1. Too much punctuation
2. word which occurs contiguously.
3. to long
"""
import os
import re
import string
import argparse
from tqdm import tqdm
from nltk import ngrams

parser = argparse.ArgumentParser()

parser.add_argument('--ngram', type=int, help='')
parser.add_argument('--convos_path', type=str, help='')
parser.add_argument('--save_dir', type=str, help='')

args = parser.parse_args()

punc_regex = re.compile('[%s]' % re.escape(string.punctuation.replace('_', '')))

def main():
    abnormal_cs = []
    abnormal_qs = []
    abnormal_rs = []

    n = args.ngram
    with open(args.convos_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            subreddit_name, conversation_id, context, query, \
                response, hash_value, score, turn = line.split(' SPLIT ')


            query = punc_regex.sub('', query)
            q_ngrams = ngrams(query.split(), n)
            for words in q_ngrams:
                if len(set(words)) == 1:
                    abnormal_qs.append(query)
                    break

            response = punc_regex.sub('', response)
            r_ngrams = ngrams(response.split(), n)
            for words in r_ngrams:
                if len(set(words)) == 1:
                    abnormal_rs.append(response)
                    break

            context_sentences = context.split('EOS')
            context_sentences = [sentence for sentence in context_sentences if len(sentence.split()) >= 2]

            if len(context_sentences) < 1:
                continue

            for sentence in context_sentences:
                sentence = punc_regex.sub('', sentence)
                s_ngrams = ngrams(sentence.split(), n)
                for words in s_ngrams:
                    if len(set(words)) == 1:
                        abnormal_cs.append(sentence)
                        break

    print('abnormal c: %d', len(abnormal_cs))
    print('abnormal q: %d', len(abnormal_qs))
    print('abnormal r: %d', len(abnormal_rs))

    with open(os.path.join(args.save_dir, 'abnormal_c.%d.txt' % n), 'w') as f:
        for line in abnormal_cs:
            f.write('%s\n' % line)

    with open(os.path.join(args.save_dir, 'abnormal_q.%d.txt' % n), 'w') as f:
        for line in abnormal_qs:
            f.write('%s\n' % line)

    with open(os.path.join(args.save_dir, 'abnormal_r.%d.txt' % n), 'w') as f:
        for line in abnormal_rs:
            f.write('%s\n' % line)

if __name__ == '__main__':
    main()
