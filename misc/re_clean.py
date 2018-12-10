#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.


import re
import string
import argparse
from tqdm import tqdm
from nltk import ngrams
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument('--pseudo_convos_path', type=str, default='../data/pseudo.convos.txt')
parser.add_argument('--train_convos_path', type=str, default='../data/train.convos.txt')

parser.add_argument('--pseudo_facts_path', type=str, default='../data/pseudo.facts.txt')
parser.add_argument('--train_facts_path', type=str, default='../data/train.facts.txt')

parser.add_argument('--vocab_freq_path', type=str, default='./vocab_freq.txt')
args = parser.parse_args()

punc_regex = re.compile('[%s]' % re.escape(string.punctuation.replace('_', '')))

def clean_number_url(text):
    text = text.replace('( )', '')
    text = text.replace('[ ]', '')
    text = text.replace('{ }', '')
    text = text.replace("' '", '')
    text = text.replace('" "', '')
    text = text.replace('. .', '.')
    text = text.replace(', ,', ',')
    text = text.replace(': :', ':')
    text = text.replace('; ;', ';')

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

def clean_repeat(text, max_ngram=5):
    tmp_text = punc_regex.sub('', text)
    text_ngrams = ngrams(tmp_text.split(), max_ngram)
    for words in text_ngrams:
        if len(set(words)) == 1:
            return ''

    words = []
    for i, word in enumerate(text.split()):
        #  print(i)
        if i == 0:
            words.append(word)
        else:
            if word == words[len(words) - 1]:
                continue
            else:
                words.append(word)

    return ' '.join(words)


def main():
    freq_dict = Counter()
    # read pair
    pair_save_f = open(args.train_convos_path, 'w')
    with open(args.pseudo_convos_path, 'r', encoding='utf-8') as f:
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

            if len(context.split()) > 0:
                context = clean_number_url(context)
                context = clean_repeat(context)
                if len(context.split()) > 0:
                    freq_dict.update(context.split())

            query = clean_number_url(query)
            query = clean_repeat(query)
            if len(query.split()) > 0:
                freq_dict.update(query.split())

            response = clean_number_url(response)
            response = clean_repeat(response)
            if len(response.split()) > 0:
                freq_dict.update(response.split())

            pair_save_f.write('%s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s\n' % \
                    (subreddit_name, conversation_id, context, query, response, hash_value, score, turn))

    pair_save_f.close()

    fact_save_f = open(args.train_facts_path, 'w')
    with open(args.pseudo_facts_path, 'r', encoding='utf-8') as f:
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
