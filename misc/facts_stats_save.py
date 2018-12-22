#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import os
import re
import string
import pickle
import argparse
from tqdm import tqdm
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter

from utils import Tokenizer
from gensim.models import KeyedVectors
from utils import remove_stop_words

parser = argparse.ArgumentParser()

parser.add_argument('--facts_path', type=str, help='', default='')
parser.add_argument('--f_min_len', type=int, help='', default='')
parser.add_argument('--save_path', type=str, help='', default='')

args = parser.parse_args()

tokenizer = Tokenizer()

def wiki_stats():
    title_count = 0
    wiki_count = 0
    wiki_table_count = 0

    is_wiki = False
    last_conversation_id = 'j5bmm'
    maybe_table = False

    table_filename = './wiki_table.txt'
    table_file = open(table_filename, 'w', encoding='utf-8')

    reference_filename = './wiki_reference.txt'
    reference_file = open(reference_filename, 'w', encoding='utf-8')

    abstract_filename = './wiki_abstract.txt'
    abstract_file = open(abstract_filename, 'w', encoding='utf-8')

    wiki_table_dict = {}
    wiki_h2_dict = {}
    h2_count = 0

    wiki_abstract_dict = {}
    wiki_reference_dict = {}
    abstract_count = 0

    with open(args.facts_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            parts = line.split('\t')
            subreddit_name = parts[0]
            conversation_id = parts[1]
            domain_name = parts[2]
            fact = parts[3]

            if fact.startswith('<title>'):
                title_count += 1

                if domain_name == 'en.wikipedia.org':
                    table_file.write(fact + '\n')
                    reference_file.write(fact + '\n')
                    abstract_file.write(fact + '\n')
                    wiki_count += 1

                    is_wiki = True

                    table = []
                    maybe_table = False
                    h2s = []

                    abstracts = []
                    maybe_abstract = False

                    references = []
                    maybe_refenrence = False

                    last_conversation_id = conversation_id
                else:
                    is_wiki = False

                continue

            if is_wiki and conversation_id == last_conversation_id:

                if fact.find('jump to : navigation , search') != -1:
                    maybe_table = True
                    maybe_abstract = True

                if fact.find('<h2>') != -1:
                    #  h2s.append(fact)
                    soup = BeautifulSoup(fact, 'lxml')
                    h2 = soup.text.replace('[ edit ]', '').strip()
                    h2_words = remove_stop_words(tokenizer.tokenize(h2))
                    h2s.append(h2_words)

                if fact.startswith('1.') or fact.startswith('2.') or fact.startswith('3.') or fact.startswith('4.') or \
                        fact.startswith('5.') or fact.startswith('6.') or fact.endswith('early life') or \
                        fact.endswith('see also') or fact.find('external links') != -1:
                    maybe_abstract = False

                if fact.startswith('<h2> references') or fact.startswith('<h2> notes') \
                        or fact.startswith('<h2> see also'):
                    maybe_refenrence = True

                if fact.startswith('<h2> external') or fact.startswith('<h2> navigation') or \
                        fact.startswith('<h2> further'):
                    maybe_refenrence = False

                if fact.find('<p>') != -1 and maybe_abstract:
                    soup = BeautifulSoup(fact, 'lxml')
                    abstract = soup.text.strip()
                    abstract_words = remove_stop_words(tokenizer.tokenize(abstract))
                    abstracts.append(abstract_words)
                    #  abstracts.append(fact)

                if fact.startswith('^') and maybe_refenrence:
                    reference = fact.strip()
                    parts = re.findall(r'".+"', reference)
                    if len(parts) > 0:
                        reference = ' '.join(parts).replace('"', '')
                        refenrence_words = remove_stop_words(tokenizer.tokenize(reference))
                        references.append(refenrence_words)
                        #  references.append(reference)

                if maybe_table:
                    if fact.find('<p>') != -1 or fact.find('<h2>') != -1:
                        if len(table) > 2:
                            wiki_table_dict[conversation_id] = table
                            wiki_table_count += 1

                        maybe_table = False
                        table = []
                        table_file.write('----------------------\n')
                    else:
                        if len(fact) > 3:
                            table_file.write(fact + '\n')
                            fact_words = remove_stop_words(tokenizer.tokenize(fact))
                            table.append(fact_words)
                            #  table.append(fact)

            if conversation_id != last_conversation_id or fact.startswith('<h2> navigation') or fact.startswith('<h2> external'):
                if is_wiki:
                    if len(h2s) > 0:
                        wiki_h2_dict[conversation_id] = h2s
                        h2_count += 1

                    if len(references) > 0:
                        wiki_reference_dict[conversation_id] = references
                        for reference in references:
                            reference_file.write(' '.join(reference) + '\n')
                    reference_file.write('----------------\n')

                    if len(abstracts) > 0:
                        wiki_abstract_dict[conversation_id] = abstracts
                        abstract_count += 1

                        for abstract in abstracts:
                            abstract_file.write(' '.join(abstract) + '\n')
                    abstract_file.write('----------------\n')

                    is_wiki = False
                    maybe_table = False
                    table = []
                    h2s = []
                    abstracts = []
                    maybe_refenrence = False
                    maybe_abstract = False
                    references = []


    table_file.close()
    reference_file.close()
    abstract_file.close()

    print('title_count: %d' % title_count)
    print('wiki_count: %d' % wiki_count)
    print('wiki ratio: %.4f ' % (wiki_count / title_count))

    print('wiki_table_count: %d' % wiki_table_count)
    print('table_ratio: %.4f' % (wiki_table_count / wiki_count))

    print('h2 count: %d' % h2_count)
    print('h2 ratio: %.4f' % (h2_count / wiki_count))

    print('abstract count: %d' % abstract_count)
    print('abstract ratio: %.4f' % (abstract_count / wiki_count))

    pickle.dump(wiki_table_dict, open('./../data/wiki_table_dict.pkl', 'wb'))
    pickle.dump(wiki_h2_dict, open('./../data/wiki_h2_dict.pkl', 'wb'))
    pickle.dump(wiki_abstract_dict, open('./../data/wiki_abstract_dict.pkl', 'wb'))
    pickle.dump(wiki_reference_dict, open('./../data/wiki_reference_dict.pkl', 'wb'))

    wiki_dict_file = open('./../data/wiki_dict.txt', 'w', encoding='utf-8')
    wiki_dict = {}
    for (hash_value, table) in wiki_table_dict.items():
        reference = wiki_reference_dict.get(hash_value, None)
        abstract = wiki_abstract_dict.get(hash_value, None)

        texts = []

        if table is not None:
            texts.extend([' '.join(words) for words in table if len(words) > 1])

        if reference is not None:
            texts.extend([' '.join(words) for words in reference if len(words) > 1])

        if abstract is not None:
            for words in abstract:
                text = ' '.join(words)
                for text in line.split('.'):
                    if len(text) > 3:
                        texts.append(text)

        for text in texts:
            wiki_dict_file.write(text + '\n')

        wiki_dict_file.write('-----------------' + '\n')

        wiki_dict[hash_value] = texts

    pickle.dump(wiki_dict, open('./../data/wiki_dict.pkl', 'wb'))
    wiki_dict_file.close()


    return wiki_table_dict, wiki_h2_dict, wiki_abstract_dict, wiki_reference_dict

def conversation_table_stats(wiki_table_dict, wiki_h2_dict, wiki_abstract_dict, conversation_path, fasttext=None):
    conversation_ids = set()
    word_counter_dict = {}
    save_f = open('./fact_conversation_word_count.txt', 'w', encoding='utf-8')
    with open(conversation_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            conversation_id, conversation, response, hash_value, score, turn = line.split(
                '\t')

            conversation_words = conversation.split(' ')
            response_words = response.split(' ')

            if conversation_id not in word_counter_dict.keys():
                word_counter_dict[conversation_id] = Counter()

            word_counter_dict[conversation_id].update(conversation_words)
            word_counter_dict[conversation_id].update(response_words)

            conversation_ids.add(conversation_id)


        conversation_ids = list(conversation_ids)
        for conversation_id in tqdm(conversation_ids):
            save_f.write('%s:\n' % conversation_id)

            # stats count
            table_words = set()
            for words in wiki_table_dict.get(conversation_id, []):
                for word in words:
                    table_words.add(word)
                    try:
                        similar_words = fasttext.most_similar(word, topn=5)
                        for s_word in similar_words:
                            table_words.add(word + '_fasttext')
                    except KeyError:
                        continue

            save_f.write('\ttable:\n')
            table_words = sorted(table_words, key=lambda item: len(item), reverse=True)
            for word in table_words:
                if word.find('fasttext') != -1:
                    tmp_word = word.split('_')[0]
                else:
                    tmp_word = word
                count = word_counter_dict[conversation_id].get(tmp_word, 0)
                save_f.write('\t\t%s: %d\n' % (word, count))

            h2_words = set()
            for words in wiki_h2_dict.get(conversation_id, []):
                for word in words:
                    h2_words.add(word)
                    try:
                        similar_words = fasttext.most_similar(word, topn=5)
                        for s_word in similar_words:
                            h2_words.add(word + '_fasttext')
                    except KeyError:
                        continue

            save_f.write('\th2:\n')
            h2_words = sorted(h2_words, key=lambda item: len(item), reverse=True)
            for word in h2_words:
                if word.find('fasttext') != -1:
                    tmp_word = word.split('_')[0]
                else:
                    tmp_word = word
                count = word_counter_dict[conversation_id].get(tmp_word, 0)
                save_f.write('\t\t%s: %d\n' % (word, count))

            abstract_words = set()
            for words in wiki_abstract_dict.get(conversation_id, []):
                for word in words:
                    abstract_words.add(word)
                    try:
                        similar_words = fasttext.most_similar(word, topn=5)
                        for s_word in similar_words:
                            abstract_words.add(word + '_fasttext')
                    except KeyError:
                        continue

            save_f.write('\tabstract:\n')
            abstract_words = sorted(abstract_words, key=lambda item: len(item), reverse=True)
            for word in abstract_words:
                if word.find('fasttext') != -1:
                    tmp_word = word.split('_')[0]
                else:
                    tmp_word = word
                count = word_counter_dict[conversation_id].get(tmp_word, 0)
                save_f.write('\t\t%s: %d\n' % (word, count))

            save_f.write('---------------------------------------------------\n')

    save_f.close()


def facts_stas():
    total = 0
    count_dict = Counter()
    len_dict = Counter()
    facts_dict = {}
    with open(args.facts_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            # should filter by data_type
            _, _, conversation_id, _, fact = line.split('\t')

            words = fact.split()

            if len(words) < args.f_min_len:
                continue

            if count_dict.get(conversation_id, 0) == 0:
                count_dict[conversation_id] = 0
                facts_dict[conversation_id] = []

            total += 1

            count_dict[conversation_id] += 1

            len_dict[len(words)] = len_dict.get(len(words), 0) + 1

            facts_dict[conversation_id].append(' '.join(words))

    print('total facts: %d' % total)
    print('avg facts: %.4f' % (total / len(count_dict)))

    """
    total : 2139145
    avg : 110.3164
    """

    save_distribution(count_dict, 'facts_count')
    save_distribution(len_dict, 'facts_len')

    facts_save_path = os.path.join(args.save_path, 'facts_dict.pkl')
    pickle.dump(facts_dict, open(facts_save_path, 'wb'))


def facts_tag_stats():
    total_tag = 0
    tag_count_dict = Counter()
    tag_len_dict = Counter()
    facts_tag_dict = {}
    with open(args.facts_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            _, conversation_id, _, fact = line.split('\t')

            if tag_count_dict.get(conversation_id, 0) == 0:
                tag_count_dict[conversation_id] = 0
                facts_tag_dict[conversation_id] = []

            if fact.startswith('<p>') or fact.startswith('<h2>') or fact.startswith('<h3>') or fact.startswith('<h4>'):
                total_tag += 1
                tag_count_dict[conversation_id] += 1
                tag_len_dict[len(fact.split())] = tag_len_dict.get(len(fact.split()), 0) + 1

                text = BeautifulSoup(fact, 'lxml').text
                text = re.sub(r'\[ [\d|\w]+ ]', '', text)
                words = tokenizer.tokenize(text)
                # remove <number>, <url>
                words = [word for word in words if word not in ['<number>', '<url>']]

                if len(words) < 2:
                    continue
                words = words[:250]
                facts_tag_dict[conversation_id].append(' '.join(words))

    print('total tag: %d' % total_tag)
    print('avg tag: %.4f' % (total_tag / len(tag_count_dict)))
    """
    total tag: 577087
    avg tag: 29.7437
    """

    save_distribution(tag_count_dict, 'facts_tag_count')
    save_distribution(tag_len_dict, 'facts_tag_len')

    pickle.dump(facts_tag_dict, open('./../data/facts_tag_dict.pkl', 'wb'))

def save_distribution(distribution, name):
    distribution_list = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    with open(os.path.join(args.save_path, name + '.dist.txt'), 'w', encoding="utf-8") as f:
        for i, j in distribution_list:
            f.write('%s\t%s\n' % (str(i), str(j)))

if __name__ == '__main__':
    #  conversation_path = '../data/train.convos.txt'

    #  wiki_table_dict, wiki_h2_dict, wiki_abstract_dict, wiki_reference_dict = wiki_stats(facts_path)
    #  #  similarity words
    #  vec_file = '/home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin'
    #  fasttext = KeyedVectors.load_word2vec_format(vec_file, binary=True)
    #  conversation_table_stats(wiki_table_dict, wiki_h2_dict, wiki_abstract_dict, conversation_path, fasttext)

    #  facts_tag_stats(facts_path)
    facts_stas()

    print('stats finishing...')


"""
<title>(total): 8471
Wikipedia: 6925

title_count: 8555
wiki_count: 6925
wiki ratio: 0.8095
wiki_table_count: 5351
table_ratio: 0.7727
h2 count: 6909
h2 ratio: 0.9977
abstract count: 6916
abstract ratio: 0.9987


title_count: 33386
wiki_count: 15211

wiki ratio: 0.4556

wiki_table_count: 11062
table_ratio: 0.7272

h2 count: 139
h2 ratio: 0.0091

abstract count: 15116
abstract ratio: 0.9938
"""

