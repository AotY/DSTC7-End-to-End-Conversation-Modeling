#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

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
import re
import string
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter

from utils import Tokenizer
from gensim.models import KeyedVectors

#  import es_helper
#  es = es_helper.get_connection()

stopWords = set(stopwords.words('english'))
punctuations = list(string.punctuation)

tokenizer = Tokenizer()

def remove_stop_words(words):
    words = [word for word in words if word not in stopWords]
    words = [word for word in words if word not in punctuations]
    return words


def wiki_stats(facts_path):
    title_count = 0
    wiki_count = 0
    wiki_table_count = 0

    is_wiki = False
    last_domain = None
    table_line_count = 0
    maybe_table = False

    table_filename = './wiki_table.txt'
    table_file = open(table_filename, 'w', encoding='utf-8')

    wiki_table_dict = {}
    wiki_h2_dict = {}
    h2_count = 0

    wiki_abstract_dict = {}
    wiki_reference_dict = {}
    abstract_count = 0

    with open(facts_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            parts = line.split('\t')

            subreddit_name = parts[0]
            conversation_id = parts[1]
            domain_name = parts[2]
            fact = parts[3]

            if fact.find('<title>') != -1:
                title_count += 1

                if domain_name == 'en.wikipedia.org':
                    table_file.write(fact + '\n')
                    wiki_count += 1

                    # parser table
                    """
                    jump to: navigation, search
                    ~
                    <p>
                    """
                    is_wiki = True
                else:
                    is_wiki = False

                table = []
                table_line_count = 0
                maybe_table = False
                h2s = []

                abstracts = []
                maybe_abstract = True

                references = []
                maybe_refenrence = False

                last_domain = domain_name
                continue

            if is_wiki and domain_name == last_domain:

                if fact.find('jump to : navigation , search') != -1:
                    maybe_table = True
                    table = []
                    continue

                if fact.find('<h2>') != -1:
                    soup = BeautifulSoup(fact, 'lxml')
                    h2 = soup.text.replace('[ edit ]', '').strip()
                    h2_words = remove_stop_words(tokenizer.tokenize(h2))
                    h2s.append(h2_words)

                    if len(abstracts) > 0 and maybe_abstract:
                        wiki_abstract_dict[conversation_id] = abstracts
                        abstract_count += 1

                    abstracts = []
                    maybe_abstract = False

                if fact.find('<h2> references') != -1 or fact.find('<h2> notes') != -1:
                    maybe_refenrence = True

                if fact.find('<h2> external') != -1 or fact.find('<h2> navigation') != -1 or fact.find('<h2> further ') != -1:
                    maybe_refenrence = False

                if fact.find('<p>') != -1 and maybe_abstract:
                    soup = BeautifulSoup(fact, 'lxml')
                    abstract = soup.text.strip()
                    abstract_words = remove_stop_words(tokenizer.tokenize(abstract))
                    abstracts.append(abstract_words)

                if fact.find('"') != -1 and maybe_refenrence:
                    reference = fact.strip()
                    parts = re.findall(r'".+"', reference)
                    if len(parts) > 0:
                        reference = ' '.join(parts).replace('"', '')
                        refenrence_words = remove_stop_words(tokenizer.tokenize(reference))
                        references.append(refenrence_words)

                if maybe_table:
                    if fact.find('<p>') != -1 or fact.find('<h2>') != -1:
                        if table_line_count > 2:
                            if len(table) > 0:
                                wiki_table_dict[conversation_id] = table
                                wiki_table_count += 1

                        maybe_table = False
                        table_line_count = 0
                        table = []
                        table_file.write('----------------------\n')
                    else:
                        if len(fact) > 3:
                            table_line_count += 1
                            table_file.write(fact + '\n')
                            fact_words = remove_stop_words(tokenizer.tokenize(fact))
                            table.append(fact_words)

            if domain_name != last_domain or fact.find('<h2> navigation') != -1 or fact.find('<h2> external') != -1:
                if is_wiki and len(h2s) > 0:
                    wiki_h2_dict[conversation_id] = h2s
                    h2_count += 1

                if is_wiki and len(references) > 0:
                    wiki_reference_dict[conversation_id] = references

                is_wiki = False
                domain_name = None
                maybe_table = False
                table_line_count = 0
                table = []
                h2s = []
                abstracts = []
                maybe_refenrence = False
                references = []

    table_file.close()

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

    return wiki_table_dict, wiki_h2_dict, wiki_abstract_dict, wiki_reference_dict

def conversation_table_stats(wiki_table_dict, wiki_h2_dict, wiki_abstract_dict, conversation_path, fasttext=None):
    conversation_ids = set()
    word_counter_dict = {}
    save_f = open('./fact_conversation_word_count.txt', 'w', encoding='utf-8')
    with open(conversation_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            conversation_id, conversation, response, _ = line.rstrip().split('SPLITTOKEN')

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


if __name__ == '__main__':
    facts_path = './../data/raw_facts.txt'
    conversation_path = '../data/conversations_responses.pair.txt'
    wiki_table_dict, wiki_h2_dict, wiki_abstract_dict, wiki_reference_dict = wiki_stats(facts_path)

    # similarity words
    #  vec_file = '/home/taoqing/Research/data/wiki-news-300d-1M-subword.vec.bin'
    #  fasttext = KeyedVectors.load_word2vec_format(vec_file, binary=True)
    #  conversation_table_stats(wiki_table_dict, wiki_h2_dict, wiki_abstract_dict, conversation_path, fasttext)



