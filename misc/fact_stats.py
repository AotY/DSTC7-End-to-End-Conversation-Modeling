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
"""
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from collections import Counter

from utils import Tokenizer

#  import es_helper
#  es = es_helper.get_connection()

stopWords = set(stopwords.words('english'))

tokenizer = Tokenizer()

def remove_stop_words(words):
    return [word for word in words if word not in stopWords]


def table_h2_stats(facts_path):
    title_count = 0
    wiki_count = 0
    wiki_table_count = 0

    is_wiki = False
    last_domain = None
    line_count = 0
    maybe_table = False

    table_filename = './wiki_table.txt'
    table_file = open(table_filename, 'w', encoding='utf-8')

    wiki_table_dict = {}
    wiki_h2_dict = {}
    h2_count = 0

    wiki_abstract_dict = {}
    wiki_reference_dict = {} # TODO
    abstract_count = 0

    with open(facts_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            parts = line.split('\t')

            hash_value = parts[0]
            subreddit_name = parts[1]
            conversation_id = parts[2]
            domain_name = parts[3]
            fact = parts[4]

            if fact.find('<title>') != -1:
                title_count += 1

                if domain_name == 'en.wikipedia.org':
                    table_file.write(fact)
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

                line_count = 0
                last_domain = domain_name
                maybe_table = False
                table = []
                h2s = []
                abstracts = []
                maybe_abstract = True
                continue

            if is_wiki and domain_name == last_domain:
                if fact.find('jump to : navigation , search') != -1:
                    maybe_table = True
                    table = []
                    continue
                elif fact.find('<h2>') != -1:
                    soup = BeautifulSoup(fact, 'lxml')
                    h2 = soup.text.replace('[ edit ]', '').strip()
                    h2_words = remove_stop_words(tokenizer.tokenize(h2.split(' ')))
                    h2s.append(h2_words)

                    if len(abstracts) > 0 and maybe_abstract:
                        wiki_abstract_dict[conversation_id] = abstracts
                        abstract_count += 1

                    abstracts = []
                    maybe_abstract = False

                elif fact.find('<p>') != -1 and maybe_abstract:
                    soup = BeautifulSoup(fact, 'lxml')
                    abstract = soup.text.strip()
                    abstract_words = remove_stop_words(tokenizer.tokenize(abstract.split(' ')))
                    abstracts.append(abstract_words)

            if is_wiki and domain_name == last_domain and maybe_table:
                if fact.find('<p>') != -1 or fact.find('<h2>') != -1:
                    if line_count > 2:
                        if len(table) > 0:
                            wiki_table_dict[conversation_id] = table
                            wiki_table_count += 1

                    maybe_table = False
                    line_count = 0
                    table = []
                    table_file.write('----------------------\n')
                else:
                    if len(fact) > 3:
                        line_count += 1
                        table_file.write(fact)
                        fact_words = remove_stop_words(Tokenizer.tokenize(fact.split(' ')))
                        table.append(fact_words)
                    continue

            if domain_name != last_domain or fact == 'mobile view': # mobile view, last line
                if is_wiki and len(h2s) > 0:
                    wiki_h2_dict[conversation_id] = h2s
                    h2_count += 1

                is_wiki = False
                domain_name = ''
                maybe_table = False
                line_count = 0
                table = []
                h2s = []
                abstracts = []

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

    return wiki_table_dict, wiki_h2_dict, wiki_abstract_dict


def conversation_table_stats(wiki_table_dict, wiki_h2_dict, conversation_path):
    last_conversation_id = ''
    word_counter = Counter()
    save_f = open('./fact_conversation_word_count.txt', 'w', encoding='utf-8')
    with open(conversation_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            conversation_id, conversation, response, hash_value = line.rstrip().split('SPLITTOKEN')

            if not bool(conversation) or not bool(response):
                continue

            #  conversation_id = es_helper.search_conversation_id(es, hash_value)
            if len(word_counter) > 0 and (conversation_id != last_conversation_id or hash_value == '014b92c1c7785b9aab90593fd465d9deab0fb88f9baa97a913077805'):

                # stats count
                table_words = set()
                for words in wiki_table_dict.get(conversation_id, []):
                    for word in words:
                        table_words.add(word)

                h2_words = set()
                for words in wiki_h2_dict.get(conversation_id, []):
                    for word in words:
                        h2_words.add(word)

                save_f.write('%s:\n' % conversation_id)

                save_f.write('\ttable:\n')
                for word in table_words:
                    count = word_counter.get(word, 0)
                    save_f.write('\t\t%s: %d\n' % (word, count))

                save_f.write('\th2:\n')
                for word in h2_words:
                    count = word_counter.get(word, 0)
                    save_f.write('\t\t%s: %d\n' % (word, count))

                abstract_words = set()
                for words in wiki_abstract_dict.get(conversation_id, []):
                    for word in words:
                        abstract_words.add(word)

                save_f.write('\tabstract:\n')
                for word in abstract_words:
                    count = word_counter.get(word, 0)
                    save_f.write('\t\t%s: %d\n' % (word, count))

                save_f.write('-------------------------\n')

                del word_counter
                word_counter = Counter()

            else:
                last_conversation_id = conversation_id

                conversation_words = conversation.split(' ')
                response_words = response.split(' ')

                word_counter.update(conversation_words)
                word_counter.update(response_words)

    save_f.close()


if __name__ == '__main__':
    facts_path = './../data/train.facts.txt'
    conversation_path = '../data/conversations_responses.pair.txt'
    wiki_table_dict, wiki_h2_dict, _ = table_h2_stats(facts_path)
    conversation_table_stats(wiki_table_dict, wiki_h2_dict, conversation_path)

