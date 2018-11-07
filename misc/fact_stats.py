#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
<title> 8471

Wikipedia: 6925

Wikipedia contains table

title_count: 8471
wiki_count: 6925
wiki ratio: 0.8175
wiki_table_count: 5052
table_ratio: 0.7295
"""
import os
import pickle
from tqdm import tqdm


title_count = 0
wiki_count = 0
wiki_table_count = 0

is_wiki = False
last_domain = None
line_count = 0
maybe_table = False

facts_path = './../data/train.facts.txt'
table_filename = './wiki_table.txt'
table_file = open(table_filename, 'w', encoding='utf-8')

wiki_table_dict = {}

with open(facts_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        if not bool(line.rstrip()):
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
            continue

        if is_wiki and domain_name == last_domain:
            if fact.find('jump to : navigation , search') != -1:
                maybe_table = True
                table = []
                continue

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
                    table.append(fact)
                continue

        if domain_name != last_domain:
            is_wiki = False
            domain_name = ''
            maybe_table = False
            line_count = 0
            table = []

table_file.close()


#  print('title_count: %d' % title_count)
#  print('wiki_count: %d' % wiki_count)
#  print('wiki ratio: %.4f ' % (wiki_count / title_count))
#  print('wiki_table_count: %d' % wiki_table_count)
#  print('table_ratio: %.4f' % (wiki_table_count / wiki_count))

#  pickle.dump(wiki_table_dict, open('./../data/wiki_table_dict.pkl', 'wb'))






