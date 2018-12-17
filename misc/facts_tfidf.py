#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
compute tf-idf.
"""

import argparse
import csv
import pickle
from tqdm import tqdm
from collections import Counter
import es_helper

parser = argparse.ArgumentParser()
parser.add_argument('--idf_path', type=str, help='path of idf file.', default='../data/wiki_terms_idf.csv')
parser.add_argument('--facts_path', type=str, help='facts path', default='../data/train.facts.txt')
parser.add_argument('--save_path', type=str, help='tf-idf save path', default='../data/facts_tfidf_dict.pkl')
args = parser.parse_args()

def main():
    es = es_helper.get_connection()
    """
    conversation_ids = es_helper.search_all_conversation_ids(es, es_helper.index, es_helper.fact_type)
    print('conversation_idf: %d' % len(conversation_ids))
    """

    # 1, get all conversation_ids
    conversation_ids = set()
    with open(args.facts_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            if not bool(line):
                continue

            parts = line.split('\t')
            if len(parts) != 4:
                #  print('parts: %d' % len(parts))
                continue

            conversation_id = parts[1]
            conversation_ids.add(conversation_id)
    #  conversation_ids = list(conversation_ids)


    # 2, load wiki idf.
    idf_dict = {}
    with open(args.idf_path, 'r') as csvf:
        csv_reader = csv.reader(csvf, delimiter=',')
        next(csv_reader, None)  # skip the headers
        for row in tqdm(csv_reader):
            #  print('row: ', row)
            token, frequency, total, idf = row
            idf_dict[token] = float(idf)

    # 3, compute tf-idf
    facts_tfidf_dict = {}
    for conversation_id in conversation_ids:
        query_body = es_helper.assemble_search_fact_body(conversation_id)
        _, total = es_helper.search(es, es_helper.index, es_helper.fact_type, query_body, size=0)
        #  print('total: %d' % total)
        if total == 0:
            continue
        hits, _ = es_helper.search(es, es_helper.index, es_helper.fact_type, query_body, size=total)
        texts = []
        tfidf_dict = {}
        for hit in hits:
            text = hit['_source']['text']
            text.replace('__number__', '')
            text.replace('__unk__', '')
            text.replace('__url__', '')
            texts.append(text)
        words = ' '.join(texts).split()
        words = [word for word in words if len(word.split()) > 0]
        counter = Counter(words)
        total_count = sum(counter.values())
        for token, count in counter.items():
            tf = count / total_count
            tfidf = tf * idf_dict.get(token, 0.0)
            tfidf_dict[token] = tfidf

        facts_tfidf_dict[conversation_id] = tfidf_dict

    # 4, save
    #  print(facts_tfidf_dict)
    pickle.dump(facts_tfidf_dict, open(args.save_path, 'wb'))


if __name__ == '__main__':
    main()

