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
#  import es_helper

parser = argparse.ArgumentParser()
parser.add_argument('--idf_path', type=str, help='path of idf file.')
parser.add_argument('--facts_dict_path', type=str, help='facts dict path')
parser.add_argument('--tf_idf_path', type=str, help='tf-idf save path')
args = parser.parse_args()


def main():
    # 1, load wiki idf.
    idf_dict = {}
    with open(args.idf_path, 'r') as csvf:
        spamreader = csv.reader(csvf, delimiter=',')
        for row in tqdm(spamreader):
            token, frequency, total, idf = row
            idf_dict[token] = idf

    # 2, get all conversation_ids
    """
    es = es_helper.get_connection
    conversation_ids = es_helper.search_all_conversation_ids(es, es_helper.index, es_helper.fact_type)
    """
    facts_dict = pickle.load(open(args.facts_dict_path, 'rb'))
    conversation_ids = list(facts_dict.keys())

    # 3, compute tf-idf
    facts_tfidf_dict = {}
    for conversation_id in conversation_ids:
        '''
        query_body = es_helper.assemble_search_fact_body(conversation_id)
        _, total = es_helper.search(es, es_helper.index, es_helper.fact_type, query_body, size=0)
        hits, _ = es_helper.search(es, es_helper.index, es_helper.fact_type, query_body, size=total)

        texts = ''
        tf_idf_dict = {}
        for hit in hits:
            text = hit['_source']['text']
            text.replace('__number__', '')
            text.replace('__url__', '')
            texts += text + ' '
        '''

        tf_idf_dict = {}
        texts = facts_dict[conversation_id]
        words = ' '.join(texts).split()
        words = [word for word in words if len(word) > 0]

        counter = Counter(words)
        total_count = sum(counter.values())
        for word, count in counter.items():
            tf = count / total_count
            tfidf = tf * idf_dict.get(word, 0)
            tf_idf_dict[word] = tfidf

        facts_tfidf_dict[conversation_id] = facts_tfidf_dict

    # 4, save
    print(facts_tfidf_dict)
    pickle.dump(facts_tfidf_dict, open(args.tf_idf_path, 'wb'))


if __name__ == '__main__':
    main()

