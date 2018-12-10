#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Save facts to elastic, and retrieval them.
"""
import argparse
from tqdm import tqdm

import es_helper

parser = argparse.ArgumentParser()
parser.add_argument('--facts_path', type=str, help='facts path.')
parser.add_argument('--task', type=str, help='save | search')
parser.add_argument('--conversation_id', type=str, help='conversation id')
parser.add_argument('--query_text', type=str, help='query text')
args = parser.parse_args()

#  1. get connection
es = es_helper.get_connection()

def save():

    # 2. delete index
    try:
        es_helper.delete_index(es, es_helper.index)
    except Exception as e:
        print('%s does not exist' % es_helper.index)

    # 3. create index
    es_helper.create_index(es, es_helper.index)

    # 4. create settings, mappings
    settings_body = {
        "index": {
            "analysis": {
                "analyzer": {
                    "my_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                                "lowercase"
                        ]
                    },
                    "my_stop_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                                "lowercase",
                                "english_stop"
                        ]
                    }
                },
                "filter": {
                    "english_stop": {
                        "type": "stop",
                        "stopwords": "_english_"
                    }
                }
            }
        }
    }
    es_helper.close_index(es, es_helper.index)
    es_helper.put_settings(es, es_helper.index, settings_body)
    es_helper.open_index(es, es_helper.index)

    mappings_body = {
        "fact": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "my_analyzer",
                    "search_analyzer": "my_stop_analyzer",
                    "search_quote_analyzer": "my_analyzer"
                },
                "conversation_id": {
                    "type": "keyword"
                }
            }
        }
    }

    es_helper.close_index(es, es_helper.index)
    es_helper.put_mapping(es, es_helper.index, es_helper.fact_type, mappings_body)
    es_helper.open_index(es, es_helper.index)

    # 5. insert to es
    with open(args.facts_path, 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()

            _, conversation_id, _, fact = line.split('\t')
            if len(fact.split()) == 0:
                continue

            body = {
                "conversation_id": conversation_id,
                "text": fact
            }
            es_helper.insert_to_es(es, es_helper.index, es_helper.fact_type, body)

    print('save success.')

def search(conversation_id=None, query_text=None):
    conversation_id = 'j76an' or args.conversation_id
    query_text = 'musical ride' or args.query_text

    query_body = {
        'query': {
            "bool": {
                "must":{
                    "match": {
                        "conversation_id": conversation_id
                    }
                },
                "should": [
                    {
                        "match": {
                            "text": {
                                "query": query_text
                            }
                        }
                    }
                ]
            }
        }
    }
    print(query_body)

    hits = es_helper.search(es, es_helper.index, es_helper.fact_type, query_body)
    print(hits)

if __name__ == '__main__':
    if args.task == 'save':
        save()
    elif args.task == 'search':
        search()
