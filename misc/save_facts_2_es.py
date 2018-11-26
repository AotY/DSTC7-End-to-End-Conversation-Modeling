#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Save facts to elastic, and retrieval them.
"""
import sys
import string
import argparse

import es_helper
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path of pkl file.')
args = parser.parse_args()

#  facts_p_dict_path = './../facts_p_dict.pkl'

# 1. get connection
es = es_helper.get_connection

# 2. delete index
es_helper.delete_index(es, es_helper.index)

# 3. create settings, mappings
settings_body = {
    "settings": {
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

mappings_body = {
    "mappings": {
        "_doc": {
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
}

es_helper.put_settings(es, es_helper.index, es_helper.fact_type, mappings_body)
es_helper.put_mapping(es, es_helper.index, es_helper.fact_type, settings_body)


# 4. insert to es
facts_dict = pickle.load(open(args.path, 'rb'))
for conversation_id, texts in tqdm(facts_dict.items()):
    if len(ps) == 0:
        continue
    for text in texts:
        body = {
            "conversation_id": conversation_id,
            "text": text
        }
        es_helper.insert_to_es(es, es_helper.index, es_helper.fact_type, body)


print('save success.')

# 5. search test
conversation_id = 'j76an'
query_text = 'musical ride'

query_body = {
    'query': {
        "bool": {
            "must": {
                "term": {"conversation_id": conversation_id}
            }
        },
        "match": {
            "text": {
                "query": query_text
            }
        }
    }
}

result = es.search(index, conversation_type, query_body)
hits = result['hits']['hits']
print(hits)



