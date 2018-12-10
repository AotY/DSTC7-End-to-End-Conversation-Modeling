# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from elasticsearch import Elasticsearch

import logging

for _ in ("boto", "elasticsearch", "urllib3"):
    logging.getLogger(_).setLevel(logging.CRITICAL)

'''
default config
'''
index = 'dstc-7'
conversation_type = 'conversation'
fact_type = 'fact'
#  fact_type = 'fact_tag'

def get_connection():
    es = Elasticsearch()
    return es

def create_index(es, index):
    # ignore 400 cause by IndexAlreadyExistsException when creating an index
    es.indices.create(index=index, ignore=400)

def delete_index(es, index):
    '''delete index.'''
    result = es.indices.delete(index)
    return result

def close_index(es, index):
    es.indices.close(index)

def open_index(es, index):
    es.indices.open(index)

def put_settings(es, index, settings):
    es.indices.put_settings(index=index, body=settings)

def put_mapping(es, index, doc_type, mapping):
    es.indices.put_mapping(index=index, doc_type=doc_type, body=mapping)

def insert_to_es(es, index, doc_type, body):
    '''insert item into es.'''
    es.index(index=index, doc_type=doc_type, body=body)

def search(es, index, doc_type, query_body, size=20):
    result = es.search(index, doc_type, query_body, size=size)
    hits = result['hits']['hits']
    count = result['hits']['total']
    return hits, count

def assemble_search_fact_body(conversation_id, query_text=None):
    if query_text is None:
        query_body = {
            '_source': ['text'],
            'query': {
                "bool": {
                    "must":{
                        "match": {
                            "conversation_id": conversation_id
                        }
                    }
                }
            }
        }
    else:
        query_body = {
            '_source': ['text'],
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
                                    "query": query_text,
                                    "operator": "and"
                                }
                            }
                        }
                    ]
                }
            }
        }

    return query_body

def search_all_conversation_ids(es, index, doc_type):
    query_body = {
        '_source': ['conversation_id'],
        'query': {}
    }
    _, total = search(es, index, doc_type, query_body, size=0)

    conversation_ids = set()
    hits, _ = search(es, index, doc_type, query_body, size=total)
    for hit in hits:
        conversation_id = hit['_source']['conversation_id']
        conversation_ids.add(conversation_id)

    return list(conversation_ids)



if __name__ == '__main__':
    es = get_connection()
    hash_value = 'e16b4ec79cbf3bce455c55c82cacb2585faa4f312b680c6d3fbf5b8e'
    pass
