# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from elasticsearch import Elasticsearch

'''
default config
'''
index = 'dstc-7'
conversation_type = 'conversation'
fact_type = 'fact'


def get_connection():
    es = Elasticsearch()
    return es


def create_index(es, index, doc_type=None):
    # ignore 400 cause by IndexAlreadyExistsException when creating an index
    es.indices.create(index=index, ignore=400)


def delete_index(es, index):
    '''delete index.'''
    result = es.indices.delete(index)
    return result


def insert_to_es(es, body, index, doc_type):
    '''insert item into es.'''
    es.index(index=index, doc_type=doc_type, body=body)


def simple_search(es, doc_type, query):
    '''

    :param es: connection
    :param index: index
    :param doc_type: doc_type
    :param query: search query
    :return:
    '''
    body = {
        'query': {
            'match': {
                'fact': query
            }
        }
    }
    result = es.search(index=index, doc_type=doc_type, body=body)
    hit_count = result['hits']['total']
    hits = result['hits']['hits']
    return hit_count, hits


def get_normal_search(query):
    # match
    normal_search = [
        {
            'match': {
                'title': {
                    'query': query.get('query'),
                    'minimum_should_match': '50%',
                    'boost': 3
                }
            }
        },
        {
            'match': {
                'description': {
                    'query': query.get('query'),
                    'boost': 2,
                    'minimum_should_match': '50%',
                }

            }
        },
        {
            'match': {
                'keywords': {
                    'query': query.get('query'),
                    'boost': 3,
                    'minimum_should_match': '50%',
                }

            }
        },
        {
            'match': {
                'content': {
                    'query': query.get('query'),
                    'minimum_should_match': '50%',
                }

            }
        }
    ]
    return normal_search


def normal_search(es, query, doc_type, page_from=0, page_size=10):
    body = {
        'from': page_from,
        'size': page_size,
        'query': {
            'bool': {
                'should': {
                    'match': {
                        'fact': {
                            'query': query,
                        }
                    }
                },
                'minimum_should_match': 1,
            }
        }
    }
    result = es.search(index=index, doc_type=doc_type, body=body)
    hit_count = result['hits']['total']
    hits = result['hits']['hits']
    return hit_count, hits
