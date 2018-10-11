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



def search_facts_by_conversation_hash_value(es, hash_value):
    # obtain subreddit_name, conversation_id
    query_body = {
        'query': {
            'match': {
                'hash_value': hash_value
            }
        }
    }

    result = es.search(index, conversation_type, query_body)
    hits = result['hits']['hits']
    subreddit_name, conversation_id = hits[0]['_source']['subreddit_name'], hits[0]['_source']['conversation_id']

    # obtain facts by conversation_id
    query_body = {
        'query': {
            'match': {
                'conversation_id': conversation_id
            }
        }
    }
    result = es.search(index, fact_type, query_body)
    hit_count = result['hits']['total']
    hits = result['hits']['hits']

    facts = []
    facts_length = []
    for hit in hits:
        hit = hit['_source']
        fact = hit['fact']
        if len(fact) <= 1:
            continue

        facts.append(fact)
        facts_length.append(len(fact))

    #  assert hit_count == len(facts)
    return hit_count, facts


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
