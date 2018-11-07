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


def search_response_score_turn(es, hash_value):
    query_body = {
        'query': {
            'match': {
                'hash_value': hash_value
            }
        }
    }
    result = es.search(index, conversation_type, query_body)
    hits = result['hits']['hits']
    response_score, response_turn = hits[0]['_source']['response_score'], hits[0]['_source']['dialogue_turn']

    return response_score, response_turn

def search_fact_count(es, hash_value):
    query_body = {
        'query': {
            'match': {
                'hash_value': hash_value
            }
        }
    }
    result = es.search(index, conversation_type, query_body)
    hits = result['hits']['hits']
    conversation_id = hits[0]['_source']['conversation_id']

    # obtain  by conversation_id
    query_body = {
        'query': {
            'match': {
                'conversation_id': conversation_id
            }
        }
    }
    hits = result['hits']['hits']
    total_count = 0
    for hit in hits:
        if len(hit['_source']['fact']) <= 1:
            continue
        total_count += 1

    return total_count


def search_conversation_id(es, hash_value):
    query_body = {
        'query': {
            'match': {
                'hash_value': hash_value
            }
        }
    }

    result = es.search(index, conversation_type, query_body)
    hits = result['hits']['hits']
    if len(hits) == 0:
        return 0, None

    subreddit_name, conversation_id = hits[0]['_source']['subreddit_name'], hits[0]['_source']['conversation_id']

    return conversation_id




def search_facts(es, hash_value):
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
    if len(hits) == 0:
        return 0, None

    subreddit_name, conversation_id = hits[0]['_source']['subreddit_name'], hits[0]['_source']['conversation_id']

    # obtain  by conversation_id
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
    for hit in hits:
        hit = hit['_source']
        fact = hit['fact']
        if len(fact) <= 1:
            continue

        facts.append(fact)

    #  assert hit_count == len()
    return hit_count, facts


if __name__ == '__main__':
    es = get_connection()
    hash_value = 'e16b4ec79cbf3bce455c55c82cacb2585faa4f312b680c6d3fbf5b8e'
    id = search_conversation_id(es, hash_value)
    print(id)
