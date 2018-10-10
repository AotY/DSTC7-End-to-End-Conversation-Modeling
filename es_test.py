# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import es_helper


def search_conversation_by_hash(es):
    body = {
        'query': {
            'match': {
                'hash_value': '8e832cdf160d1492fedc75d01a6596a0f8c7c537428b0732c470468a'
            }
        }
    }

    res = es.count(es_helper.index, es_helper.conversation_type, body)

    print('hash res: {}'.format(res))


def search_conversation_by_subreddit_and_id(es):
    body = {
        'query': {
            'bool': {
                'must': [
                    {'match': {'subreddit_name': 'TodayILearned'}},
                    {'match': {'conversation_id': 'f2ruz'}}
                ]
            }
        }
    }

    res = es.count(es_helper.index, es_helper.conversation_type, body)
    print(res)


def search_facts_by_subbreddit_and_id(es):
    body = {
        'query': {
            'bool': {
                'must': [
                    {'match': {'subreddit_name': 'TodayILearned'}},
                    {'match': {'conversation_id': 'f2ruz'}}
                ]
            }
        }

    }
    res = es.count(es_helper.index, es_helper.fact_type, body)
    print(res)

if __name__ == "__main__":
    es = es_helper.get_connection()

    search_conversation_by_hash(es)

    search_conversation_by_subreddit_and_id(es)

    search_facts_by_subbreddit_and_id(es)



