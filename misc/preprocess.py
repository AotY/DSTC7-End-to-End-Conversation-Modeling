# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import logging
import argparse


from utils import Tokenizer
from misc_opts import preprocess_opt

'''
Generate

vocab
source_num.txt
target_num.txt

and
fact_num.txt
'''

tokenizer = Tokenizer()

'''
Read convos file.
'''


def read_convos(args, logger):
    queries = list()

    contexts = list()
    responses = list()

    subreddit_names = list()
    conversation_ids = list()
    response_scores = list()
    dialogue_turns = list()
    hash_values = list()

    logger.info('read convos...')
    n = 0
    with open(args.convos_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            n += 1
            if n % 1e5 == 0:
                logger.info('checked %.2fM' % (n / 1e6))

            if n == 1000:
                break

            sub = line.split('\t')

            conversation = sub[-2]
            response = sub[-1]

            # skip if source has nothing
            if conversation == 'START' or len(conversation.rstrip()) == 0:
                continue

            if conversation.startswith('START EOS'):
                conversation = conversation[10:]
            elif conversation.startswith('EOS'):
                conversation = conversation[4:]
            elif conversation.startswith('... EOS'):
                conversation = conversation[7:]
            elif conversation.startswith('... '):
                conversation = conversation[4:]

            sentences = conversation.split('EOS')
            sentences_tokens = list()

            for sentence in sentences:
                # token
                tokens = tokenizer.tokenize(sentence)
                if tokens is None or len(tokens) < args.min_len:
                    continue
                sentences_tokens.append(tokens)

            if len(sentences_tokens) == 0:
                continue

            print('sentences_tokens: ', sentences_tokens)

            query_tokens = sentences_tokens[-1]
            context_tokens = sentences_tokens[:-1]
            print('query_tokens: ', query_tokens)
            print('context_tokens: ', context_tokens)

            response_tokens = tokenizer.tokenize(response)
            response_length = len(response_tokens)

            if response_length < args.min_len or response_length > args.r_max_len:
                continue

            contexts.append(context_tokens)
            queries.append(query_tokens)
            responses.append(response_tokens)

            hash_values.append(sub[0])
            subreddit_names.append(sub[1])
            conversation_ids.append(sub[2])
            response_scores.append(sub[3])
            dialogue_turns.append(sub[4])

    return contexts, queries, responses, \
        hash_values, subreddit_names, conversation_ids, \
        response_scores, dialogue_turns




'''
Read facts file.
'''


def read_facts(args, logger):
    facts = []

    hash_values = []
    subreddit_names = []
    conversation_ids = []
    domain_names = []

    n = 0
    with open(args.facts_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            n += 1
            if n % 1e5 == 0:
                logger.info('checked %.2fM' % (n / 1e6))

            if n == 1000:
                break

            sub = line.split('\t')

            fact = sub[-1]

            fact_tokens = tokenizer.tokenize(fact, html=True)
            fact_len = len(fact_tokens)

            # skip if source has nothing
            if fact_len < args.min_len or fact_len > args.f_max_len:
                continue

            facts.append(fact_tokens)

            hash_values.append(sub[0])
            subreddit_names.append(sub[1])
            conversation_ids.append(sub[2])
            domain_names.append(sub[3])


    return facts, hash_values, subreddit_names, \
            conversation_ids, domain_names


'''
Statistical frequency
datas, may be contexts + responses or contexts individually.
'''


def stat_frequency(datas, datas_name, logger=None):
    freq_dict = {}
    for data in datas:
        for token in data:
            freq_dict.setdefault(token, 0)
            freq_dict[token] += 1


    sorted_freq_list = sorted(freq_dict.items(), key=lambda d: d[1], reverse=True)
    freq_path = '_'.join(datas_name) + '.freq.txt'
    with open(freq_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))

    return sorted_freq_list


def save_distribution(distribution, name):
    distribution_list = sorted(distribution.items(), key=lambda item: item[0])
    with open(name + '.len.distribution.txt', 'w', encoding="utf-8") as f:
        f.write('length\tcount\n')
        for length, count in distribution_list:
            f.write('%d\t%d\n' % (length, count))


''' save data to pair, conversation - response '''


def save_data_to_pair(args, contexts, queries,
                      responses, names, conversation_ids,
                      hash_values, scores, turns, filename):

    '''Save data in pair format.'''
    save_file = open(os.path.join(args.save_path, filename), 'w', encoding='utf-8')
    for name, conversation_id, context, \
        query, response, hash_value, score, turn in \
            zip(names, conversation_ids, contexts, \
                queries, responses, hash_values, scores, turns):

        print('context: ', context)
        print('query: ', query)
        print('response: ', response)

        if len(context) == 0:
            context = ''
        else:
            texts = []
            for tokens in context:
                text = ' '.join(tokens)
                texts.append(text)
            context = ' EOS '.join(texts)

        query = ' '.join(query)
        response = ' '.join(response)
        save_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % \
                (name, conversation_id, context, query, response, hash_value, score, turn))

    save_file.close()




def save_facts(facts, subreddit_names, conversation_ids, domain_names, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for fact, subreddit, conversation_id, domain in zip(facts, subreddit_names, conversation_ids, domain_names):
            if isinstance(fact, list):
                fact = ' '.join(fact)
            f.write('%s\t%s\t%s\t%s\n' %
                    (subreddit, conversation_id, domain, fact))


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s", ' '.join(sys.argv))

    # get optional parameters
    parser = argparse.ArgumentParser(description=program,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    preprocess_opt(parser)
    args = parser.parse_args()
    model_name = args.model_name

    contexts, queries, responses, \
        hash_values, subreddit_names, conversation_ids, \
        response_scores, dialogue_turns = read_convos(args, logger)

    save_data_to_pair(
        args,
        contexts,
        queries,
        responses,
        subreddit_names,
        conversation_ids,
        hash_values,
        response_scores,
        dialogue_turns,
        filename='train.pair.txt'
    )

    #  read facts
    facts, facts_hash_values, \
        facts_subreddit_names, facts_conversation_ids, \
        domain_names = read_facts(args, logger)

    #  save raw facts to txt
    save_facts(facts, facts_subreddit_names, facts_conversation_ids, \
            domain_names, os.path.join(args.save_path, 'facts.txt'))

    datas = queries + responses + facts
    for context in contexts:
        datas += context

    datas_name = ['contexts', 'queries', 'responses', 'facts']

    sorted_freq_list = stat_frequency(datas, datas_name, logger)

    logger.info('Preprocessing finished.')
