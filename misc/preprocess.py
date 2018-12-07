# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import logging
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append('..')

from misc.vocab import Vocab
from misc.utils import Tokenizer
from misc.misc_opts import preprocess_opt
from misc import es_helper
from embedding.embedding_opt import train_embedding_opt
from embedding.utils import build_vocab_word2vec, build_vocab_fastText
from embedding import train_embedding

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


def read_convos(convos_file_path, logger, args):
    queries = list()

    contexts = list()
    responses = list()

    subreddit_names = list()
    conversation_ids = list()
    response_scores = list()
    dialogue_turns = list()
    hash_values = list()

    logger.info('read convos...')
    with open(convos_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):

            line = line.rstrip()
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

            query_tokens = sentences_tokens[-1]
            context_tokens = sentences_tokens[:-1]

            response_tokens = tokenizer.tokenize(response)
            response_length = len(response_tokens)

            if response_length < args.r_min_len or response_length > args.r_max_len:
                continue

            queries.append(query_tokens)
            contexts.append(context_tokens)
            responses.append(response_tokens)

            hash_values.append(sub[0])
            subreddit_names.append(sub[1])
            conversation_ids.append(sub[2])
            response_scores.append(sub[3])
            dialogue_turns.append(sub[4])

    return contexts, queries, responses, \
        hash_values, subreddit_names, conversation_ids, \
        response_scores, dialogue_turns


''' save abnormal datas'''


def save_abnormal_datas(datas, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write("%s\n" % (' '.join(data)))


'''
Read facts file.
'''


def read_facts(facts_file_path, logger, args):
    facts = []

    hash_values = []
    subreddit_names = []
    conversation_ids = []
    domain_names = []

    with open(facts_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()

            sub = line.split('\t')

            fact = sub[-1]

            fact_tokens = tokenizer.tokenize(fact)
            fact_len = len(fact_tokens)

            # skip if source has nothing
            if fact_len < args.f_min_len or fact_len > args.f_max_len:
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


def stat_frequency(datas, datas_name, min_count=3, vocab_size=8e5, logger=None):
    freq_dict = {}
    vocab_size = int(vocab_size)
    total_token_nums = 0
    token_len_dict = {}
    for data in datas:
        total_token_nums += len(data)
        for token in data:
            freq_dict.setdefault(token, 0)
            freq_dict[token] += 1

            token_len_dict.setdefault(len(token), 0)
            token_len_dict[len(token)] += 1

    total_type_nums = len(freq_dict)

    sorted_freq_list = sorted(freq_dict.items(), key=lambda d: d[1], reverse=True)
    freq_path = '_'.join(datas_name) + '.freq.txt'
    with open(freq_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))

    sorted_len_list = sorted(token_len_dict.items(), key=lambda d: d[0], reverse=False)
    token_len_path = '_'.join(datas_name) + '_token_len.freq.txt'
    with open(token_len_path, 'w', encoding='utf-8') as f:
        for item in sorted_len_list:
            f.write('%d\t%d\n' % (item[0], item[1]))

    print('token size: %d' % len(sorted_freq_list))
    return sorted_freq_list, total_token_nums, total_type_nums


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
    for name, conversation_id, context, query, response, hash_value, score, turn in \
            zip(names, conversation_ids, queries, contexts, responses, hash_values, scores, turns):

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


def save_to_es(es, datas_zip, doc_type='conversation'):
    if doc_type == es_helper.conversation_type:
        for hash_value, subreddit_name, conversation_id, response_score, dialogue_turn in datas_zip:
            body = {
                'hash_value': hash_value,
                'subreddit_name': subreddit_name,
                'conversation_id': conversation_id,
                'response_score': response_score,
                'dialogue_turn': dialogue_turn
            }
            es_helper.insert_to_es(
                es, body, es_helper.index, doc_type)
    elif doc_type == es_helper.fact_type:
        for hash_value, subreddit_name, conversation_id, domain_name, fact in datas_zip:
            body = {
                'hash_value': hash_value,
                'subreddit_name': subreddit_name,
                'conversation_id': conversation_id,
                'domain_name': domain_name,
                'fact': fact
            }
            es_helper.insert_to_es(
                es, body, es_helper.index, doc_type)


'''save count'''


def save_conversations_responses_facts_count(contexts, responses, facts):
    with open('conversations_responses_count.txt', 'w', encoding='utf-8') as f:
        f.write("%s\t%d\n" % ('contexts', len(contexts)))
        f.write("%s\t%d\n" % ('responses', len(responses)))
        f.write("%s\t%d\n" % ('facts', len(facts)))


'''save raw pair'''


def save_raw_pair(raw_conversations, raw_responses, hash_values):
    with open(os.path.join(args.save_path, 'conversations_responses_raw_pair.txt'), 'w', encoding='utf-8') as f:
        for conversation, response, hash_value in zip(raw_conversations, raw_responses, hash_values):
            f.write("%sSPLITTOKEN%sSPLITTOKEN\%s\n" %
                    (conversation, response, hash_value))


'''save token, type nums '''


def save_token_type_nums(total_token_nums, total_type_nums):
    with open('token_type_nums.txt', 'w', encoding='utf-8') as f:
        f.write("%s\t%d\n" % ('token', total_token_nums))
        f.write("%s\t%d\n" % ('type', total_type_nums))
        f.write("%s\t%.4f\n" %
                ('token/type', total_token_nums/total_type_nums))


''' save_conversation_response_facts_nums '''


def save_conversation_response_facts_nums(name_num_dict, filename):
    avg_num = sum(list(name_num_dict.values())) / len(name_num_dict)
    # sort
    sorted_list = sorted(name_num_dict.items(),
                         key=lambda item: item[1], reverse=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('%s\t%.4f\n' % ('avg', avg_num))
        for name, count in sorted_list:
            f.write("%s\t%d\n" % (name, count))


''' save_out_of_vocab_words '''


def save_out_of_vocab_words(out_of_vocab_words, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for word in out_of_vocab_words:
            f.write('%s\n' % word)


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
    train_embedding_opt(parser)
    args = parser.parse_args()
    model_name = args.model_name

    logger.info('args.vocab_size: %d ' % int(args.vocab_size + 4))
    args.vocab_path = args.vocab_path.format(args.model_name, int(args.vocab_size + 4))

    contexts, queries, responses, \
        hash_values, subreddit_names, conversation_ids, \
        response_scores, dialogue_turns = read_convos(args.convos_file_path, logger, args)

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
        domain_names = read_facts(args.facts_file_path, logger, args)

    #  save raw facts to txt
    save_facts(facts, facts_subreddit_names, facts_conversation_ids, domain_names, os.path.join(args.save_path, 'facts.txt'))

    datas = queries + responses + facts
    for context in contexts:
        datas += context

    datas_name = ['contexts', 'queries', 'responses', 'facts']

    sorted_freq_list, total_token_nums, total_type_nums = stat_frequency(
        datas, datas_name, args.min_count, args.vocab_size, logger)

    logger.info('Preprocessing finished.')
