# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import logging
import argparse


from utils import Tokenizer
from collections import Counter
from misc_opts import preprocess_opt

'''
Generate

'''

tokenizer = Tokenizer()

'''
Read convos file.
'''


def read_convos(args, logger):
    contexts = list()
    queries = list()
    responses = list()

    subreddit_names = list()
    conversation_ids = list()
    response_scores = list()
    dialogue_turns = list()
    hash_values = list()

    logger.info('read convos...')
    n = 0
    remove_lines = [121784, 160504, 161111, 537231, 537712, 633371, \
                    740969, 741684, 1140310, 1141582, 1171210, 1172501, \
                    1171390, 1172681, 11245722, 1244313, 1244316, 1245725]
    with open(args.raw_convos_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            n += 1

            if n in remove_lines:
                continue

            #  if n <= 1244316:
                #  continue

            #  print("line: %d" % n)
            #  print("line: %s" % line)
            if n % 5e4 == 0:
                logger.info('read %d' % n)

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

            for si, sentence in enumerate(sentences):
                # token
                tokens = tokenizer.tokenize(sentence)
                if len(tokens) > args.q_max_len or len(tokens) < args.min_len:
                    if si != len(sentences) - 1:
                        sentences_tokens.clear()
                    continue
                else:
                    sentences_tokens.append(tokens)

            if len(sentences_tokens) == 0:
                continue

            query_tokens = sentences_tokens[-1]
            context_tokens = sentences_tokens[:-1]

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

    logger.info('read facts...')
    n = 0
    with open(args.raw_facts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            n += 1

            #  if n == 200:
                #  break

            #  print('line: %d' % n)
            if n % 5e4 == 0:
                logger.info('read %d' % n)

            sub = line.split('\t')
            fact = sub[-1]

            #  if fact[0] not in ['<', '"', '^']:
                #  continue

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


''' save data to pair, conversation - response '''


def save_train_convos(args, contexts, queries,
                      responses, subreddit_names, conversation_ids,
                      hash_values, scores, turns, save_path):

    '''Save data in pair format.'''
    save_file = open(save_path, 'w', encoding='utf-8')
    for subreddit_name, conversation_id, context, \
        query, response, hash_value, score, turn in zip(subreddit_names, conversation_ids, contexts, \
                queries, responses, hash_values, scores, turns):

        if len(context) == 0:
            context = ''
        else:
            context_texts = []
            for tokens in context:
                text = ' '.join(tokens)
                context_texts.append(text)
            context = ' EOS '.join(context_texts)

        query = ' '.join(query)
        response = ' '.join(response)
        save_file.write('%s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s\n' % \
                (subreddit_name, conversation_id, context, query, response, hash_value, score, turn))

    save_file.close()


def save_facts(facts, subreddit_names, conversation_ids, domain_names, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for fact, subreddit, conversation_id, domain in zip(facts, subreddit_names, conversation_ids, domain_names):
            if isinstance(fact, list):
                fact = ' '.join(fact)
            f.write('%s\t%s\t%s\t%s\n' %
                    (subreddit, conversation_id, domain, fact))

def stat_frequency(datas, freq_save_path):
    #  freq_dict = {}
    freq_dict = Counter()
    for data in datas:
        freq_dict.update(data)
        #  for token in data:
            #  freq_dict.setdefault(token, 0)
            #  freq_dict[token] += 1

    sorted_freq_list = sorted(freq_dict.items(), key=lambda d: d[1], reverse=True)
    with open(freq_save_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))


def main(args, logger):
    contexts, queries, responses, \
    hash_values, subreddit_names, conversation_ids, \
    response_scores, dialogue_turns = read_convos(args, logger)

    save_train_convos(
        args,
        contexts,
        queries,
        responses,
        subreddit_names,
        conversation_ids,
        hash_values,
        response_scores,
        dialogue_turns,
        save_path=args.train_convos_path
    )

    #  read facts
    facts, facts_hash_values, \
        facts_subreddit_names, facts_conversation_ids, \
        domain_names = read_facts(args, logger)

    #  save raw facts to txt
    save_facts(facts, facts_subreddit_names, facts_conversation_ids, \
            domain_names, args.train_facts_path)

    # stats freq
    """
    datas = queries + responses + facts
    for context in contexts:
        if len(context) != 0:
            datas.extend(context)

    stat_frequency(datas, args.freq_save_path)

    """
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

    main(args, logger)

    logger.info('Preprocessing finished.')
