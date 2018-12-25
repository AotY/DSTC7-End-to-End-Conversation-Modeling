# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import logging
import argparse
from tqdm import tqdm

from utils import Tokenizer
from misc_opts import preprocess_opt

tokenizer = Tokenizer()

'''
Read convos file.
'''


def read_convos(args, logger):
    convos = list()

    hash_values_set = set()

    logger.info('read convos...')
    n = 0
    #  remove_lines = [172480, 172525, 206247, 649956, 726379, 1032984, 1032990, 1033080, \
                    #  1033109, 1033112, 1033152, 1033545, 1239300, 1294540, 1733732, 1764651, \
                    #  1764831, 1798849, 1798858, 1843289, 1850559, 1991542, 1991548, 1992017, \
                    #  2100661, 2100695, 2100887, 2100888, 2101321, 2163997, 2164372, 2170863, \
                    #  2171263, 2178114, 2178117, 2181342, 2398186, 2587101]
    with open(args.raw_convos_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            n += 1

            #  if n in remove_lines:
                #  continue

            #  if n >= 200:
                #  break

            #  if n <= 2550000:
                #  continue

            #  print("line: %d" % n)
            #  print("line: %s" % line)

            parts = line.split('\t')
            if len(parts) != 8:
                print('line: %s' % line)
                continue

            #  data_type, hash_value, subreddit_name, \
                #  conversation_id, turn, score, \
                #  conversation, response = line.split('\t')

            data_type = parts[0]
            #  if data_type != 'VALID':
                #  continue

            hash_value = parts[1]
            if hash_value in hash_values_set:
                continue
            else:
                hash_values_set.add(hash_value)

            conversation = parts[-2]
            response = parts[-1]

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

            sentences = conversation.split(' EOS ')
            sentences_tokens = list()

            for si, sentence in enumerate(sentences):
                if len(sentence.split()) > args.q_max_len or len(sentence.split()) < args.min_len:
                    continue

                # token
                tokens = tokenizer.tokenize(sentence)
                if len(tokens) < args.min_len:
                    if si != len(sentences) - 1:
                        sentences_tokens.clear()
                    continue

                sentences_tokens.append(tokens)

            if len(sentences_tokens) == 0:
                continue

            query_tokens = sentences_tokens[-1]
            context_tokens = sentences_tokens[:-1]

            if data_type != 'TEST':
                response_tokens = tokenizer.tokenize(response)
                response_length = len(response_tokens)
                if response_length < args.min_len or response_length > args.r_max_len:
                    continue
            else:
                response_tokens = response.split()

            convos.append((context_tokens, query_tokens, response_tokens, \
                           data_type, hash_value, parts[2], parts[3], \
                           parts[4], parts[5]))

    return convos

'''
Read facts file.
'''


def read_facts(args, logger):
    facts = []

    data_types = []
    hash_values = []
    subreddit_names = []
    conversation_ids = []
    domain_names = []

    hash_values_set = set()

    logger.info('read facts...')
    n = 0
    with open(args.raw_facts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            n += 1

            #  if n == 200:
                #  break

            #  if n <= 15158440:
                #  continue

            #  print(line)

            #  print('line: %d' % n)
            if n % 5e4 == 0:
                logger.info('read %d' % n)

            parts = line.split('\t')

            if len(parts) != 6:
                print('line: %d' % n)
                continue

            hash_value = parts[1]
            if hash_value in hash_values_set:
                continue
            else:
                hash_values_set.add(hash_value)

            fact = parts[-1]
            data_type = parts[0]

            if len(fact.split()) < args.min_len or len(fact.split()) > args.f_max_len:
                continue

            fact_tokens = tokenizer.tokenize(fact, html=True)
            fact_len = len(fact_tokens)

            # skip if source has nothing
            if fact_len < args.min_len or fact_len > args.f_max_len:
                continue

            facts.append(fact_tokens)

            data_types.append(data_type)
            hash_values.append(hash_value)
            subreddit_names.append(parts[2])
            conversation_ids.append(parts[3])
            domain_names.append(parts[4])

    return facts, data_types, hash_values, \
        subreddit_names, conversation_ids, domain_names


''' save data to pair, conversation - response '''


def save_convos(args, convos, save_path):
    # shuffle
    #  random.shuffle(convos)

    # sort by hash_value
    convos.sort(key=lambda item: item[4], reverse=False)

    '''Save data in pair format.'''
    save_file = open(save_path, 'w', encoding='utf-8')
    for c_tokens, q_tokens, r_tokens, data_type, hash_value, \
        subreddit_name, conversation_id, score, turn in convos:

        if len(c_tokens) == 0:
            context = ''
        else:
            context_texts = []
            for tokens in c_tokens:
                text = ' '.join(tokens)
                context_texts.append(text)
            context = ' EOS '.join(context_texts)

        query = ' '.join(q_tokens)
        response = ' '.join(r_tokens)
        save_file.write('%s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s SPLIT %s\n' % \
                (data_type, subreddit_name, conversation_id, context, query, response, hash_value, score, turn))

    save_file.close()


def save_facts(facts, data_types, subreddit_names, conversation_ids, domain_names, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for data_type, fact, subreddit, conversation_id, domain \
            in zip(data_types, facts, subreddit_names, conversation_ids, domain_names):

            if isinstance(fact, list):
                fact = ' '.join(fact)

            f.write('%s\t%s\t%s\t%s\t%s\n' %
                    (data_type, subreddit, conversation_id, domain, fact))

def main(args, logger):
    """
    convos = read_convos(args, logger)

    save_convos(
        args,
        convos,
        save_path=args.train_convos_path
    )

    """
    #  read facts
    facts, facts_data_types, facts_hash_values, \
        facts_subreddit_names, facts_conversation_ids, \
        domain_names = read_facts(args, logger)

    #  save raw facts to txt
    save_facts(facts, facts_data_types, facts_subreddit_names, facts_conversation_ids, \
            domain_names, args.train_facts_path)


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
