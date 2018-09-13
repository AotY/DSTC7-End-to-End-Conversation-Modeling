# -*- coding: utf-8 -*-
import os
import sys
import logging
import argparse

from collections import Counter

from vocab import Vocab
from vocab import PAD, SOS, EOS, UNK
from tokenize import Tokenize
from opts import preprocess_opt

'''
Generate

vocab
source_num.txt
target_num.txt

and
fact_num.txt
'''

tokenizer = Tokenize()

'''
Read convos file.
'''


def read_convos(convos_file_path, logger=None):
    with open(convos_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 读取  convos，保存到
    conversations = []
    responses = []

    # subreddit_names = []
    # conversation_ids = []
    # responses_score = []
    # dialogues_turn = []

    conversation_max_length = 0
    response_max_length = 0

    n = 0
    for line in lines:
        n += 1

        if n % 1e5 == 0:
            logger.info('checked %.2fM/%.2fM lines' % (n / 1e6, len(lines) / 1e6))

        sub = line.split('\t')

        conversation = sub[-2]
        response = sub[-1]
        if conversation == 'START' or len(conversation.rstrip()) == 0:  # skip if source has nothing
            continue

        conversation_tokens = tokenizer.preprocess(conversation)
        conversation_max_length = max(conversation_max_length, len(conversation_tokens))
        conversations.append(conversation_tokens)

        response_tokens = tokenizer.preprocess(response)
        response_max_length = max(response_max_length, len(response_tokens))
        responses.append(response_tokens)

    return conversations, responses, conversation_max_length, response_max_length


'''
Read facts file.
'''


def read_facts(facts_file_path, logger):
    with open(facts_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 读取  facts，保存到
    facts = []

    # subreddit_names = []
    # conversation_ids = []
    # domain_names = []

    n = 0
    for line in lines:
        n += 1

        if n % 1e5 == 0:
            logger.info('checked %.2fM/%.2fM lines' % (n / 1e6, len(lines) / 1e6))

        sub = line.split('\t')

        fact = sub[-1]
        if fact == 'START' or len(fact.rstrip()) == 0:  # skip if source has nothing
            continue

        facts.append(tokenizer.preprocess(fact))

    return facts


'''
Statistical frequency
datas, may be conversations + responses or conversations individually.
'''


def stat_frequency(datas, datas_name, min_count=3, max_vocab_size=8e5, logger=None):
    freq_dict = {}
    max_vocab_size = int(max_vocab_size)
    for data in datas:
        for token in data:
            freq_dict.setdefault(token, 0)
            freq_dict[token] += 1

    sorted_freq_list = sorted(
        freq_dict.items(), key=lambda d: d[1], reverse=True)

    if min_count > 0:
        logger.info('Clip tokens by min_count')
        sorted_freq_list = [item for item in sorted_freq_list if item[1] > min_count]

    if max_vocab_size > 0 and len(sorted_freq_list) > max_vocab_size:
        logger.info('Clip tokens by max_vocab_size')
        sorted_freq_list = sorted_freq_list[:max_vocab_size]

    freq_save_path = '_'.join(datas_name) + '.freq.txt'

    with open(freq_save_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))

    print('token size: %d' % len(sorted_freq_list))
    return sorted_freq_list


'''
build vocab
'''


def build_vocab(freq_list):
    vocab = Vocab()
    vocab.build_for_frequency(freq_list)
    vocab.save()
    return vocab


def generate_num(datas, vocab, save_path):
    nums = []
    for tokens in datas:
        nums.append(' '.join([str(id) for id in vocab.words_to_id(tokens)]))
    with open(save_path, 'a', encoding="utf-8") as f:
        f.write('\n'.join(nums))


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
    opt = parser.parse_args()

    logger.info('opt.max_vocab_size: %f ' % opt.max_vocab_size)

    conversations, responses, conversation_max_length, response_max_length = read_convos(opt.convos_file_path, logger)
    logger.info('conversation_max_length: %d ' % conversation_max_length)  # 2429
    logger.info('response_max_length: %d ' % response_max_length)  # 186

    # facts = read_facts(opt.facts_file_path)

    stat_frequency(conversations, ['conversations'], 0, 0, logger)
    stat_frequency(responses, ['responses'], 0, 0, logger)

    datas = conversations + responses
    datas_name = ['conversations', 'responses']
    sorted_freq_list = stat_frequency(datas, datas_name, opt.min_count, opt.max_vocab_size, logger)

    vocab = build_vocab(sorted_freq_list)
    # vocab = Vocab()
    # vocab.load()
    logger.info('vocab_size: %s' % vocab.get_vocab_size())  # 93423

    generate_num(conversations, vocab, opt.conversations_num_save_path)
    generate_num(responses, vocab, opt.responses_num_save_path)

    logger.info('Preprocess finished.')
