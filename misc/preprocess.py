# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import logging
import argparse

sys.path.append('..')

import numpy as np
from elasticsearch import Elasticsearch

from misc.vocab import Vocab
from misc.tokenizer import Tokenizer
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


def read_convos(convos_file_path, logger=None):
    with open(convos_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 读取  convos，保存到
    conversations = []
    responses = []

    raw_conversations = []
    raw_responses = []

    subreddit_names = []
    conversation_ids = []
    response_scores = []
    dialogue_turns = []
    hash_values = []

    conversation_max_length = 0
    response_max_length = 0

    conversations_length_distribution = {}
    responses_length_distribution = {}

    # each conversation contains the number of response
    conversation_response_nums = {}

    # abnormal_conversations
    abnormal_conversations = []
    abnormal_responses = []

    n = 0
    for line in lines:
        n += 1

        if n % 1e5 == 0:
            logger.info('checked %.2fM/%.2fM lines' %
                        (n / 1e6, len(lines) / 1e6))

        sub = line.split('\t')

        conversation = sub[-2]
        response = sub[-1]

        # skip if source has nothing
        if conversation == 'START' or len(conversation.rstrip()) == 0:
            continue

        # raw data
        raw_conversations.append(conversation)
        raw_responses.append(response)

        # token
        # maybe url --> TAG
        conversation_tokens = tokenizer.preprocess(conversation)
        # replace url
        conversation_tokens = tokenizer.replace_url(conversation_tokens)
        conversation_tokens = tokenizer.replace_number(conversation_tokens)
        conversation_tokens = tokenizer.split_by_hyphen(conversation_tokens)
        conversation_tokens = tokenizer.remove_by_len(conversation_tokens, 13)

        # abnormal lengths: 203, 204, 205, 206, 207
        conversation_length = len(conversation_tokens)

        if conversation_length in [203, 204, 205, 206, 207]:
            abnormal_conversations.append(conversation_tokens + [sub[1], sub[2]])

        conversation_max_length = max(
            conversation_max_length, conversation_length)

        conversations_length_distribution[conversation_length] = conversations_length_distribution.get(
            conversation_length, 0) + 1


        response_tokens = tokenizer.preprocess(response)
        response_tokens = tokenizer.replace_url(response_tokens)
        response_tokens = tokenizer.replace_number(response_tokens)
        response_tokens = tokenizer.split_by_hyphen(response_tokens)
        response_tokens = tokenizer.remove_by_len(response_tokens, 13)

        # abnormal lengths: < 7
        response_length = len(response_tokens)
        if response_length <= 7:
            # save_abnormal_response(response_tokens)
            abnormal_responses.append(response_tokens)

        response_max_length = max(response_max_length, response_length)
        responses_length_distribution[response_length] = responses_length_distribution.get(
            response_length, 0) + 1

        # append to data set
        if response_length <= 3:
            continue

        conversations.append(conversation_tokens)
        responses.append(response_tokens)

        hash_values.append(sub[0].rstrip().replace('\t', '').replace('\\', ''))
        subreddit_names.append(sub[1])
        conversation_ids.append(sub[2])
        response_scores.append(sub[3])
        dialogue_turns.append(sub[4])

        # TodayILearned-f2ruz nums
        key_value = sub[1] + '-' + sub[2]
        conversation_response_nums[key_value] = conversation_response_nums.get(key_value, 0) + 1

    return raw_conversations, raw_responses, \
        conversations, responses, \
        conversations_length_distribution, conversation_max_length, \
        responses_length_distribution, response_max_length, \
        hash_values, subreddit_names, conversation_ids, \
        response_scores, dialogue_turns, conversation_response_nums, \
        abnormal_conversations, abnormal_responses




''' save abnormal datas'''
def save_abnormal_datas(datas, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write("%s\n" % (' '.join(data)))

'''
Read facts file.
'''


def read_facts(facts_file_path, logger):
    with open(facts_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 读取  facts，保存到
    facts = []

    hash_values = []
    subreddit_names = []
    conversation_ids = []
    domain_names = []

    conversation_fact_nums = {}

    max_len = 0
    len_distribution = {}

    abnormal_facts = []

    n = 0
    for line in lines:
        n += 1

        if n % 1e5 == 0:
            logger.info('checked %.2fM/%.2fM lines' %
                        (n / 1e6, len(lines) / 1e6))

        sub = line.split('\t')

        fact = sub[-1]
        # skip if source has nothing
        if fact == 'START' or len(fact.rstrip()) == 0:
            continue

        fact_tokens = tokenizer.preprocess(fact)
        # url -> tag
        fact_tokens = tokenizer.replace_url(fact_tokens)
        fact_tokens = tokenizer.replace_url(fact_tokens)
        fact_tokens = tokenizer.replace_number(fact_tokens)
        fact_tokens = tokenizer.split_by_hyphen(fact_tokens)
        fact_tokens = tokenizer.remove_by_len(fact_tokens, 13)

        fact_len = len(fact_tokens)
        max_len = max(max_len, fact_len)
        len_distribution[fact_len] = len_distribution.get(fact_len, 0) + 1

        if fact_len <= 7:
            abnormal_facts.append(fact_tokens)

        facts.append(fact_tokens)

        hash_values.append(sub[0].rstrip().replace('\t', '').replace('\\', ''))
        subreddit_names.append(sub[1])
        conversation_ids.append(sub[2])
        domain_names.append(sub[3])

        # TodayILearned-f2ruz nums
        key_value = sub[1] + '-' + sub[2]
        conversation_fact_nums[key_value] = conversation_fact_nums.get(key_value, 0) + 1

    return facts, hash_values, subreddit_names, \
           conversation_ids, domain_names, \
           conversation_fact_nums, max_len, \
           len_distribution, abnormal_facts


'''
Statistical frequency
datas, may be conversations + responses or conversations individually.
'''


def stat_frequency(datas, datas_name, min_count=3, max_vocab_size=8e5, logger=None):
    freq_dict = {}
    max_vocab_size = int(max_vocab_size)
    total_token_nums = 0
    for data in datas:
        total_token_nums += len(data)
        for token in data:
            freq_dict.setdefault(token, 0)
            freq_dict[token] += 1

    total_type_nums = len(freq_dict)

    sorted_freq_list = sorted(
        freq_dict.items(), key=lambda d: d[1], reverse=True)

    if min_count > 0:
        logger.info('Clip tokens by min_count')
        sorted_freq_list = [
            item for item in sorted_freq_list if item[1] > min_count]

    if max_vocab_size > 0 and len(sorted_freq_list) > max_vocab_size:
        logger.info('Clip tokens by max_vocab_size')
        sorted_freq_list = sorted_freq_list[:max_vocab_size]

    freq_save_path = '_'.join(datas_name) + '.freq.txt'

    with open(freq_save_path, 'w', encoding='utf-8') as f:
        for item in sorted_freq_list:
            f.write('%s\t%d\n' % (item[0], item[1]))

    print('token size: %d' % len(sorted_freq_list))
    return sorted_freq_list, total_token_nums, total_type_nums


'''
build vocab
'''


def build_vocab(freq_list):
    vocab = Vocab()
    vocab.build_for_frequency(freq_list)
    vocab.save(opt.vocab_save_path)
    return vocab


def generate_num(datas, vocab, save_path):
    nums = []
    for tokens in datas:
        nums.append(' '.join([str(id) for id in vocab.words_to_id(tokens)]))
    with open(save_path, 'a', encoding="utf-8") as f:
        f.write('\n'.join(nums))


def save_distribution(distribution, name):
    distribution_list = sorted(distribution.items(), key=lambda item: item[0])
    with open(name + '.len.distribution.txt', 'w', encoding="utf-8") as f:
        f.write('length\tcount\n')
        for length, count in distribution_list:
            f.write('%d\t%d\n' % (length, count))


''' save data to pair, conversation - response '''


def save_data_to_pair(opt, conversations, responses, hash_values, filename):
    '''Save data in pair format.'''
    save_file = open(os.path.join(opt.save_path, filename),
                     'w', encoding='utf-8')
    for conversation, response, hash_value in zip(conversations, responses, hash_values):
        save_file.write('%s\t%s\t%s\n' % (
            ' '.join(conversation), ' '.join(response), hash_value))

    save_file.close()


def save_to_es(es, datas_zip, type='conversation'):
    if type == es_helper.conversation_type:
        for hash_value, subreddit_name, conversation_id, response_score, dialogue_turn in datas_zip:
            body = {
                'hash_value': hash_value,
                'subreddit_name': subreddit_name,
                'conversation_id': conversation_id,
                'response_score': response_score,
                'dialogue_turn': dialogue_turn,
            }
            es_helper.insert_to_es(
                es, body, es_helper.index, es_helper.conversation_type)
    elif type == es_helper.fact_type:
        for hash_value, subreddit_name, conversation_id, domain_name, fact in datas_zip:
            body = {
                'hash_value': hash_value,
                'subreddit_name': subreddit_name,
                'conversation_id': conversation_id,
                'domain_name': domain_name,
                'fact': fact,
            }
            es_helper.insert_to_es(
                es, body, es_helper.index, es_helper.conversation_type)


'''save count'''


def save_conversations_responses_facts_count(conversations, responses, facts):
    with open('conversations_responses_count.txt', 'w', encoding='utf-8') as f:
        f.write("%s\t%d\n" % ('conversations', len(conversations)))
        f.write("%s\t%d\n" % ('responses', len(responses)))
        f.write("%s\t%d\n" % ('facts', len(facts)))


'''save raw pair'''


def save_raw_pair(raw_conversations, raw_responses, hash_values):
    with open(os.path.join(opt.save_path, 'conversations_responses_raw_pair.txt'), 'w', encoding='utf-8') as f:
        for conversation, response, hash_value in zip(raw_conversations, raw_responses, hash_values):
            f.write("%s\t%s\t\%s\n" % (conversation, response, hash_value))


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
    sorted_list = sorted(name_num_dict.items(), key=lambda item: item[1], reverse=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('%s\t%.4f\n' % ('avg', avg_num))
        for name, count in sorted_list: 
            f.write("%s\t%d\n" % (name, count))



''' save_out_of_vocab_words '''

def save_out_of_vocab_words(out_of_vocab_words, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for word in out_of_vocab_words:
            f.write('%s\n' % word)


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
    opt = parser.parse_args()

    logger.info('opt.max_vocab_size: %f ' % opt.max_vocab_size)

    raw_conversations, raw_responses, \
        conversations, responses, \
        conversations_length_distribution, conversation_max_length, \
        responses_length_distribution, response_max_length, \
        hash_values, subreddit_names, conversation_ids, \
        response_scores, dialogue_turns, conversation_response_nums, \
        abnormal_conversations, abnormal_responses = read_convos(
            opt.convos_file_path, logger)

    logger.info('conversation_max_length: %d ' %
                conversation_max_length)  # 2429
    logger.info('response_max_length: %d ' % response_max_length)  # 186

    # save conversation response nums
    save_conversation_response_facts_nums(conversation_response_nums, 'conversation_response_nums.txt')

    # save raw pair
    save_raw_pair(raw_conversations, raw_responses, hash_values)

    # save abnormal_conversations, abnormal_responses
    save_abnormal_datas(abnormal_conversations, 'abnormal_conversations.txt')
    save_abnormal_datas(abnormal_responses, 'abnormal_responses.txt')

    # re-save conversations, responses, and facts
    # (%s\t%s\t\%s\t%s) conversation, response, subreddit_name, and conversation_id
    save_data_to_pair(opt, conversations, responses, hash_values,
                      filename='conversations_responses.pair.txt')

    # read facts
    facts, hash_values, \
        subreddit_names, conversation_ids, \
        domain_names, conversation_fact_nums, \
        fact_max_length, facts_length_distribution, \
        abnormal_facts = read_facts(opt.facts_file_path, logger)

    logger.info('fact_max_length: %d ' % fact_max_length)  # 2728

    # save conversations, responses and facts count
    save_conversations_responses_facts_count(conversations, responses, facts)

    # save conversation fact nums
    save_conversation_response_facts_nums(conversation_fact_nums, 'conversation_fact_nums.txt')

    # save abnormal_facts
    save_abnormal_datas(abnormal_facts, 'abnormal_facts.txt')


    '''
    logger.info('Save to ElasticSearch ...')
    es = es_helper.get_connection()

    # delete index
    es_helper.delete_index(es, es_helper.index)

    # save to elasticsearch, conversaton - response
    save_to_es(es, zip(hash_values, subreddit_names, conversation_ids,
                       response_scores, dialogue_turns), type=es_helper.conversation_type)

    # save to elasticsearch, facts
    save_to_es(es, zip(hash_values, subreddit_names, conversation_ids,
                       domain_names, facts), type=es_helper.fact_type)

    '''

    # save lens distribution
    save_distribution(conversations_length_distribution, 'conversations')
    save_distribution(responses_length_distribution, 'responses')
    save_distribution(facts_length_distribution, 'facts')

    stat_frequency(conversations, ['conversations'], 0, 0, logger)
    stat_frequency(responses, ['responses'], 0, 0, logger)

    datas = conversations + responses
    datas_name = ['conversations', 'responses']
    sorted_freq_list, total_token_nums, total_type_nums = stat_frequency(
        datas, datas_name, opt.min_count, opt.max_vocab_size, logger)

    # save token_nums, total_type_nums
    save_token_type_nums(total_token_nums, total_type_nums)
    vocab = build_vocab(sorted_freq_list)
    vocab_size = int(vocab.get_vocab_size())
    logger.info('vocab_size: %s' % vocab_size)  # 93423

    generate_num(conversations, vocab, opt.conversations_num_save_path)
    generate_num(responses, vocab, opt.responses_num_save_path)

    ''' Load pre-trained word embedding, and obtain these word's embedding which in the vocab. '''

    # google word2vec
    vocab_embedding, out_of_vocab_count, out_of_vocab_words = build_vocab_word2vec(
        None,
        vocab,
        opt.google_vec_file,
        opt.google_vec_dim,
        opt.binary,
        os.path.join(opt.save_path, 'google_vec_for_vocab.%d.%dd.txt' %
                     (vocab_size, opt.google_vec_dim)),
        logger)

    np.save(os.path.join(opt.save_path, 'google_vec_for_vocab.%d.%dd.npy' % (vocab_size, opt.google_vec_dim)),
            vocab_embedding)
    logger.info('build_vocab_word2vec(google_vec_file) finished. out_of_vocab_count: %d' %
                out_of_vocab_count)  #

    # save out of vocab words
    save_out_of_vocab_words(out_of_vocab_words, 'out_of_vocab_words_word2vec.txt')

    # fastText
    vocab_embedding, out_of_vocab_count, out_of_vocab_words = build_vocab_fastText(
        None,
        vocab,
        opt.fasttext_vec_file,
        opt.fasttext_vec_dim,
        None,
        os.path.join(opt.save_path, 'fasttext_vec_for_vocab.%d.%dd.txt' % (vocab_size, opt.google_vec_dim)), logger)

    np.save(os.path.join(opt.save_path, 'fasttext_vec_for_vocab.%d.%dd.npy' % (vocab_size, opt.google_vec_dim)),
            vocab_embedding)
    logger.info('build_vocab_word2vec(fasttext_vec_file) finished. out_of_vocab_count: %d' %
                out_of_vocab_count)  #

    # save out of vocab words
    save_out_of_vocab_words(out_of_vocab_words, 'out_of_vocab_words_fastText.txt')

    # training own word embedding.
    max_sentence_length = (int)(conversation_max_length * 2.0 / 4)
    word2vec_model = train_embedding.start_train(
        datas, opt, max_sentence_length, opt.word_embedding_model_name)
    logger.info('train word embedding has finished. ')

    vocab_embedding, out_of_vocab_count, out_of_vocab_words = build_vocab_word2vec(
        word2vec_model,
        vocab,
        None,
        opt.size,
        None,
        os.path.join(opt.save_path, opt.word_embedding_model_name +
                     '.%d.300d.txt' % vocab_size),
        logger)

    # save out of vocab words
    save_out_of_vocab_words(out_of_vocab_words, 'out_of_vocab_words_vocabVec.txt')

    np.save(os.path.join(opt.save_path, opt.word_embedding_model_name + '.%d.300d.npy' % vocab_size),
            vocab_embedding)

    logger.info('build_vocab_word2vec() finished. out_of_vocab_count: %d' %
                out_of_vocab_count)  #

    logger.info('Preprocess finished.')

