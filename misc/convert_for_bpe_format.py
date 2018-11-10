#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('Covert for training bpe.')

parser.add_argument('--type', type=str, help='seq2seq or kg')

args = parser.parse_args()


def convert_format(pair_path, fact_path):
    converted_file = open('./../data/bpe_train.txt', 'w', encoding='utf-8')

    with open(pair_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            _, conversation, response, hash_value = line.split('SPLITTOKEN')

            # skip if source has nothing
            if conversation == 'START' or len(conversation.rstrip()) == 0:
                continue

            if conversation.startswith('start eos'):
                conversation = conversation[10:]
            elif conversation.startswith('eos'):
                conversation = conversation[4:]

            conversation_turns = conversation.split('eos')

            for conversation_turn in conversation_turns:
                converted_file.write(conversation_turn + '\n')

            converted_file.write(response + '\n')

    if fact_path is not None:
        with open(fact_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.rstrip()

                _, _, _, fact = line.split('\t')

                converted_file.write(fact + '\n')

    converted_file.close()

if __name__ == '__main__':
    pair_path = './../data/conversations_responses.pair.txt'
    fact_path = None
    if args.type == 'kg':
        fact_path = './../data/facts.txt'

    convert_format(pair_path, fact_path)

