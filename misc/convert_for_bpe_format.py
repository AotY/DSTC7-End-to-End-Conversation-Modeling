#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('Covert for training bpe.')

parser.add_argument('--pair_path', type=str, help='pair path')
parser.add_argument('--fact_path', type=str, help='facts path')
parser.add_argument('--type', type=str, help='seq2seq or kg')
parser.add_argument('--save_path', type=str, help='path to save converted file.')

args = parser.parse_args()

def convert_format():
    converted_file = open(args.save_path, 'w', encoding='utf-8')

    with open(args.pair_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.rstrip()
            _, context, response, hash_value, _, _ = line.split('\t')

            # skip if source has nothing
            if context == 'start' or len(context.rstrip()) == 0:
                continue

            if context.startswith('start eos'):
                context = context[10:]
            elif context.startswith('eos'):
                context = context[4:]
            elif context.startswith('... eos'):
                context = context[7:]
            elif context.startswith('...'):
                context = context[4:]

            sentences = context.split('eos')
            for sentence in sentences:
                converted_file.write(sentence + '\n')

            converted_file.write(response + '\n')

    if args.fact_path is not None:
        with open(args.fact_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.rstrip()
                _, _, _, fact = line.split('\t')
                converted_file.write(fact + '\n')
    converted_file.close()

if __name__ == '__main__':
    convert_format()
