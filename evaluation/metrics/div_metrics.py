#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Diversity metric
"""

import argparse
from tqdm import tqdm
from collections import defaultdict
from collections import Counter

parser = argparse.ArgumentParser()

parser.add_argument(
    '--predicted', help="predicted text file, one example per line")

args = parser.parse_args()


def calc_diversity():
    tokens = [0.0, 0.0]
    types = [defaultdict(int), defaultdict(int)]
    with open(args.predicted, 'r') as f:
        for line in tqdm(f):
            words = line.strip('\n').split()
            for n in range(2):
                for idx in range(len(words)-n):
                    ngram = ' '.join(words[idx:idx+n+1])
                    types[n][ngram] = 1
                    tokens[n] += 1
    div1 = len(types[0].keys())/tokens[0]
    div2 = len(types[1].keys())/tokens[1]
    return [div1, div2]


def main():
    sentence_dict = {}
    line_num = 0
    with open(args.predicted, 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            sentence_dict[line] = sentence_dict.get(line, 0) + 1

            line_num += 1

    diff_num = len(sentence_dict)
    sentence_list = sorted(sentence_dict.items(), key=lambda item: item[1], reverse=True)

    avg_diff_num = line_num / diff_num
    most_comm_sentence = sentence_list[0][0]
    print('avg_diff_num: %.4f' % avg_diff_num)
    print('most_comm_sentence: %s' % most_comm_sentence)

    div1, div2 = calc_diversity()
    print('div1: %.4f' % div1)
    print('div2: %.4f' % div2)


if __name__ == '__main__':
    main()
