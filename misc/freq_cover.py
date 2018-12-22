#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.


import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()


parser.add_argument('--vocab', action='store_true', help='')
parser.add_argument('--dist', type=str, help='')
parser.add_argument('--num', type=int, help='')

args = parser.parse_args()


def cover():
    total_value = 0
    cover_value = 0
    with open(args.dist) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if args.vocab:
                _, value = line.split()
                value = int(value)
                total_value += value
                if i < args.num:
                    cover_value += value
            else:
                key, value = map(lambda x: int(x), line.split())
                total_value += value
                if key <= args.num:
                    cover_value += value

    cover_ratio = cover_value / total_value
    print('cover ratio: %.5f' % cover_ratio)

if __name__ == '__main__':
    cover()
