#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.


import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--distribution', type=str, help='')
parser.add_argument('--max_num', type=int, help='')

args = parser.parse_args()


def cover():
    total_value = 0
    cover_value = 0
    with open(args.distribution) as f:
        for line in f:
            line = line.rstrip()
            key, value = map(lambda x: int(x), line.split())

            total_value += value
            if key <= args.max_num:
                cover_value += value

    cover_ratio = cover_value / total_value
    print('cover ratio: %.3f' % cover_ratio)

if __name__ == '__main__':
    cover()



