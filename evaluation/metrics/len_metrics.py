#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
stats length.
"""

import re
import string
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--predicted', help="predicted text file, one example per line")

args = parser.parse_args()


def main():
    lens = []
    with open(args.predicted, 'r') as f:
        for line in tqdm(f):
            line = line.rstrip()
            lens.append(len(line.split()))

    min_len = min(lens)
    max_len = max(lens)
    sum_len = sum(lens)
    avg_len = sum_len / len(lens)

    print('len stats: %s' % args.predicted)
    print('min_len: %d' % min_len)
    print('max_len: %d' % max_len)
    print('avg_len: %.3f' % avg_len)

if __name__ == '__main__':
    main()




