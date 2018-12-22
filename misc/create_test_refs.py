#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
1. Split test.refs.txt from cleaned.convos.txt
2. create test.refs

"""

import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--convos_path', type=str, help='')
parser.add_argument('--test_refs_txt_path', type=str, help='')
parser.add_argument('--test_refs_path', type=str, help='')

args = parser.parse_args()

# 1.
test_refs_txt = open(args.test_refs_txt_path, 'w')
with open(args.convos_path) as f:
    for line in tqdm(f):
        line = line.rstrip()

        data_type, subreddit_name, conversation_id, context, query, \
            response, hash_value, score, turn = line.split(' SPLIT ')

        if data_type == 'REFS':
            test_refs_txt.write('%s\n' % line)

test_refs_txt.close()

# 2.
os.system('cat %s | python data_extraction/src/ids2refs.py data_extraction/lists/test-multiref.sets > %s' %
          (args.test_refs_txt_path, args.test_refs_path))

