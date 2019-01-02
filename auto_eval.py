#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Automatic evaluation.

"""

import os
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--submission', type=str, help='')
parser.add_argument('--refs', type=str, help='')
parser.add_argument('--n_refs', type=str, help='', default=6)
parser.add_argument('--keys_2k', type=str, help='')
parser.add_argument('--n_lines', type=int, help='', default=-1)

args = parser.parse_args()

# 2k keys
hash_values_2k = set()
with open(args.keys_2k, 'r') as f:
    for line in f:
        line = line.rstrip()
        hash_values_2k.add(line.split()[0])

final_hash_values = set()

tmp_refs_path = 'tmp_refs.txt'
tmp_refs_file = open(tmp_refs_path, 'w')
with open(args.refs, 'r') as f:
    for line in f:
        line = line.rstrip()
        parts = line.split('\t')
        hash_value = parts[0]
        if len(parts) > (args.n_refs - 1):
            if hash_value in hash_values_2k or len(parts) >= (args.n_refs + 2):
                final_hash_values.add(hash_value)
                tmp_refs_file.write('%s\n' % line)
tmp_refs_file.close()

final_hash_values = list(final_hash_values)

# tmp submission
tmp_submission_path = 'tmp_submission.txt'
tmp_submission_file = open(tmp_submission_path, 'w')
with open(args.submission, 'r') as f:
    for line in f:
        line = line.rstrip()
        parts = line.split('\t')
        hash_value = parts[0]
        if hash_value in final_hash_values:
            tmp_submission_file.write('%s\n' % line)
tmp_submission_file.close()

# save keys
tmp_keys_path = 'tmp_keys.txt'
tmp_keys_file = open(tmp_keys_path, 'w')
for hash_value in final_hash_values:
    tmp_keys_file.write('%s\n' % hash_value)

tmp_keys_file.close()

print('submission: %s' % args.submission)

os.system('python evaluation/src/dstc.py %s --refs %s --keys %s --n_lines %s' % \
          (tmp_submission_path, tmp_refs_path, tmp_keys_path, args.n_lines))

time.sleep(1)

os.system('rm -rf tmp_*')
