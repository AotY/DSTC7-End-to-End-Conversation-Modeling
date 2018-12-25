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
parser.add_argument('--keys_2k', type=str, help='')
parser.add_argument('--n_lines', type=int, help='', default=-1)

args = parser.parse_args()

# 2k keys
hash_values_2k = set()
with open(args.keys_2k, 'r') as f:
    for line in f:
        line = line.rstip()
        hash_values_2k.add(line.split()[0])

eval_hash_values = []

tmp_refs_path = '.tmp.refs'
tmp_refs_file = open(tmp_refs_path, 'w')
with open(args.refs, 'r') as f:
    for line in f:
        line = line.rstrip()
        parts = line.split('\t')
        hash_value = parts[0]
        if len(parts) > 1:
            if hash_value in hash_values_2k:
                eval_hash_values.append(hash_value)
                tmp_refs_file.write('%s\n' % line)
tmp_refs_file.close()

# tmp submission
tmp_submission_path = '.tmp.submission'
tmp_submission_file = open(tmp_submission_path, 'w')
with open(args.submission, 'r') as f:
    for line in f:
        line = line.rstrip()
        parts = line.split('\t')
        hash_value = parts[0]
        if hash_value in eval_hash_values:
            tmp_submission_file.write('%s\n' % line)
tmp_submission_file.close()


# save keys
tmp_keys_path = '.tmp.keys'
tmp_keys_file = open(tmp_keys_path, 'w')
for hash_value in eval_hash_values:
    tmp_keys_file.write('%s\n' % hash_value)

tmp_keys_file.close()

os.system('python evaluation/src/dstc.py %s --refs %s --keys %s --n_lines %s' % \
          (tmp_submission_path, tmp_refs_path, tmp_keys_path, args.n_lines))

time.sleep(1)

os.system('rm %s' % tmp_refs_path)
os.system('rm %s' % tmp_submission_path)
os.system('rm %s' % tmp_keys_path)
