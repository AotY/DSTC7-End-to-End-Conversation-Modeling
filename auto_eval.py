#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Automatic evaluation.

"""

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--submission', type=str, help='')
parser.add_argument('--refs', type=str, help='')
#  parser.add_argument('--keys', type=str, help='')

args = parser.parse_args()

tmp_refs_path = 'tmp.refs'
removal_keys = []
refs = open(tmp_refs_path, 'w')
with open(args.refs, 'r') as f:
    for line in f:
        line = line.rstrip()
        parts = line.split('\t')
        if len(parts) == 1:
            removal_keys.append(parts[0])
        else:
            refs.write('%s\n' % line)
refs.close()
print('removal_keys: ', removal_keys)

# create keys
tmp_keys_path = 'tmp.keys'
keys = open(tmp_keys_path, 'w')
with open(args.submission, 'r') as f:
    for line in f:
        line = line.rstrip()
        hash_value = line.split('\t')[0]
        if hash_value not in removal_keys:
            keys.write('%s\n' % hash_value)

keys.close()

os.system('python evaluation/src/dstc.py %s --refs %s --keys %s --n_lines %s' % \
          (args.submission, tmp_refs_path, tmp_keys_path, -1))

os.system('rm %s' % tmp_refs_path)
os.system('rm %s' % tmp_keys_path)
