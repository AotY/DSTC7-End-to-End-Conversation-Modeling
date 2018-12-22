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

# create keys
keys_path = 'tmp.keys'
keys = open(keys_path, 'w')
with open(args.submission, 'r') as f:
    for line in f:
        line = line.rstrip()
        hash_value = line.split('\t')[0]
        keys.write('%s\n' % hash_value)

keys.close()

os.system('python evaluation/src/dstc.py %s --refs %s --keys %s' % \
          (args.submission, args.refs, keys_path))

#  os.system('rm %s' % keys_path)
