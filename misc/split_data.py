#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Split data to four parts: train, dev, valid, test
"""


import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--convos_path', type=str, default='./data/cleaned.convos.txt')
parser.add_argument('--facts_path', type=str, default='./data/cleaned.facts.txt')

# save
parser.add_argument('--train.convos.txt', type=str, default='./data/train.facts.txt')
parser.add_argument('--dev.convos.txt', type=str, default='./data/train.facts.txt')
args = parser.parse_args()




