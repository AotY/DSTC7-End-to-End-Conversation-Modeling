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

args = parser.parse_args()


os.system('python evaluation/src/dstc.py -c %s --refs %s' % (args.submission, args.refs))
