#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--second', type=int, help='')
parser.add_argument('--file', type=str, help='', default=6)


args = parser.parse_args()

time.sleep(args.second)

os.system(args.file)
