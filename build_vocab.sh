#! /bin/sh
#
# build_vocab.sh
# Copyright (C) 2018 taoqing <taoqing@gpu3>
#
# Distributed under terms of the MIT license.
#


python misc/build_vocab.py \
    --freq_path ./data/word.3.freq.txt \
    --vocab_size 6e4 \
    --min_count 3 \
    --vocab_path ./data/vocab_word2idx.3.40004.dict
 
