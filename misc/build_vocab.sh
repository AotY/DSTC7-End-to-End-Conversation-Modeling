#! /bin/sh
#
# build_vocab.sh
# Copyright (C) 2018 taoqing <taoqing@gpu3>
#
# Distributed under terms of the MIT license.
#


python build_vocab.py \
    --distribution ./conversations_responses_facts.freq.txt \
    --vocab_size 4e4 \
    --min_count 3 \
    --vocab_path ./../data/vocab_word2idx.40004.pkl \
 
/

