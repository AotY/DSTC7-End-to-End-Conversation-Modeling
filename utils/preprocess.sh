#!/usr/bin/env bash

python preprocess.py \
    -convos_file_path ./../data/train.convos.txt \
    -facts_file_path ./../data/train.facts.txt \
    -conversations_num_save_path ./../data/train.convos.num.txt \
    -responses_num_save_path ./../data/train.facts.num.txt \
    -min_count 3 \
    -max_vocab_size 2e5 \



/
