#!/usr/bin/env bash

python misc/preprocess.py \
    --raw_convos_path ./data/raw.2.convos.txt \
    --raw_facts_path ./data/raw.2.facts.txt \
    --train_convos_path ./data/train.2.convos.txt \
    --train_facts_path ./data/train.2.facts.txt \
    --freq_save_path ./data/word.2.freq.txt \
    --save_path ./data \
    --min_len 3 \
    --c_max_len 210 \
    --q_max_len 280 \
    --r_max_len 160 \
    --f_max_len 500 \
/
