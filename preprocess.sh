#!/usr/bin/env bash

python misc/preprocess.py \
    --raw_convos_path ./data/raw.convos.txt \
    --raw_facts_path ./data/raw.facts.txt \
    --train_convos_path ./data/train.convos.txt \
    --train_facts_path ./data/train.facts.txt \
    --freq_save_path ./data/word.freq.txt \
    --save_path ./data \
    --min_len 3 \
    --c_max_len 220 \
    --q_max_len 270 \
    --r_max_len 150 \
    --f_max_len 500
/
