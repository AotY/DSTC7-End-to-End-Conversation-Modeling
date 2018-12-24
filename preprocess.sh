#!/usr/bin/env bash

python misc/preprocess.py \
    --raw_convos_path ./data/raw.3.convos.txt \
    --raw_facts_path ./data/raw.3.facts.txt \
    --train_convos_path ./data/cleaned.3.convos.txt \
    --train_facts_path ./data/cleaned.3.facts.txt \
    --save_path ./data \
    --min_len 3 \
    --c_max_len 210 \
    --q_max_len 280 \
    --r_max_len 160 \
    --f_max_len 500
/
