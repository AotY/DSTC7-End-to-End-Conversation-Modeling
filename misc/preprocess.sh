#!/usr/bin/env bash

python preprocess.py \
    --convos_file_path ./../data/train.convos.txt \
    --facts_file_path ./../data/train.facts.txt \
    --save_path ./../data \
    --min_len 3 \
    --c_max_len 500 \
    --q_max_len 220 \
    --r_max_len 150 \
    --f_max_len 350
/
