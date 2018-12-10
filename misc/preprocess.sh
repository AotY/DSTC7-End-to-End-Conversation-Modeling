#!/usr/bin/env bash

python preprocess.py \
    --convos_file_path ./../data/raw.convos.txt \
    --facts_file_path ./../data/raw.facts.txt \
    --save_path ./../data \
    --min_len 3 \
    --c_max_len 200 \
    --q_max_len 250 \
    --r_max_len 120 \
    --f_max_len 350
/
