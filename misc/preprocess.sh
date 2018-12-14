#!/usr/bin/env bash

python preprocess.py \
    --convos_file_path ./../data/raw.convos.txt \
    --facts_file_path ./../data/raw.facts.txt \
    --convos_save_path ./../data/train.pseudo.convos.txt \
    --facts_save_path ./../data/train.pseudo.facts.txt \
    --save_path ./../data \
    --min_len 3 \
    --c_max_len 220 \
    --q_max_len 270 \
    --r_max_len 150 \
    --f_max_len 500
/
