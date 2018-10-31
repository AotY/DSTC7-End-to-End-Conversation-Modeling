#!/usr/bin/env bash

python preprocess.py \
    --convos_file_path ./../data/train.convos.txt \
    --facts_file_path ./../data/train.facts.txt \
    --save_path ./../data \
    --vocab_path ./../data/vocab_word2idx_{}.dict \
    --c_max_len 210 \
    --c_min_len 7 \
    --r_max_len 50 \
    --r_min_len 3 \
    --f_max_len 220\
    --f_min_len 3 \
    --min_count 4 \
    --max_vocab_size 5e4 \
    --word_embedding_model_name v1.0_word_embedding \
    --google_vec_file /home/taoqing/Research/data/GoogleNews-vectors-negative300.bin \
    --google_vec_dim 300 \
    --fasttext_vec_file /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \
    --fasttext_vec_dim 300 \
    --binary \
    --window 7 \
    --alpha 0.025 \
    --negative 7 \
    --epochs 7 \
    --model_name seq2seq

/
