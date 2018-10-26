#!/usr/bin/env bash

python preprocess.py \
    --convos_file_path ./../data/train.convos.txt \
    --facts_file_path ./../data/train.facts.txt \
    --save_path ./../data \
    --vocab_save_path ./../data/vocab_word2idx_{}.dict \
    --min_count 4 \
    --max_vocab_size 9e4 \
    --word_embedding_model_name v1.0_word_embedding \
    --google_vec_file /home/taoqing/Research/data/GoogleNews-vectors-negative300.bin \
    --google_vec_dim 300 \
    --fasttext_vec_file /home/taoqing/Research/data/crawl-300d-2M-subword.vec \
    --fasttext_vec_dim 300 \
    --binary \
    --window 7 \
    --alpha 0.025 \
    --negative 7 \
    --epochs 7 \
    --model_name kg

/
