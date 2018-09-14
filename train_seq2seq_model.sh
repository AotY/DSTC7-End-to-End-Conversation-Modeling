#!/usr/bin/env bash

python train_seq2seq_model.py \
    --convos_file_path ./../data/train.convos.txt \
    --facts_file_path ./../data/train.facts.txt \
    --conversations_num_save_path ./../data/train.convos.num.txt \
    --responses_num_save_path ./../data/train.facts.num.txt \

    --save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data \
    --min_count 3 \
    --max_vocab_size 8e4 \

    --binary \
    --max_words 80 \
    --size 512 \
    --window 7 \
    --alpha 0.025 \
    --negative 7 \
    --epochs 5 \
    --lower \

/