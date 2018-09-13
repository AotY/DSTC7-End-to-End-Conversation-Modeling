#!/usr/bin/env bash

python train_embedding.py \
    -corpus_path_list /home/Research/EnChat/askqa/train.post.txt /home/Research/EnChat/askqa/train.response.txt \
    -save_path /home/taoqing/Research/EnChat/Classification/data/reddit/askqa \
    -binary False \
    -max_words 60 \
    -size 256 \
    -window 7 \
    -alpha 0.025 \
    -min_count 5 \
    -negative 7 \
    -epochs 5 \
    -lower \



#python train_embedding.py \
#    -corpus_path_list /home/Research/EnChat/askqa/train.post.txt /home/Research/EnChat/askqa/train.response.txt \
#    -save_path /home/taoqing/Research/EnChat/Classification/data/reddit/askqa \
#    -binary False \
#    -max_words 60 \
#    -size 200 \
#    -window 7 \
#    -alpha 0.025 \
#    -min_count 7 \
#    -negative 7 \
#    -epochs 7 \

/