#!/usr/bin/env bash

python merge_convos_facts.py \
    -convos_facts_folder_list /home/taoqing/Research/DSTC7/reddit/data-official-2011 /home/taoqing/Research/DSTC7/reddit/data-official-2012-13  \
    -save_convos_path ./../data/train.convos.txt \
    -save_facts_path ./../data/train.facts.txt \

/
