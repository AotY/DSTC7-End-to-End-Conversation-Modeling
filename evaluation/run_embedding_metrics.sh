#!/usr/bin/env bash
#
# run_embedding_metrics.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


python metrics/embedding_metrics.py \
    --embeddings ~/Research/data/GoogleNews-vectors-negative300.bin \
    --ground_truth ../data/ground_truth/1_4.txt \
    --predicted ../data/predicted/none_3_1_4.txt \

/
