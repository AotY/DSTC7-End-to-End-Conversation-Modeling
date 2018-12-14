#!/usr/bin/env bash
#
# run_embedding_metrics.sh
# Copyright (C) 2018 LeonTao #
# Distributed under terms of the MIT license.
#


python metrics/embedding_metrics.py \
    --embeddings ~/Research/data/GoogleNews-vectors-negative300.bin \
    --ground_truth ../data/ground_truth/eval_1_3.txt \
    --predicted ../data/predicted/seq2seq_qc_3_1_3_2018_12_14_17:26.txt \

/
