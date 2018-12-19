#!/usr/bin/env bash
#
# run_embedding_metrics.sh
# Copyright (C) 2018 LeonTao #
# Distributed under terms of the MIT license.
#


python evaluation/metrics/embedding_metrics.py \
    --embeddings ~/Research/data/GoogleNews-vectors-negative300.bin \
    --ground_truth ./data/ground_truth/qc_1_3.txt \
    --predicted ./data/predicted/seq2seq_qc_6_1_3_2018_12_19_10:28.txt \

/
