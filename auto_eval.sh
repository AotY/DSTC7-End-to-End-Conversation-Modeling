#!/usr/bin/env bash
#
# auto_eval.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


python auto_eval.py \
    --submission data/submission/kg_qc_h_3_1_3_2019_01_04_19:23.txt \
    --refs data_extraction/test.refs \
    --keys_2k evaluation/src/keys/test.2k.txt \
    --n_lines -1 \

/
