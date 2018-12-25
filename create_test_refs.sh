#!/usr/bin/env bash
#
# create_test_refs.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#


python create_test_refs.py \
    --convos_path data/cleaned.3.convos.txt \
    --test_refs_txt_path data_extraction/test.refs.txt \
    --test_refs_path data_extraction/test.refs

/
