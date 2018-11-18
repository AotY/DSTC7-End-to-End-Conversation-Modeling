#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
LDA model.
"""
from gensim import models

def build_model(texts):
    lda = LdaModel(texts, num_topics=50, alpha='auto', eval_every=5)  # learn asymmetric alpha from data

def get_topics(text):
    get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)



