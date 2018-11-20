#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Extract Keyword From text.
1. TextRank
2. Rapid Automatic Keyword Extraction (RAKE)
3.
"""
from nltk.corpus import stopwords
from rake_nltk import Rake
from summa import keywords
import pickle

class KeywordExtract:
    def __init__(self):
        self.rake = Rake(stopwords=stopwords.words())
        self.textrank = keywords

    def keywords_rake(self, text):
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()

    def keywords_textrank(self, text, ratio=0.5, split=True):
        return self.textrank.keywords(text, ratio=ratio, split=split)



def build_facts_p_keywords(path, topk=25):
    r = Rake()
    facts_p_dict = pickle.load(open(path, 'wb'))
    facts_topk_phrases = {}
    for conversation_id, ps in facts_p_dict.items():
        r.extract_keywords_from_sentences(ps)
        phrases = r.get_ranked_phrases()[:topk]

        facts_topk_phrases[conversation_id] = phrases

    pickle.dump(facts_topk_phrases, open('./../data/facts_topk_phrases.pkl', 'wb'))


if __name__ == '__name__':
    facts_p_dict_path = './../data/facts_p_dict.pkl'

