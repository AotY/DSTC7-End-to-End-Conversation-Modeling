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

class KeywordExtract:
    def __init__(self):
        self.rake = Rake(stopwords=stopwords.words())
        self.textrank = keywords

    def keywords_rake(self, text):
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()
    
    def keywords_textrank(self, text, ratio=0.5, split=True):
        return self.textrank.keywords(text, ratio=ratio, split=split)




