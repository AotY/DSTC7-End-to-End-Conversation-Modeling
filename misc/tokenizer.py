# -*- coding: utf-8 -*-

''''
Tokenise a reddit sequence.
'''

import re


class Tokenizer:
    def __init__(self):
        emoticons_str = r"""
                    (?:
                        [:=;] # Eyes
                        [oO\-]? # Nose (optional)
                        [D\)\]\(\]/\\OpP] # Mouth
                    )"""

        regex_str = [
            emoticons_str,
            r'<[^>]+>',  # HTML tags
            r'(?:@[\w_]+)',  # @-mentions
            r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
            # URLs
            r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',

            r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
            r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
            r'(?:[\w_]+)',  # other words
            r'(?:\S)'  # anything else
        ]

        # URLs
        url_regex_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'

        self.url_re = re.compile(url_regex_str, re.VERBOSE | re.IGNORECASE)

        self.tokens_re = re.compile(
            r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
        self.emoticon_re = re.compile(
            r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

    ''' replace url by URL_TAG'''

    def replace_url(self, tokens):
        return ['URL_TAG' if self.url_re.search(token) else token for token in tokens]

    def tokenize(self, s):
        return self.tokens_re.findall(s)

    def preprocess(self, s, lowercase=False):
        tokens = self.tokenize(s)
        if lowercase:
            tokens = [token if self.emoticon_re.search(
                token) else token.lower() for token in tokens]
        return tokens


if __name__ == '__main__':
    sequence = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
    tokenize = Tokenizer()
    print(tokenize.preprocess(sequence, lowercase=True))
    # ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']
