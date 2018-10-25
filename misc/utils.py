# -*- coding: utf-8 -*-

''''
Tokenise a reddit sequence.
'''

import re
from tokenizer import tokenizer


class Tokenizer:
    def __init__(self):
        # URLs
        url_regex_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'

        number_regex_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'  # numbers

        self.number_re = re.compile(number_regex_str, re.VERBOSE | re.IGNORECASE)

        self.url_re = re.compile(url_regex_str, re.VERBOSE | re.IGNORECASE)
        self.R = tokenizer.RedditTokenizer(preserve_case=False,
                                           preserve_handles=True,
                                           preserve_hashes=True,
                                           regularize=True,
                                           preserve_emoji=True,
                                           preserve_url=True)
        self.split_hyphen_str = r"([-|_|]+)"

        self.split_hyphen_re = re.compile(self.split_hyphen_str, re.VERBOSE | re.IGNORECASE)

    ''' replace url by URL_TAG'''
    def replace_url(self, tokens):
        return ['URL' if self.url_re.search(token) else token for token in tokens]

    ''' replace number by NUMBER_TAG'''
    def replace_number(self, tokens):
        return ['NUMBER' if self.number_re.search(token) else token for token in tokens]

    def split_hyphen(self, tokens):
        new_tokens = []
        for token in tokens:
            splits = set(self.split_hyphen_re.findall(token))
            if len(splits) > 0:
                for split in splits:
                    new_tokens += [item for item in token.split(split) if len(item) > 0]
            else:
                new_tokens.append(token)
        return new_tokens

    def split_quotation(self, tokens):
        new_tokens = []
        for token in tokens:
            if token.find("'") != -1:
                splits = token.split(r"'")
                for i, split in enumerate(splits):
                    if i != 0:
                        new_tokens.append("'" + split)
                    else:
                        new_tokens.append(split)
            else:
                new_tokens.append(token)
        return new_tokens


    def tokenize(self, text):
        tokens = self.R.tokenize(text)
        tokens = self.replace_url(tokens)
        tokens = self.replace_number(tokens)
        tokens = self.split_hyphen(tokens)
        tokens = self.split_quotation(tokens)
        # remove by max len
        tokens = [token for token in tokens if len(token) <= 20]
        return tokens


if __name__ == '__main__':
    sequence = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
    tokenize = Tokenizer()
    print(tokenize.preprocess(sequence, lowercase=True))
    # ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']


"""
    emoticons_str = r'''
                (?:
                    [:=;] # Eyes
                    [oO\-]? # Nose (optional)
                    [D\)\]\(\]/\\OpP] # Mouth
                )'''

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

    #  hyphen_regex_str = r"(?:[a-z][a-z'\-_]+[a-z])"  # words with - and '
    #  split_hyphen_str = r"(-|_|'\w+)" # - _ '
    #  self.split_hyphen_re = re.compile(split_hyphen_str, re.VERBOSE | re.IGNORECASE)
    #  self.hyphen_re = re.compile(hyphen_regex_str, re.VERBOSE | re.IGNORECASE)
    #  self.tokens_re = re.compile(
        #  r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    #  self.emoticon_re = re.compile(
        #  r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

    '''split by hyphen'''
    def split_by_hyphen(self, tokens):
        new_tokens = []
        for token in tokens:
            if self.hyphen_re.search(token):
                new_tokens += [t for t in re.split(self.split_hyphen_re, token) if len(t) > 0]
            else:
                new_tokens.append(token)

        return new_tokens

    ''' remove by length '''
    def remove_by_len(self, tokens, max_len=15):
        return [token for token in tokens if len(token) < max_len]

    def tokenize(self, s):
        return self.tokens_re.findall(s)

    def preprocess(self, s, lowercase=False):
        tokens = self.tokenize(s)
        if lowercase:
            tokens = [token if self.emoticon_re.search(
                token) else token.lower() for token in tokens]

"""
