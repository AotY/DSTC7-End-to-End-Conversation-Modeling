# -*- coding: utf-8 -*-

''''
Tokenize a reddit sequence.
'''
import re
#  from tokenizer import tokenizer
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup

def is_english(text):
    try:
        _ = text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

class Tokenizer:
    def __init__(self):
        # URLs
        #  url_regex_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'

        number_regex_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'  # numbers

        self.number_re = re.compile(number_regex_str, re.VERBOSE | re.IGNORECASE)

    ''' replace number by NUMBER_TAG'''
    def replace_number(self, tokens):
        return ['<number>' if self.number_re.search(token) else token for token in tokens]

    def tokenize(self, text):
        if isinstance(text, list):
            text = ' '.join(text)

        if not is_english(text):
            return []

        text = BeautifulSoup(text, "lxml").text

        tokens = self.clean_str(text).split()

        tokens = self.replace_number(tokens)
        tokens = [token for token in tokens if len(token.split()) > 0]

        # clip max len of token
        tokens = [token for token in tokens if len(token) < 16]
        return tokens

    def clean_str(self, text):
        text = text.lower()
        text = re.sub('^',' ', text)
        text = re.sub('$',' ', text)

        # url
        words = []
        for word in text.split():
            i = -1
            if word.find('http') != -1:
                i = word.find('http')
            elif word.find('www') != -1:
                i = word.find('www')
            elif word.find('.com') != -1:
                i = word.find('.com')

            if i >= 0:
                word = word[:i] + ' ' + '<url>'
            words.append(word.strip())
        text = ' '.join(words)

        # remove markdown url
        text = re.sub(r'\[([^\]]*)\] \( *<url> *\)', r'\1', text)

        # remove illegal char
        text = re.sub('<url>', 'url', text)
        text = re.sub(r"[^a-za-z0-9():,.!?\"\']", " ", text)
        text = re.sub('url', '<url>',text)

        # contraction
        add_space = ["'s", "'m", "'re", "n't", "'ll","'ve","'d","'em"]
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)
        text = ' ' + ' '.join(tokenizer.tokenize(text)) + ' '
        text = text.replace(" won't ", " will n't ")
        text = text.replace(" can't ", " can n't ")
        for a in add_space:
            text = text.replace(a+' ', ' '+a+' ')

        text = re.sub(r'^\s+', '', text)
        text = re.sub(r'\s+$', '', text)
        text = re.sub(r'\s+', ' ', text) # remove extra spaces

        return text

if __name__ == '__main__':
    sequence = 'RT @marcobonzanini: just an example! :D http://example.com #NLP <title> 223: 113'
    tokenizer = Tokenizer()
    print(tokenizer.tokenize(sequence))
    # ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']

