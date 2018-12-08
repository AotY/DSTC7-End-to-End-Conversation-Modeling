# -*- coding: utf-8 -*-

''''
Tokenize a reddit sequence.
'''
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup
import warnings


def is_english(text):
    try:
        _ = text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)

def remove_stop_words(words):
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in punctuations]
    return words


class Tokenizer:
    def __init__(self):
        # URLs
        url_regex_str = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'

        number_regex_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'  # numbers

        self.url_re = re.compile(url_regex_str, re.VERBOSE | re.IGNORECASE)

        self.number_re = re.compile(
            number_regex_str, re.VERBOSE | re.IGNORECASE)

    def tokenize(self, text, html=False):
        if isinstance(text, list):
            text = ' '.join(text)

        if not is_english(text):
            return []

        if html:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                text = BeautifulSoup(text, "lxml").text

        tokens = self.clean_str(text).split()

        tokens = [token for token in tokens if len(token.split()) > 0]

        return tokens

    def clean_str(self, text):
        text = text.lower()
        text = re.sub('^', ' ', text)
        text = re.sub('$', ' ', text)

        # url
        text = self.url_re.sub('__url__', text)

        #  words = []
        #  for word in text.split():
            #  i = word.find('http')
            #  if i >= 0:
                #  word = word[:i] + ' ' + '__url__'
            #  words.append(word.strip())
        #  text = ' '.join(words)

        # remove markdown URL
        text = re.sub(r'\[([^\]]*)\] \( *__url__ *\)', r'\1', text)

        # remove illegal char
        text = re.sub('__url__', 'URL', text)
        text = re.sub(r"[^A-Za-z0-9():,.!?\"\']", " ", text)
        text = re.sub('URL', '__url__', text)

        # number
        text = self.number_re.sub('__number__', text)

        # merge multi to single
        text = text.replace('__number__ __number__', '')
        text = text.replace('__url__ __url__', '')
        text = text.replace('__number__ __url__', '')
        text = text.replace('__url__ __number__', '')

        text = re.sub(r'\(\s__url__\s\)', '__url__', text)
        text = re.sub(r'(\s__url__)+', ' __url__', text)
        text = re.sub(r'(__url__\s)+', '__url__ ', text)
        text = re.sub(r'(__url__)+', '__url__', text)

        text = re.sub(r'\(\s__number__\s\)', '__number__', text)
        text = re.sub(r'(\s__number__\s)+', ' __number__ ', text)
        text = re.sub(r'(\s__number__)+', ' __number__', text)
        text = re.sub(r'(__number__\s)+', '__number__ ', text)

        text = text.replace('__number__ __number__', '')
        text = text.replace('__url__ __url__', '')
        text = text.replace('__number__ __url__', '')
        text = text.replace('__url__ __number__', '')
        text = text.replace('__number__ __number__', '')
        text = text.replace('__url__ __url__', '')
        text = text.replace('__number__ __number__', '')
        text = text.replace('__url__ __url__', ' ')
        text = text.replace('__number__ __number__', '')
        text = text.replace('__number__ __number__', '')

        text = text.replace('.com', ' ')
        text = text.replace('.org', ' ')
        text = text.replace('.net', ' ')
        text = text.replace('.gov', ' ')
        text = text.replace('.edu', ' ')

        # contraction
        add_space = ["'s", "'m", "'re", "n't", "'ll", "'ve", "'d", "'em"]
        tweet_tokenizer = TweetTokenizer(
            preserve_case=False, strip_handles=False, reduce_len=False)
        text = ' ' + ' '.join(tweet_tokenizer.tokenize(text)) + ' '
        text = text.replace(" won't ", " will n't ")
        text = text.replace(" can't ", " can n't ")
        for a in add_space:
            text = text.replace(a+' ', ' '+a+' ')

        text = re.sub(r'^\s+', '', text)
        text = re.sub(r'\s+$', '', text)
        text = re.sub(r'\s+', ' ', text)  # remove extra spaces

        return text


if __name__ == '__main__':
    sequence = 'RT @marcobonzanini: just an example! 342 23424 '\
        ' trio.com www.trio.com :D https://www.youtube.com/watch?v=gGRyC8fjTUM http://example.com #NLP <title> 223: 113'
    tokenizer = Tokenizer()
    print(sequence)
    print(tokenizer.tokenize(sequence))
    # ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']
