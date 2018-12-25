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
from nltk import ngrams


def is_english(text):
    try:
        _ = text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


stop_words = set(stopwords.words('english'))
punctuations = list(string.punctuation)
punc_regex = re.compile('[%s]' % re.escape(string.punctuation.replace('_', '')))

def remove_stop_words(words):
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in punctuations]
    return words


class Tokenizer:
    def __init__(self):
        # URLs
        #  url_regex_str = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'
        url_regex_str = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+' # URLs

        #  number_regex_str = r'(?:(?:\d+,?)+(?:\.?\d+)?)'  # numbers

        #  self.url_re = re.compile(url_regex_str, re.VERBOSE | re.IGNORECASE)
        self.url_re = re.compile(url_regex_str)

        #  self.number_re = re.compile(
            #  number_regex_str, re.VERBOSE | re.IGNORECASE)

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
        #  tokens = [token for token in tokens if len(token) <= 25]

        return tokens

    def clean_repeat(self, text, max_ngram=5):
        tmp_text = punc_regex.sub('', text)
        text_ngrams = ngrams(tmp_text.split(), max_ngram)
        for ngram in text_ngrams:
            if len(set(ngram)) == 1:
                return ''

        """
        words = []
        for i, word in enumerate(text.split()):
            if i == 0:
                words.append(word)
            else:
                if word in punctuations:
                    words.append(word)
                else:
                    if word != words[len(words) - 1]:
                        words.append(word)
        text = ' '.join(words)
        """

        return text

    def clean_str(self, text):
        text = text.lower()
        text = re.sub('^', ' ', text)
        text = re.sub('$', ' ', text)

        # url
        text = self.url_re.sub('__url__', text)

        if text.count('__url__') > 6:
            return ''

        # remove markdown URL
        text = re.sub(r'\[([^\]]*)\] \( *__url__ *\)', r'\1', text)

        # remove illegal char
        text = re.sub('__url__', 'URL', text)
        text = re.sub(r"[^A-Za-z0-9():,.!?\"\']", " ", text)
        text = re.sub('URL', '__url__', text)

        # number
        text = text.replace(':', ' : ')
        text = text.replace(',', ' , ')
        words = []
        for word in text.split():
            try:
                float(word)
                words.append('__number__')
            except ValueError as e:
                words.append(word)
                continue
        #  print('words: ', words)
        #  text = self.number_re.sub('__number__', text)
        text = ' '.join(words)

        if text.count('__number__') > 12:
            return '' # merge multi to single

        text = self.clean_repeat(text)

        if text == '':
            return text

        text = text.replace('( __number__ )', '__number__')
        text = text.replace('( __url__  )', '__url__')

        text = re.sub(r'(\s?__number__\s?)+', ' __number__ ', text)
        text = re.sub(r'(\s?__url__\s?)+', ' __url__ ', text)

        text = re.sub(r'(\s?__url__ __number__\s?)+', ' __url__ ', text)
        text = re.sub(r'(\s?__number__ __url__\s?)+', ' __number__ ', text)

        text = re.sub(r'__number__\w+', '__number__', text)
        text = re.sub(r'\w+__number__', '__number__', text)
        text = re.sub(r'__url__\w+', '__url__', text)
        text = re.sub(r'\w+__url__', '__url__', text)

        text = re.sub(r'__number__ \S __number__', '__number__', text)
        text = re.sub(r'__url__ \S __url__', '__url__', text)

        text = re.sub(r'__url__ \S __number__', '__url__', text)
        text = re.sub(r'__number__ \S __url__', '__number__', text)

        text = text.replace('__number __', ' __number__ ')
        text = text.replace('__url __', ' __url__ ')

        text = text.replace('__number__ __url__', '__number__')
        text = text.replace('__url__ __number__', '__url__')
        text = text.replace('( __number__ )', '__number__')
        text = text.replace('( __url__ )', '__url__')

        # removal duplicate words
        text_words = text.split()
        words = [text_words[0]]

        for word in text_words[1:]:
            if word in ['__number__', '__url__']:
                if words[len(words) - 1] in ['__number__', '__url__']:
                    continue
            words.append(word)

        text = ' '.join(words)

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

        #  text = text.replace('"', '')
        #  text = text.replace('^', '')
        text = text.replace('( )', '')
        text = text.replace('[ ]', '')
        text = text.replace('{ }', '')
        text = text.replace("' '", '')
        text = text.replace('" "', '')
        text = text.replace('. .', '.')
        text = text.replace(', ,', ',')
        text = text.replace(': :', ':')
        text = text.replace('; ;', ';')

        return text


if __name__ == '__main__':
    #  sequence = 'RT @marcobonzanini: just an example! 342 23424 '\
        #  ' trio.com www.trio.com :D https://www.youtube.com/watch?v=gGRyC8fjTUM http://example.com #NLP <title> 223: 113'
    sequence = """
     ... 100,3.32 many "fsd" of the characters cited in the lawsuit . EOS as someone who 435:23 did a lot of pvp in wow and remembers people like that , fuck that guy for bringing them into pvp . it's one thing to multibox pve , but it's straight up cheating in pvp         . EOS it's so unfair that 10 pcs attacking a single pc would win 100 % of the time . coordination in pvp is cheating ! EOS but it's not coordination , it's ten char        acters being controlled simultaneously with each button press . there's a huge difference . EOS yeah , there is a difference . it's considerably easier to ruin the o        perator's macros and rotations by sending the characters scrambling with ccs and fears making it incredibly difficult , if not impossible , for him to regroup before         he's killed off . not everyone is smart enough to do that though . here's the real question though : would you single handedly try to take on and kill 10 individual         players ? if not , why complain about 1 player controlling 10 characters ? you would still lose . EOS you never actually played against these people did you ?   yes         , on many occasions over the 11 years i played the game , a good portion of which i spent inside bgs and arenas . if you ever had a problem with them it's because y        ou weren't playing intelligently .
     Great infographic about lots of things F1. https://www.reddit.com/r/formula1/comments/ignfy/ hello 432

     ... there just in case ( i didn't r        eally count in atlantic city as well - it was more of me explaining how it worked , so i was being obvious , but not making any money  )         . EOS macau ( also hong kong  ) is a [ special administrative region  ] ( http://en.wikipedia.org/wiki/Special_administrative_region_\(P        eople%27s_Republic_of_China\)  ) within china . which lets it operate very differently from the rest of china . EOS is taiwan the same d        eal , or a completely different country ? i've never quite understood its ties to china ... EOS it is independent . it was never really         part of china . at times it was administered by china , other times it was a pirate island . japan won it in a war , and after wwii ch        ina got it back . chiang kai-shek , the nationalist anti-communist leader of china , fled there after he knew the communist would win t        he mainland . he brought a lot of supporters with him . for awhile it was a dictatorship . it was one of the asian tigers , and experie        nced rapid economic growth . china wants it . think of taiwan as a democratic china .   * de facto * independent ; * de jure * part of         china ( both sides actually agree on this , with taiwan claiming to be the rightful government of all china .  )

    """
    tokenizer = Tokenizer()
    print(sequence)
    print(tokenizer.tokenize(sequence))
    # ['RT', '@marcobonzanini', ':', 'just', 'an', 'example', '!', ':D', 'http://example.com', '#NLP']
