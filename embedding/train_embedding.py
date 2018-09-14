# -*- coding: utf-8 -*-

'''
train word embedding using gensim
data from reddit
'''

from __future__ import division
from __future__ import print_function

import multiprocessing

import os
from gensim.models.word2vec import Word2Vec, LineSentence


# a memory-friendly iterator
class MemoryOptimalSentences(object):
    def __init__(self, corpus_path_list, max_words, lower, logger):
        self.corpus_path_list = corpus_path_list
        self.max_words = max_words
        self.lower = lower
        self.logger = logger

    def __iter__(self):
        for corpus_path in self.corpus_path_list:
            # check there exists the corpus_path or not.

            self.logger.info("corpus_path: %s .", corpus_path)

            with open(corpus_path, "r", encoding='utf-8') as f:
                for line in f:
                    words = line.strip().replace('\t', ' ').split()
                    # to lower
                    if self.lower:
                        words = [word.lower() for word in words]
                    yield words[:self.max_words]


def start_train(source, opt, max_sentence_length, word_embedding_model_name):

    params = {
        'size': opt.size,
        'window': opt.window,
        'min_count': opt.min_count,
        'workers': max(1, multiprocessing.cpu_count() - 20),
        'sample': opt.sample,
        'alpha': opt.alpha,
        'hs': opt.hs,
        'negative': opt.negative,
        'iter': opt.epochs
    }

    word2vec = Word2Vec(LineSentence(source, max_sentence_length=max_sentence_length),
                        **params)

    # word2vec = Word2Vec(
    #     MemoryOptimalSentences(
    #         opt.corpus_path_list,
    #         max_words=max_words,
    #         lower=opt.lower,
    #         logger=logger),
    #                     **params)

    # vocab_size = len(word2vec.wv.vocab)
    # vocab_size_str = str(vocab_size)

    word2vec.save(os.path.join(opt.save_path, word_embedding_model_name + '.model'))  # Save the model.

    return word2vec
    # save word2vec
    # word2vec.wv.save_word2vec_format(
    #     opt.save_path + '/reddit.w2v.{}.{}.{}d.txt'.format(vocab_size_str[0], len(vocab_size_str), opt.size),
    #     binary=opt.binary)


if __name__ == '__main__':

    pass
