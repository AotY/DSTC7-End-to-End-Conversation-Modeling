# -*- coding: utf-8 -*-

'''
load pre-trained word embedding
google word2vec or
standford glove
'''
from __future__ import division

import codecs
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
'''
buid vocab embedding from word2vec
'''


def build_vocab_word2vec(word2vec_model, vocab, vec_file, embedding_dim, binary, save_vec_file, logger):

    vocab_size = vocab.get_vocab_size()
    # init
    vocab_embedded = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

    pad_embedded = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    unk_embedded = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    sos_embedded = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    eos_embedded = np.random.uniform(-0.25, 0.25, (embedding_dim,))

    # load any vectors from the word2vec
    logger.info("Load file: {} to gensim model. \n".format(vec_file))

    if word2vec_model is None:
        if binary:
            word2vec_model = KeyedVectors.load_word2vec_format(fname=vec_file, binary=True)
        else:
            word2vec_model = KeyedVectors.load_word2vec_format(fname=vec_file, binary=False)

    save_f = open(save_vec_file, 'w', encoding='utf-8')

    out_of_vocab_count = 0
    out_of_vocab_words = []

    header = "%d %d\n" % (vocab_size, embedding_dim)

    # write header
    save_f.write(header)

    for id, word in vocab.idx2word.items():
        if id == vocab.padid:
            word_embedded = pad_embedded
        elif word == vocab.sosid:
            word_embedded = sos_embedded
        elif word == vocab.eosid:
            word_embedded = eos_embedded
        elif word == vocab.unkid:
            word_embedded = unk_embedded
        else:
            try:
                word_embedded = word2vec_model.wv[word]
            except KeyError:
                out_of_vocab_words.append(word)
                out_of_vocab_count += 1
                word_embedded = unk_embedded

        vocab_embedded[id] = word_embedded

        vector_str = ' '.join([str(s) for s in word_embedded])
        save_f.write('%s %s\n' % (word, vector_str))

    save_f.close()
    del word2vec_model

    return vocab_embedded, out_of_vocab_count, out_of_vocab_words

'''
buid vocab embedding from glove
'''


def build_vocab_glove(vocab, glove_file, embedding_dim, binary,
                      pre_trained_vocab_embedding_file):

    vocab_size = vocab.get_vocab_size()
    
    # init
    vocab_embedded = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

    # load any vectors from the word2vec
    print("Load glove file {}\n".format(glove_file))

    pad_embedded = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    unk_embedded = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    sos_embedded = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    eos_embedded = np.random.uniform(-0.25, 0.25, (embedding_dim,))

    # load any vectors from the word2vec
    print("Load glove file: {} to gensim model. \n".format(glove_file))

    # fname, fvocab=None, binary=False, encoding='utf8'

    glove_file = datapath(glove_file)
    tmp_file = get_tmpfile("tmp_word2vec.txt")

    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, tmp_file)

    glove_model = KeyedVectors.load_word2vec_format(fname=tmp_file)

    out_of_vocab_count = 0
    out_of_vocab_words = []

    if binary:
        save_f = codecs.open(pre_trained_vocab_embedding_file, 'w', encoding='utf-8')
    else:
        save_f = codecs.open(pre_trained_vocab_embedding_file, 'wb', encoding='utf-8')

    header = "%d %d\n" % (vocab_size, embedding_dim)
    # write header
    save_f.write(header)

    for id, word in vocab.idx2word.items():
        if id == vocab.padid:
            word_embedded = pad_embedded
        elif id == vocab.sosid:
            word_embedded = sos_embedded
        elif id == vocab.eosid:
            word_embedded = eos_embedded
        elif id == vocab.unkid:
            word_embedded = unk_embedded
        else:
            try:
                word_embedded = glove_model.wv[word]
            except KeyError:
                out_of_vocab_words.append(word)
                out_of_vocab_count += 1
                word_embedded = unk_embedded

        vector_str = ' '.join([str(s) for s in word_embedded])
        save_f.write('%s %s\n' % (word, vector_str))

        vocab_embedded[id] = word_embedded

    save_f.close()
    del glove_model

    return vocab_embedded, out_of_vocab_count, out_of_vocab_words 


'''
buid vocab embedding from fastText
'''

def build_vocab_fastText(fasttest_model, vocab, vec_file, embedding_dim, binary, save_vec_file, logger):

    vocab_size = vocab.get_vocab_size()

    # init
    vocab_embedded = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

    # load any vectors from the word2vec
    logger.info("Load fasttest file: {} to gensim model. \n".format(vec_file))

    if fasttest_model is None:
        # fasttest_model = FastText.load_fasttext_format(vec_file)
        fasttest_model = KeyedVectors.load_word2vec_format(vec_file, binary=False)

    save_f = open(save_vec_file, 'w', encoding='utf-8')

    out_of_vocab_count = 0
    out_of_vocab_words = []

    header = "%d %d\n" % (vocab_size, embedding_dim)

    # write header
    save_f.write(header)

    for id, word in vocab.idx2word.items():
        word_embedded = None
        try:
            word_embedded = fasttest_model[word]
        except KeyError:
            out_of_vocab_words.append(word)
            out_of_vocab_count += 1

        if word_embedded is not None:
            vocab_embedded[id] = word_embedded

        vector_str = ' '.join([str(s) for s in word_embedded])
        save_f.write('%s %s\n' % (word, vector_str))

    save_f.close()
    del fasttest_model

    return vocab_embedded, out_of_vocab_count, out_of_vocab_words

if __name__ == '__main__':
    # Load vocab and confirm opt.vocab_size
    pass
