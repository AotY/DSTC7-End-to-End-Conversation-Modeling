# -*- coding: utf-8 -*-

'''
load pre-trained word embedding
google word2vec or
standford glove
'''
from __future__ import division
from __future__ import print_function

import codecs
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

'''
buid vocab embedding from word2vec
'''


def build_vocab_word2vec(word2vec_model, vocab, vocab_size, vec_file, embedding_dim, binary, save_vec_file, logger):

    # init
    vocab_embedding = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

    pad_embedding = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    sos_embedding = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    eos_embedding = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    unk_embedding = np.random.uniform(-0.25, 0.25, (embedding_dim,))

    # load any vectors from the word2vec
    logger.info("Load file: {} to gensim model. \n".format(vec_file))

    if word2vec_model is None:
        if binary:
            word2vec_model = KeyedVectors.load_word2vec_format(fname=vec_file, binary=True)
        else:
            word2vec_model = KeyedVectors.load_word2vec_format(fname=vec_file, binary=False)

    save_f = open(save_vec_file, 'w', encoding='utf-8')

    out_of_vocab_count = 0

    header = "%d %d\n" % (vocab_size, embedding_dim)

    # write header
    save_f.write(header)

    for id, word in vocab.idx2word.items():
        if id == vocab.padid:
            word_embedding = pad_embedding
        elif word == vocab.sosid:
            word_embedding = sos_embedding
        elif word == vocab.eosid:
            word_embedding = eos_embedding
        elif word == vocab.unkid:
            word_embedding = unk_embedding
        else:
            try:
                word_embedding = word2vec_model.wv[word]
            except KeyError:
                out_of_vocab_count += 1
                word_embedding = unk_embedding

        vocab_embedding[id] = word_embedding

        vector_str = ' '.join([str(s) for s in word_embedding])
        save_f.write('%s %s\n' % (word, vector_str))

    save_f.close()
    del word2vec_model

    return vocab_embedding, out_of_vocab_count


def load_word_embedding_for_lookup(vocab, vocab_size, vec_file, embedding_dim, binary):
    # init
    pre_trained_embedding = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))

    # load any vectors from the word2vec
    print("Load word2vec file {}\n".format(vec_file))

    if binary:
        model = 'rb'
    else:
        model = 'r'

    with codecs.open(vec_file, model, encoding='utf-8') as f:
        header = f.readline()
        word2vec_vocab_size, word2vec_embedding_dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * word2vec_embedding_dim

        for line in range(word2vec_vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break

                if ch != '\n':
                    word.append(ch)

            # word2idx
            idx = vocab.word2idx.get(word, vocab.unkid)

            if idx != vocab.unkid:
                pre_trained_embedding[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return pre_trained_embedding


'''
buid vocab embedding from glove
'''


def build_vocab_glove(vocab, vocab_size, glove_file, embedding_dim, binary,
                      pre_trained_vocab_embedding_file):
    # load any vectors from the word2vec
    print("Load glove file {}\n".format(glove_file))

    pad_embedding = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    sos_embedding = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    eos_embedding = np.random.uniform(-0.25, 0.25, (embedding_dim,))
    unk_embedding = np.random.uniform(-0.25, 0.25, (embedding_dim,))

    # load any vectors from the word2vec
    print("Load word2vec file: {} to gensim model. \n".format(glove_file))

    # fname, fvocab=None, binary=False, encoding='utf8'

    glove_file = datapath(glove_file)
    tmp_file = get_tmpfile("tmp_word2vec.txt")

    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_file, tmp_file)

    glove_model = KeyedVectors.load_word2vec_format(fname=tmp_file)

    out_of_vocab_count = 0

    if binary:
        save_f = codecs.open(pre_trained_vocab_embedding_file, 'w', encoding='utf-8')
    else:
        save_f = codecs.open(pre_trained_vocab_embedding_file, 'wb', encoding='utf-8')

    header = "%d %d\n" % (vocab_size, embedding_dim)
    # write header
    save_f.write(header)

    for word in vocab.word2idx.keys():

        if word == vocab.pad:
            word_embedding = pad_embedding
        elif word == vocab.sos:
            word_embedding = sos_embedding
        elif word == vocab.eos:
            word_embedding = eos_embedding
        elif word == vocab.unk:
            word_embedding = unk_embedding
        else:
            try:
                word_embedding = glove_model.wv[word]
            except KeyError:
                out_of_vocab_count += 1
                word_embedding = unk_embedding

        vector_str = ' '.join([str(s) for s in word_embedding])
        save_f.write('%s %s\n' % (word, vector_str))

    save_f.close()
    del glove_model

    return out_of_vocab_count


'''
buid vocab embedding from fastText
'''


def build_vocab_fastText(model, vocab, vocab_size, vec_file, embedding_dim, binary, pre_trained_vocab_embedding_file, logger):
    return build_vocab_word2vec(model, vocab, vocab_size, vec_file, embedding_dim, binary, pre_trained_vocab_embedding_file, logger)


if __name__ == '__main__':
    # Load vocab and confirm opt.vocab_size
    pass
