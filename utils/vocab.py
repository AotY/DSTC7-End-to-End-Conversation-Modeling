# -*- coding: utf-8 -*-
"""
    Manage Vocabulary,
        1), Build Vocabulary
        2), save and load vocabulary
        3),
"""
from __future__ import division
from __future__ import print_function
import pickle
import logging

logger = logging.getLogger(__name__)

PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'


class Vocab(object):
    def __init__(self):
        self.init_vocab()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}

        self.word2idx[UNK] = 0
        self.word2idx[PAD] = 1
        self.word2idx[SOS] = 2
        self.word2idx[EOS] = 3

    def get_vocab_size(self):
        return len(self.word2idx)

    def word_to_id(self, word):
        return self.word2idx.get(word, UNK)

    def words_to_id(self, words):
        word_ids = [self.word_to_id(cur_word) for cur_word in words]
        return word_ids

    def build_for_frequency(self, freq_list):
        cur_id = 4  # because of the unk, pad, sos, and eos tokens.
        for word, _ in freq_list:
            self.word2idx[word] = cur_id
            cur_id += 1

        # init idx2word
        self.idx2word = {v: k for k, v in self.word2idx.items()}


    '''save and restore'''
    def save(self):
        if len(self.idx2word) == 0:
            raise RuntimeError("Save vocab after call build_for_frequency()")

        pickle.dump(self.word2idx, open('vocab_word2idx.dict', 'wb'))
        # pickle.dump(self.idx2word, open('./vocab_idx2word.dict', 'wb'))

    def load(self):
        try:
            self.word2idx = pickle.load(open('vocab_word2idx.dict', 'rb'))
            self.idx2word = {v: k for k, v in self.word2idx.items()}
        except:
            raise RuntimeError("Make sure vocab_word2idx.dict exists.")




    ''' wordid '''

    @property
    def unkid(self):
        """return the id of unknown word
        """
        return self.word2idx.get(UNK, 0)

    @property
    def padid(self):
        """return the id of padding
        """
        return self.word2idx.get(PAD, 1)

    @property
    def sosid(self):
        """return the id of padding
        """
        return self.word2idx.get(SOS, 2)

    @property
    def eosid(self):
        """return the id of padding
        """
        return self.word2idx.get(EOS, 3)

    '''words '''

    @property
    def unk(self):
        """return the id of unknown word
        """
        return UNK

    @property
    def pad(self):
        """return the id of padding
        """
        return PAD

    @property
    def sosid(self):
        """return the id of padding
        """
        return SOS

    @property
    def eosid(self):
        """return the id of padding
        """
        return EOS
