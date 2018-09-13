# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from utils import aeq


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb * math.sqrt(self.dim)
        emb = emb + Variable(self.pe[:emb.size(0)], requires_grad=False)
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.
    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.
    .. mermaid::
       graph LR
          A[Input]
          A-->B[Word Lookup]
          A-->C[Output]
    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        dropout (float): dropout probability.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 dropout=0,
                 sparse=False):
        super(Embeddings, self).__init__()
        self.word_padding_idx = word_padding_idx

        # Dimensions and padding for constructing the word embedding matrix
        self.vocab_size = word_vocab_size
        self.emb_dim = word_vec_size
        self.pad_indice = word_padding_idx

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim,
                                       padding_idx=self.pad_indice, sparse=sparse)
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = self.emb_dim

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.

    @property
    def word_lut(self):
        return self.embeddings

    @property
    def emb_luts(self):
        return self.embeddings

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.
        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.embeddings.weight.data.copy_(pretrained)
            if fixed:
                self.embeddings.weight.requires_grad = False

    def forward(self, input):
        """
        Computes the embeddings for words and features.
        Args:
            input (`LongTensor`): index tensor `[len x batch]`
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        in_length, in_batch = input.size()
        # aeq(nfeat, len(self.emb_luts))

        emb = self.embeddings(input)

        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)

        return emb