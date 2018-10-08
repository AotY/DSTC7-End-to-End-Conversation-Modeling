# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.utils import aeq


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embeddededding size
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

    def forward(self, embedded):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap embedded(i.e. embedded.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        embedded = embedded * math.sqrt(self.dim)
        embedded = embedded + \
            Variable(self.pe[:embedded.size(0)], requires_grad=False)
        embedded = self.dropout(embedded)
        return embedded


class Embedding(nn.Module):
    """
    Words embedding for encoder/decoder.
    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.
    .. mermaid::
       graph LR
          A[Input]
          A-->B[Word Lookup]
          A-->C[Output]
    Args:
        embeddededdign_dim (int): size of the dictionary of embedding.
        padding_idx (int): padding index for words in the embedding.
        vocab_size (int): size of dictionary of embedding for words.
        dropout (float): dropout probability.
    """

    def __init__(self,
                 embedding_size,
                 vocab_size,
                 padding_idx,
                 dropout_ratio=0.0,
                 sparse=False):

        super(Embedding, self).__init__()

        self.padding_idx = padding_idx

        # Dimensions and padding for constructing the word embeddededding matrix
        self.vocab_size = vocab_size

        # This is the attribute you should access if you need to know
        # how big your embedding are going to be.
        self.embedding_size = embedding_size

        self.padding_idx = padding_idx

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size,
                                      padding_idx=self.padding_idx, sparse=sparse)

        # The sequence of operations that converts the input sequence
        # into a sequence of embedding. At minimum this consists of
        # looking up the embedding for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        self.dropout = None
        if dropout_ratio != 0.0:
            self.dropout = nn.Dropout(p=dropout_ratio)

    '''
    @property
    def word_lut(self):
        return self.embedding

    @property
    def embedded_luts(self):
        return self.embeddins
    '''

    def get_lookup_table(self):
        return self.embedding

    '''obtain weight of embedding, see opt.tied'''
    def get_embedding_weight(self):
        return self.embedding.weight

    def set_pretrained_embedding(self, pre_trained_weight=None, fixed=False):
        """Set pretrained embedding.
        Args:
          pre_trained_weight (str) : path to torch serialized embedding
          fixed (bool) : if true, embedding are not updated
        """
        if pre_trained_weight is not None:
            if not isinstance(pre_trained_weight, torch.Tensor):
                pre_trained_weight = torch.from_numpy(pre_trained_weight)
            self.embedding.weight.data.copy_(pre_trained_weight)
            if fixed:
                self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        """
        Computes the embedding for words and features.
        Args:
            inputs (`LongTensor`): index tensor `[len x batch]`
        Return:
            `FloatTensor`: word embedding `[len x batch x embeddededding_size]`
        """

        in_length, in_batch = inputs.size()
        #print("inputs shape: {}", inputs.shape)

        # aeq(nfeat, len(self.embedded_luts))

        embedded = self.embedding(inputs)
        
        #print("self.droput_ratio: %f" % self.dropout.p)

        #print("embedded shape: {}".format(embedded.shape))
        #print("embedded device: {}".format(embedded.device))
        #print("embedded: {}".format(embedded))

        if self.dropout is not None:
            embedded = self.dropout(embedded)

        out_length, out_batch, embedded_size = embedded.size()

        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(embedded_size, self.embedding_size)

        return embedded


    
