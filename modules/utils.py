# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
jmport torch.nn as nn

"""
    Utils
"""


# orthogonal initialization
def init_gru_orth(model, gain=1):
    model.reset_parameters()
    # orthogonal initialization of gru weights
    for _, hh, _, _ in model.all_weights:
        for i in range(0, hh.size(0), model.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + model.hidden_size], gain=gain)

def init_lstm_orth(model, gain=1):
    init_gru_orth(model, gain)

    #positive forget gate bias (Jozefowicz es at. 2015)
    for _, _, ih_b, hh_b in model.all_weights:
        l = len(ih_b)
        ih_b[l // 4 : l // 2].data.fill_(1.0)
        hh_b[l // 4 : l // 2].data.fill_(1.0)

def init_linear_wt(linear):
    init_wt_normal(linear.weight, linear.in_features)
    if linear.bias is not None:
        init_wt_normal(linear.bias, linear.in_features)

def init_wt_normal(weight, dim=512):
    weight.data.normal_(mean=0, std=np.sqrt(2.0 / dim))

def init_wt_unif(weight, dim=512):
    weight.data.uniform_(-np.sqrt(3.0 / dim), np.sqrt(3.0 / dim))

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel() # elements number
    max_len = max_len or lengths.max() # max_len
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat((batch_size, 1))
            .lt(lengths.unsqueeze(1)))


def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    if rnn_type in ['RNN', 'LSTM', 'GRU']:
        return getattr(nn, rnn_type)(**kwargs)
    else:
        raise ValueError("{} is not valid, ".format(rnn_type))


def get_attn_key_pad_mask(k, q, padid):
    '''
        For masking out the padding part of key sequence.
        k: [batch_size, max_len]
        q: [batch_size, max_len]
    '''
    # Expand to fit the shape of key query attention matrix.
    len_q = q.size(1)
    padding_mask = k.eq(padid)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

if __name__ == '__main__':
    print(torch.__version__)
    pass
