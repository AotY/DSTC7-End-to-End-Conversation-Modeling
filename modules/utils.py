# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
import torch.nn as nn

"""
    Utils
"""


def to_device(x, gpu_id=0, non_blocking=False):
    """Tensor => Variable"""
    if torch.cuda.is_available():
        x = x.cuda(gpu_id, non_blocking)
    return x

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

def sequence_mask(sequence_length, max_len=None):
    """
    Args:
        sequence_length (Variable, LongTensor) [batch_size]
            - list of sequence length of each batch
        max_len (int)
    Return:
        masks (bool): [batch_size, max_len]
            - True if current sequence is valid (not padded), False otherwise
    Ex.
    sequence length: [3, 2, 1]
    seq_length_expand
    [[3, 3, 3],
     [2, 2, 2]
     [1, 1, 1]]
    seq_range_expand
    [[0, 1, 2]
     [0, 1, 2],
     [0, 1, 2]]
    masks
    [[True, True, True],
     [True, True, False],
     [True, False, False]]
    """
    if max_len is None:
        max_len = sequence_length.max()

    batch_size = sequence_length.size(0)

    # [max_len]
    seq_range = torch.arange(0, max_len).long()  # [0, 1, ... max_len-1]

    # [batch_size, max_len]
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = to_device(seq_range_expand)

    # [batch_size, max_len]
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)

    # [batch_size, max_len]
    masks = seq_range_expand < seq_length_expand

    return masks

def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    if rnn_type in ['RNN', 'LSTM', 'GRU']:
        return getattr(nn, rnn_type)(**kwargs)
    else:
        raise ValueError("{} is not valid, ".format(rnn_type))



if __name__ == '__main__':
    print(torch.__version__)
    pass
