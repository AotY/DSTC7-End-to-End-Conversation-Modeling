# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as functional

"""
    Utils
"""


"""
orthogonal initialization
"""
def init_gru_orth(model, gain=1):
    model.reset_parameters()
    # orthogonal initialization of gru weights
    for _, hh, _, _ in model.all_weights:
        for i in range(0, hh.size(0), model.hidden_size):
            I.orthogonal_(hh[i:i + model.hidden_size], gain=gain)

def init_lstm_orth(model, gain=1):
    init_gru_orth(model, gain)

    #positive forget gate bias (Jozefowicz es at. 2015)
    for _, _, ih_b, hh_b in model.all_weights:
        l = len(ih_b)
        ih_b[l // 4 : l // 2].data.fill_(1.0)
        hh_b[l // 4 : l // 2].data.fill_(1.0)

rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag,
                                 rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
           (hasattr(opt, 'gpu') and opt.gpu > -1)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    if rnn_type in ['RNN', 'LSTM', 'GRU']:
        return getattr(nn, rnn_type)(**kwargs)
    else:
        raise ValueError("{} is not valid, ".format(rnn_type))


def compute_recall_ks(probas):
    recall_k = {}
    if isinstance(probas, list):
        probas = probas
    elif isinstance(probas, torch.Tensor):
        probas = probas.numpy().tolist()
    else:
        print("")
    for group_size in [2, 5, 10]:
        recall_k[group_size] = {}
        print('group_size: %d' % group_size)
        for k in [1, 2, 5]:
            if k < group_size:
                recall_k[group_size][k] = recall(probas, k, group_size)
                print('recall@%d' % k, recall_k[group_size][k])
    return recall_k


def recall(probas, k, group_size):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    for i in range(n_batches):
        batch = np.array(probas[i * test_size:(i + 1) * test_size])[:group_size]
        try:
            indices = np.argpartition(batch.reshape((-1,)), -k)[-k:]
        except:
            print(batch, k, group_size)
        # indices = np.argpartition(batch, k)[:k]
        if 0 in indices:
            n_correct += 1
    print(n_correct, len(probas), test_size)
    return n_correct * 1.0 / (len(probas) / test_size)


if __name__ == '__main__':
    print(torch.__version__)
    pass
