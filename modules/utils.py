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
