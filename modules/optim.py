#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
custom optim
https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/optim/optim.py
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
"""

import torch.nn as nn
import numpy as np
import itertools


class ScheduledOptimizer:
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, model_size, n_warmup_steps, max_grad_norm=None):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.max_grad_norm = max_grad_norm
        self.init_lr = np.power(model_size, -0.5)

    def step(self):
        self.update()

        "Step with the inner optimizer"
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            params = itertools.chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
            _ = nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def update(self, loss=None):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
