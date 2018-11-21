#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
custom optim
https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/optim/optim.py
"""

import itertools
import numpy as np

import torch
import torch.nn as nn

"""
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
"""
class ScheduledOptimizer:
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, model_dim, n_warmup_steps, init_lr=None, max_grad_norm=None):
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.max_grad_norm = max_grad_norm
        self.init_lr = init_lr or np.power(model_dim, -0.8)
        self.scheduler = None

    def set_scheduler(self, scheduler):
        """ Set the learning rate scheduler.
        Args:
            scheduler (torch.optim.lr_scheduler.*): object of learning rate scheduler,
               e.g. torch.optim.lr_scheduler.StepLR
        """
        self.scheduler = scheduler

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        #  self._update_learning_rate()

        if self.max_grad_norm > 0:
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

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update(self, loss, epoch):
        """ Update the learning rate if the criteria of the scheduler are met.
        Args:
            loss (float): The current loss.  It could be training loss or developing loss
                depending on the caller.  By default the supervised trainer uses developing
                loss.
            epoch (int): The current epoch number.
        """
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()
