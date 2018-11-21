#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
custom optim
https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/optim/optim.py
"""

import torch
import torch.nn as nn
import itertools

"""
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
"""
class ScheduledOptimizer:
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, scheduler, max_grad_norm=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

    def set_scheduler(self, scheduler):
        """ Set the learning rate scheduler.
        Args:
            scheduler (torch.optim.lr_scheduler.*): object of learning rate scheduler,
               e.g. torch.optim.lr_scheduler.StepLR
        """
        self.scheduler = scheduler

    def step(self):
        "Step with the inner optimizer"
        if self.max_grad_norm > 0:
            params = itertools.chain.from_iterable([group['params'] for group in self.optimizer.param_groups])
            _ = nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update(self, loss=None):
        """ Update the learning rate if the criteria of the scheduler are met.
        Args:
            loss (float): The current loss.  It could be training loss or developing loss
                depending on the caller.  By default the supervised trainer uses developing
                loss.
        """
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()
