#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Early Stopping.
"""
import numpy as np


class EarlyStopping:
    def __init__(self,
                 type='min',
                 min_delta=0,
                 patience=10):

        self.type = type
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(type, min_delta)

    def _init_is_better(self,
                        type,
                        min_delta):
        if type not in ['min', 'max']:
            raise ValueError('type: %s is unknown.' % type)

        if type == 'min':
            self.is_better = lambda cur, best: cur < best - min_delta
        elif type == 'max':
            self.is_better = lambda cur, best: cur > best + min_delta

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if metrics is None or np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False
