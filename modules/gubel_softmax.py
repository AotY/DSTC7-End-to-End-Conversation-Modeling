#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
ST-Gumbel-Softmax-Pytorch
https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
"""
import torch
import torch.nn as nn
import troch.functional as F

class GumbelSoftmax(nn.Module):
    def __init__(temperature, dim, device):
        super(GumbelSoftMax, self).__init__()

        self.temperature = temperature
        self.dim = self.dim
        self.device = device

    def _gumbel_sample(self, shape, eps=1e-20):
        random_tensor = torch.rand(shape, device=device)
        return torch.log(-torch.log(random_tensor + eps) + eps)

    def _gumbel_softmax_smaple(self, logits):
        y = logits + self._gumbel_sample(logits.size())
        return F.softmax(y / self.temperature, dim=self.dim)

    def forward(self, logits):
        """
        input: [*, n_class]
        return: [*, n_class] as one-hot vector
        """
        y = self._gumbel_softmax_smaple(logits, self.temperature)
        shape = y.size()
        _, ind = y.max(dim=self.dim)
        y_hard = torch.zeros_lik(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)

        return (y_hard - y).detach() + y


