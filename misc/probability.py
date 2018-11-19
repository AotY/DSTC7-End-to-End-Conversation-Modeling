#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import numpy as np
import torch
from modules.utils import to_device

def normal_logpdf(x, mean, var):
    """
    Args:
        x: [batch_size, dim]
        mean: [batch_size, dim] or [batch_size] or [1]
        var: [batch_size, dim]: positive value
    Return:
        log_p: [batch_size]
    """

    pi = to_device(torch.FloatTensor([np.pi]))
    return 0.5 * torch.sum(-torch.log(2.0 * pi) - torch.log(var) - ((x - mean).pow(2) / var), dim=1)


def normal_kl_div(mu1, var1,
                  mu2=to_device(torch.FloatTensor([0.0])),
                  var2=to_device(torch.FloatTensor([1.0]))):
    one = to_device(torch.FloatTensor([1.0]))
    return torch.sum(0.5 * (torch.log(var2) - torch.log(var1)
                            + (var1 + (mu1 - mu2).pow(2)) / var2 - one), 1)
