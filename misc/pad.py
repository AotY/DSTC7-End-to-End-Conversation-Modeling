#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

import torch
from modules.utils import to_device


def pad(tensor, length):
    if isinstance(tensor, torch.Tensor):
        if length > tensor.size(0):
            return torch.cat([tensor, to_device(torch.zeros(length - tensor.size(0), *tensor.size()[1:]))], dim=0)
        else:
            return tensor

def pad_and_pack(tensor_list):
    length_list = ([t.size(0) for t in tensor_list])
    max_len = max(length_list)
    padded = [pad(t, max_len) for t in tensor_list]
    packed = torch.stack(padded, 0)
    return packed, length_list
