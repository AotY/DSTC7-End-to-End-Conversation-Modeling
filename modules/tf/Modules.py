import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        # matmul
        attn = torch.bmm(q, k.transpose(1, 2))
        # scale
        attn = attn / self.temperature

        # mask(opt)
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        # softmax
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # matmul
        output = torch.bmm(attn, v)

        return output, attn
