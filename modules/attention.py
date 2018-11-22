#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
"""

import torch
import torch.nn as nn

from modules.utils import sequence_mask
from modules.utils import init_linear_wt


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = encoder_outputs * output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * encoder_outputs) + b * output)
            \end{array}
    Args:
        hidden_size(int): The number of expected features in the output
    Inputs: output, encoder_outputs
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **encoder_outputs** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)
        init_linear_wt(self.linear_out)

    def forward(self, output, encoder_outputs, lengths=None):
        """
        output: maybe [r_len, batch_size, hidden_size] or [1, batch_size, hidden_size]
        encoder_outputs: [c_len, batch_size, hidden_size]
        """

        output_len, batch_size, hidden_size = output.shape
        input_size = encoder_outputs.size(0)

        # (batch, out_len, hidden_size) * (batch, hidden_size, in_len) -> (batch, out_len, in_len)
        attn = torch.bmm(output.transpose(0, 1), encoder_outputs.permute(1, 2, 0))

        if lengths is not None:
            mask = sequence_mask(lengths, max_len=attn.size(-1)) # mask: [batch_size, in_len]
            mask = mask.unsqueeze(1).expanded(1, output_len, 1)  # Make it broadcastable.
            attn.data.masked_fill_(1 - mask, -float('inf'))

        # [batch, out_len, in_len]
        attn = torch.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, hidden_size) -> (batch, out_len, hidden_size)
        mix = torch.bmm(attn, encoder_outputs.transpose(0, 1))
        mix = mix.transpose(0, 1) #[out_len, batch_size, hidden_size]

        # concat -> (out_len, batch_size, 2 * hidden_size)
        combined = torch.cat((mix, output), dim=2)

        # context -> (batch, out_len, hidden_size)
        context = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(-1, batch_size, hidden_size)

        return context, attn
