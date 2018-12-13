#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Session encoder, for HRED model.
"""

import torch
import torch.nn as nn

from modules.utils import rnn_factory
from modules.utils import init_gru_orth, init_lstm_orth


class SessionEncoder(nn.Module):
    def __init__(self, config):
        super(SessionEncoder, self).__init__()

        self.bidirection_num = 2 if config.bidirectional else 1

        # rnn
        self.rnn = rnn_factory(
            config.rnn_type,
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // self.bidirection_num,
            num_layers=config.encoder_num_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout
        )

        if config.rnn_type == 'LSTM':
            init_lstm_orth(self.rnn)
        else:
            init_gru_orth(self.rnn)


    def forward(self, inputs, lengths=None, sort=False):
        """
        inputs: [turn_num, batch_size, hidden_size]
        """
        if lengths is not None and not sort:
            # sort lengths
            lengths, sorted_indexes = torch.sort(lengths, dim=0, descending=True)
            # restore to original indexes
            _, restore_indexes = torch.sort(sorted_indexes, dim=0)

            inputs = inputs.transpose(0, 1)[sorted_indexes].transpose(0, 1)

        if lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths)

        outputs, hidden_state = self.rnn(inputs)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            if not sort:
                outputs = outputs.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()
                hidden_state = hidden_state.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()

        return outputs, hidden_state


