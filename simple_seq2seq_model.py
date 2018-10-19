#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

import math
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 dropout_ratio,
                 padding_idx):

        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, self.padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout_ratio)

        # LSTM
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size)

    def forward(self, inputs, inputs_length, hidden_state):
        '''
        params:
            inputs: [seq_len, batch_size]  LongTensor
            hidden_state: [num_layers * bidirectional, batch_size, hidden_size]
        :return
            outputs: [seq_len, batch, num_directions * hidden_size]
            hidden_state: (h_n, c_n)
        '''

        # embedded
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        # [batch_size, seq_len, embedding_size]
        embedded = embedded.transpose(0, 1)

        # sort lengths
        _, sorted_indexes = torch.sort(inputs_length, dim=0, descending=True)

        new_inputs_length = inputs_length[sorted_indexes]

        # restore to origianl indexs
        _, restore_indexes = torch.sort(sorted_indexes, dim=0)

        # new embedded
        embedded = embedded[sorted_indexes].transpose(0, 1)  # [seq_len, batch_size, embedding_size]

        # pack
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, new_inputs_length)

        # batch_size, hidden_size]
        outputs, hidden_state = self.lstm(packed_embedded, hidden_state)

        # unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # to original sequence
        outputs = outputs.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous()
        hidden_state = tuple([item.transpose(0, 1)[restore_indexes].transpose(0, 1).contiguous() for item in hidden_state])

        return outputs, hidden_state

    def get_output_hidden_size(self):
        return self.hidden_size

    def init_hidden(self, batch_size, device):
        initial_state_scale = math.sqrt(3.0 / self.hidden_size)

        initial_state1 = torch.rand((1, batch_size, self.hidden_size), device=device)
        initial_state2 = torch.rand((1, batch_size, self.hidden_size), device=device)

        nn.init.uniform_(initial_state1, a=-initial_state_scale, b=initial_state_scale)
        nn.init.uniform_(initial_state2, a=-initial_state_scale, b=initial_state_scale)
        return (initial_state1, initial_state2)


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 dropout_ratio,
                 padding_idx):

        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        # embedding
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_size, self.padding_idx)

        # dropout
        self.dropout = nn.Dropout(dropout_ratio)

        # LSTM
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size)

        # linear
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

        # log softmax
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden_state, encoder_outputs=None):
        '''
        input: [1, batch_size]  LongTensor
        hidden_state: [num_layers, batch_size, hidden_size]
        output: [seq_len, batch, hidden_size] [1, batch_size, hidden_size]
        hidden_state: (h_n, c_n)
        '''

        # embedded
        embedded = self.embedding(input) #[1, batch_size, embedding_size]
        embedded = self.dropout(embedded)

        # lstm
        output, hidden_state = self.lstm(embedded, hidden_state)

        # [1, batch_size, hidden_size]
        # linear
        output = self.linear(output)

        # softmax
        output = self.softmax(output)

        return output, hidden_state



class Seq2seq(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 dropout_ratio,
                 padding_idx,
                 device=None):

        # super class init
        super(Seq2seq, self).__init__()

        self.device = device

        # encoder
        self.encoder = Encoder(vocab_size,
                               embedding_size,
                               hidden_size,
                               dropout_ratio,
                               padding_idx)

        # decoder
        self.decoder = Decoder(vocab_size,
                               embedding_size,
                               hidden_size,
                               dropout_ratio,
                               padding_idx)

    def forward(self,
                encoder_inputs,
                encoder_inputs_length,
                decoder_input,
                decoder_targets,
                batch_size,
                max_len):

        '''
        input:
            encoder_inputs: [seq_len, batch_size]
            encoder_inputs_length: [batch_size]
            decoder_input: [1, batch_size], first step: [sos * batch_size]
            decoder_targets: [seq_len, batch_size]
        '''
        # encoder
        encoder_hidden_state = self.encoder.init_hidden(batch_size, self.device)

        encoder_outputs, encoder_hidden_state = self.encoder(
            encoder_inputs, encoder_inputs_length, encoder_hidden_state)

        # decoder
        decoder_hidden_state = encoder_hidden_state
        decoder_outputs = []
        for di in range(decoder_targets.shape[0]):
            decoder_output, decoder_hidden_state = self.decoder(
                decoder_input, decoder_hidden_state, encoder_outputs)

            decoder_input = decoder_targets[di].view(1, -1)
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        return encoder_outputs, decoder_outputs

    '''evaluate'''

    def evaluate(self,
                 encoder_inputs,
                 encoder_inputs_length,
                 decoder_input,
                 max_len,
                 eosid,
                 batch_size):
        '''
        encoder_inputs: [seq_len, batch_size], maybe [max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        # encoder
        encoder_hidden_state = self.encoder.init_hidden(batch_size)

        encoder_outputs, encoder_hidden_state = self.encoder(
            encoder_inputs, encoder_inputs_length, encoder_hidden_state)

        decoder_outputs = []
        decoder_hidden_state = encoder_hidden_state
        for di in range(encoder_inputs.shape[0]):
            decoder_output, decoder_hidden_state = self.decoder(decoder_input, decoder_hidden_state, encoder_outputs)

            print('decoder_output shape: ', decoder_output.shape)
            #  topv, topi = decoder_output.topk(1, dim=1)
            #  decoder_input = topi.squeeze().detach()
            decoder_input = torch.argmax(decoder_output, dim=2).detach()
            decoder_outputs.append(decoder_output)

            ni = decoder_input[0].item()
            if ni == eosid:
                break

        return decoder_outputs
