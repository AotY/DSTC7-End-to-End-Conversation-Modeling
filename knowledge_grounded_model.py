# -*- coding: utf-8 -*-

import math
import random
import torch
import torch.nn as nn

from modules.encoder import RNNEncoder
from modules.decoder import StdRNNDecoder

"""
KnowledgeGroundedModel
1. dialog_encoder
2. facts_encoder
3. dialog_decoder

"""


class KnowledgeGroundedModel(nn.Module):
    '''
    Knowledge-ground model
    Conditoning responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 dialog_encoder_embedding_size=300,
                 dialog_encoder_vocab_size=None,
                 dialog_encoder_hidden_size=300,
                 dialog_encoder_num_layers=2,
                 dialog_encoder_rnn_type='LSTM',
                 dialog_encoder_dropout_rate=0.5,
                 dialog_encoder_max_length=32,
                 dialog_encoder_clipnorm=50.0,
                 dialog_encoder_bidirectional=True,
                 dialog_encoder_embedding=None,
                 dialog_encoder_pad_id=0,

                 facts_embedding_size=300,
                 facts_vocab_size=None,
                 facts_dropout_rate=0.5,
                 facts_max_length=32,

                 dialog_decoder_embedding_size=300,
                 dialog_decoder_vocab_size=None,
                 dialog_decoder_hidden_size=300,
                 dialog_decoder_num_layers=2,
                 dialog_decoder_rnn_type='LSTM',
                 dialog_decoder_dropout_rate=0.5,
                 dialog_decoder_clipnorm=1.0,
                 dialog_decoder_max_length=32,
                 dialog_decoder_embedding=None,
                 dialog_decoder_pad_id=0,
                 dialog_decoder_eos_id=3,
                 dialog_decoder_attention_type='dot',
                 dialog_decoder_tied=True,
                 device=None
                 ):
        # super init
        super(KnowledgeGroundedModel, self).__init__()

        '''Dialog encoder parameters'''
        self.dialog_encoder_vocab_size = dialog_encoder_vocab_size
        self.dialog_encoder_embedding_size = dialog_encoder_embedding_size
        self.dialog_encoder_hidden_size = dialog_encoder_hidden_size
        self.dialog_encoder_num_layers = dialog_encoder_num_layers
        self.dialog_encoder_rnn_type = dialog_encoder_rnn_type
        self.dialog_encoder_dropout_rate = dialog_encoder_dropout_rate
        self.dialog_encoder_max_length = dialog_encoder_max_length
        self.dialog_encoder_clipnorm = dialog_encoder_clipnorm
        self.dialog_encoder_bidirectional = dialog_encoder_bidirectional
        self.dialog_encoder_pad_id = dialog_encoder_pad_id
        self.dialog_encoder_tied = dialog_encoder_tied

        '''facts encoder parameters'''
        self.facts_embedding_size = facts_embedding_size
        self.facts_vocab_size = facts_vocab_size
        self.facts_max_length = facts_max_length
        self.facts_dropout_rate = facts_dropout_rate

        '''Dialog decoder parameters'''
        self.dialog_decoder_vocab_size = dialog_decoder_vocab_size
        self.dialog_decoder_embedding_size = dialog_decoder_embedding_size
        self.dialog_decoder_hidden_size = dialog_decoder_hidden_size
        self.dialog_decoder_num_layers = dialog_decoder_num_layers
        self.dialog_decoder_rnn_type = dialog_decoder_rnn_type
        self.dialog_decoder_dropout_rate = dialog_decoder_dropout_rate
        self.dialog_decoder_max_length = dialog_decoder_max_length
        self.dialog_decoder_clipnorm = dialog_decoder_clipnorm
        self.dialog_decoder_pad_id = dialog_decoder_pad_id
        self.dialog_decoder_eos_id = dialog_decoder_eos_id
        self.dialog_decoder_attention_type = dialog_decoder_attention_type
        self.dialog_decoder_tied = dialog_decoder_tied

        self.device = device
        # num_embedding, embedding_dim
        ''''Ps: dialog_encoder, facts_encoder, and dialog_decoder may have different
            word embedding.
        '''
        # Dialog Encoder
        self.dialog_encoder = RNNEncoder(
            rnn_type=self.dialog_encoder_rnn_type,
            bidirectional=self.dialog_encoder_bidirectional,
            num_layers=self.dialog_encoder_num_layers,
            hidden_size=self.dialog_encoder_hidden_size,
            dropout=self.dialog_encoder_dropout_rate,
            embedding=dialog_encoder_embedding)

        # facts encoder
        # mi = A * ri
        self.facts_linearA = nn.Linear(self.facts_embedding_size,
                                       self.dialog_decoder_hidden_size)
        # ci = C * ri
        self.facts_linearC = nn.Linear(self.facts_embedding_size,
                                       self.dialog_decoder_hidden_size)

        # softmax
        self.facts_softmax = nn.SoftMax(dim=0)

        # Dialog Decoder with Attention
        self.dialog_decoder = StdRNNDecoder(
            rnn_type=self.dialog_decoder_rnn_type,
            bidirectional_encoder=self.dialog_encoder_bidirectional,
            num_layers=self.dialog_decoder_num_layers,
            hidden_size=self.dialog_decoder_hidden_size,
            dropout=self.dialog_decoder_dropout_rate,
            embedding=dialog_decoder_embedding,  # maybe replace by dialog_decoder_embedding
            attn_type=self.dialog_decoder_attention_type)

        self.dialog_decoder_linear = nn.Linear(
            self.dialog_decoder_hidden_size, self.dialog_decoder_vocab_size)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if self.dialog_decoder_tied:
            if self.dialog_decoder_embedding_size != self.dialog_decoder_hidden_size:
                raise ValueError(
                    'When using the tied flag, hidden_size must be equal to embedding_size.')
            print('using tied..................')
            self.dialog_decoder_linear.weight = dialog_decoder_embedding.get_embedding_weight()

        self.dialog_decoder_softmax = nn.LogSoftmax(dim=1)

    '''
    Seq2SeqModel forward
    '''

    def forward(self,
                dialog_encoder_inputs,  # LongTensor
                dialog_encoder_inputs_length,
                facts_inputs,
                facts_inputs_length,
                dialog_decoder_inputs,
                teacher_forcing_ratio=0.5,
                batch_size=128):
        """
        Args:
            - dialog_encoder_inputs: [max_length, batch_size]
            - dialog_encoder_inputs_length: [max_length, batch_size]
            - facts_inputs: [facts_size, facts_embedding_size]
            - facts_inputs: [facts_inputs_length * len(facts_inputs)]
            - dialog_decoder_inputs: [max_length, batch_size]
        """
        # init, [-sqrt(3/hidden_size), sqrt(3/hidden_size)]

        dialog_encoder_state = self.dialog_encoder.init_hidden(
            batch_size, self.device)

        '''dialog_encoder forward'''
        dialog_encoder_state, dialog_encoder_memory_bank = self.dialog_encoder(
            src=dialog_encoder_inputs,
            lengths=dialog_encoder_inputs_length,
            encoder_state=dialog_encoder_state,
        )

        '''dialog_decoder forward'''

        # init decoder state
        dialog_decoder_state = self.dialog_decoder.init_decoder_state(
            encoder_final=dialog_encoder_state)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        dialog_decoder_outputs = torch.ones((self.dialog_decoder_max_length,
                                             batch_size, self.dialog_decoder_hidden_size),
                                            device=self.device) * self.dialog_decoder_pad_id

        dialog_decoder_attns_std = torch.zeros((self.dialog_decoder_max_length,
                                                batch_size, self.dialog_decoder_max_length-2))

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.dialog_decoder_max_length):
                '''facts encoder forward'''
                new_hidden_tuple = self.facts_forward(facts_inputs,
                                                      dialog_decoder_state,
                                                      batch_size)
                dialog_decoder_state.update_state(new_hidden_tuple)

                dialog_decoder_state, dialog_decoder_output, \
                    dialog_decoder_attn = self.dialog_decoder(
                        tgt=dialog_decoder_inputs[di].view(1, -1),
                        memory_bank=dialog_encoder_memory_bank,
                        state=dialog_decoder_state,
                        memory_lengths=dialog_encoder_inputs_length)
                dialog_decoder_outputs[di] = dialog_decoder_output.squeeze(0)
                dialog_decoder_attns_std[di] = dialog_decoder_attn['std'].squeeze(
                    0)
        else:
            # Without teacher forcing: use its own predictions as the next
            # input
            dialog_decoder_input = dialog_decoder_inputs[0]
            for di in range(self.dialog_decoder_max_length):
                '''facts encoder forward'''
                new_hidden_tuple = self.facts_forward(
                    facts_inputs,
                    dialog_decoder_state,
                    batch_size)
                dialog_decoder_state.update_state(new_hidden_tuple)
                dialog_decoder_state, dialog_decoder_output, \
                    dialog_decoder_attn = self.dialog_decoder(
                        tgt=dialog_decoder_input.view(1, -1),
                        memory_bank=dialog_encoder_memory_bank,
                        state=dialog_decoder_state,
                        memory_lengths=dialog_encoder_inputs_length)
                dialog_decoder_output = dialog_decoder_output.detach().squeeze(0)
                dialog_decoder_outputs[di] = dialog_decoder_output
                dialog_decoder_attns_std[di] = dialog_decoder_attn['std'].squeeze(
                    0)
                dialog_decoder_input = torch.argmax(
                    dialog_decoder_output, dim=1)

                if dialog_decoder_input[0].item() == self.dialog_decoder_eos_id:

                    # beam search  dialog_decoder_outputs -> [tgt_len x batch x hidden]
        dialog_decoder_outputs = self.dialog_decoder_linear(
            dialog_decoder_outputs)
        # log softmax
        dialog_decoder_outputs = self.dialog_decoder_softmax(
            dialog_decoder_outputs)

        return ((dialog_encoder_state, dialog_encoder_memory_bank),
                (dialog_decoder_state, dialog_decoder_outputs, dialog_decoder_attns_std))

    def facts_forward(self,
                      facts_inputs,
                      dialog_decoder_state,
                      batch_size):
                # M
        facts_M = self.facts_linearA(facts_inputs)
        # C
        facts_C = self.facts_lienarC(facts_inputs)
        # softmax
        hidden_tuple = dialog_decoder_state
        # hidden_tuple is a tuple object
        new_hidden_list = []
        for hidden_state in hidden_tuple:
            new_hidden_state = torch.zeros((self.dialog_decoder_num_layers,
                                            batch_size,
                                            self.dialog_decoder_hidden_size),
                                           device=self.device)
            for i in range(self.dialog_decoder_num_layers):
                u = hidden_state[i]
                tmpP = u @ facts_M.t()
                P = self.facts_softmax(tmpP.view(-1))
                # o
                o = P.view(1, -1) * facts_C  # [1, hidden_size]
                # u_
                u_ = o + u
                new_hidden_state[i] = u_.squeeze(0)
            new_hidden_list.append(new_hidden_state)

        new_hidden_tuple = tuple(new_hidden_list)
        return new_hidden_tuple

    def evaluate(self,
                 dialog_encoder_inputs,  # LongTensor
                 dialog_encoder_inputs_length,
                 dialog_decoder_inputs,
                 batch_size=128
                 ):

        dialog_encoder_state = self.dialog_encoder.init_hidden(
            batch_size, self.device)

        '''dialog_encoder forward'''
        dialog_encoder_state, dialog_encoder_memory_bank = self.dialog_encoder(
            src=dialog_encoder_inputs,
            lengths=dialog_encoder_inputs_length,
            encoder_state=dialog_encoder_state,
        )

        '''dialog_decoder forward'''
        dialog_decoder_state = self.dialog_decoder.init_decoder_state(
            encoder_final=dialog_encoder_state)

        dialog_decoder_outputs = torch.ones((self.dialog_decoder_max_length,
                                             batch_size, self.dialog_decoder_hidden_size),
                                            device=self.device) * self.dialog_decoder_pad_id

        dialog_decoder_attns_std = torch.zeros((self.dialog_decoder_max_length,
                                                batch_size, self.dialog_decoder_max_length-2))

        dialog_decoder_input = dialog_decoder_inputs[0]
        for di in range(self.dialog_decoder_max_length):
            '''facts encoder forward'''
            new_hidden_tuple = self.facts_forward(facts_inputs,
                                                  dialog_decoder_state,
                                                  batch_size)
            dialog_decoder_state.update_state(new_hidden_tuple)
            dialog_decoder_state, dialog_decoder_output, \
                dialog_decoder_attn = self.dialog_decoder(
                    tgt=dialog_decoder_input.view(1, -1),
                    memory_bank=dialog_encoder_memory_bank,
                    state=dialog_decoder_state,
                    memory_lengths=dialog_encoder_inputs_length)
            dialog_decoder_output = dialog_decoder_output.detach().squeeze(0)
            dialog_decoder_outputs[di] = dialog_decoder_output
            dialog_decoder_attns_std[di] = dialog_decoder_attn['std'].squeeze(
                0)
            dialog_decoder_input = torch.argmax(dialog_decoder_output, dim=1)
            # greedy search

            if dialog_decoder_input[0].item() == self.dialog_decoder_eos_id:
                break

        # beam search  dialog_decoder_outputs -> [tgt_len x batch x hidden]
        dialog_decoder_outputs = self.dialog_decoder_linear(
            dialog_decoder_outputs)

        # log softmax
        dialog_decoder_outputs = self.dialog_decoder_softmax(
            dialog_decoder_outputs)

        return ((dialog_encoder_state, dialog_encoder_memory_bank),
                (dialog_decoder_state, dialog_decoder_outputs, dialog_decoder_attns_std))