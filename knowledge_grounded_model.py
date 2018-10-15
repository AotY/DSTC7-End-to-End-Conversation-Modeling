# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import RNNEncoder
from modules.decoder import StdRNNDecoder
from modules.utils import init_lstm_orth, init_gru_orth

"""
KnowledgeGroundedModel
1. dialogue_encoder
2. facts_encoder
3. dialogue_decoder

"""


class KnowledgeGroundedModel(nn.Module):
    '''
    Knowledge-ground model
    Conditoning responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 dialogue_encoder_embedding_size=300,
                 dialogue_encoder_vocab_size=None,
                 dialogue_encoder_hidden_size=300,
                 dialogue_encoder_num_layers=2,
                 dialogue_encoder_rnn_type='LSTM',
                 dialogue_encoder_dropout_probability=0.5,
                 dialogue_encoder_max_length=32,
                 dialogue_encoder_clipnorm=50.0,
                 dialogue_encoder_bidirectional=True,
                 dialogue_encoder_embedding=None,

                 fact_embedding_size=300,
                 fact_vocab_size=None,
                 fact_dropout_probability=0.5,
                 fact_max_length=32,

                 dialogue_decoder_embedding_size=300,
                 dialogue_decoder_vocab_size=None,
                 dialogue_decoder_hidden_size=300,
                 dialogue_decoder_num_layers=2,
                 dialogue_decoder_rnn_type='LSTM',
                 dialogue_decoder_dropout_probability=0.5,
                 dialogue_decoder_clipnorm=50.0,
                 dialogue_decoder_max_length=32,
                 dialogue_decoder_embedding=None,
                 dialogue_decoder_pad_id=0,
                 dialogue_decoder_eos_id=3,
                 dialogue_decoder_attention_type='general',
                 dialogue_decoder_tied=True,
                 device=None
                 ):
        # super init
        super(KnowledgeGroundedModel, self).__init__()

        '''Dialog encoder parameters'''
        self.dialogue_encoder_vocab_size = dialogue_encoder_vocab_size
        self.dialogue_encoder_embedding_size = dialogue_encoder_embedding_size
        self.dialogue_encoder_hidden_size = dialogue_encoder_hidden_size
        self.dialogue_encoder_num_layers = dialogue_encoder_num_layers
        self.dialogue_encoder_rnn_type = dialogue_encoder_rnn_type
        self.dialogue_encoder_dropout_probability = dialogue_encoder_dropout_probability
        self.dialogue_encoder_max_length = dialogue_encoder_max_length
        self.dialogue_encoder_clipnorm = dialogue_encoder_clipnorm
        self.dialogue_encoder_bidirectional = dialogue_encoder_bidirectional

        '''facts encoder parameters'''
        self.fact_embedding_size = fact_embedding_size
        self.fact_vocab_size = fact_vocab_size
        self.fact_max_length = fact_max_length
        self.fact_dropout_probability = fact_dropout_probability

        '''Dialog decoder parameters'''
        self.dialogue_decoder_vocab_size = dialogue_decoder_vocab_size
        self.dialogue_decoder_embedding_size = dialogue_decoder_embedding_size
        self.dialogue_decoder_hidden_size = dialogue_decoder_hidden_size
        self.dialogue_decoder_num_layers = dialogue_decoder_num_layers
        self.dialogue_decoder_rnn_type = dialogue_decoder_rnn_type
        self.dialogue_decoder_dropout_probability = dialogue_decoder_dropout_probability
        self.dialogue_decoder_max_length = dialogue_decoder_max_length
        self.dialogue_decoder_clipnorm = dialogue_decoder_clipnorm
        self.dialogue_decoder_pad_id = dialogue_decoder_pad_id
        self.dialogue_decoder_eos_id = dialogue_decoder_eos_id
        self.dialogue_decoder_attention_type = dialogue_decoder_attention_type
        self.dialogue_decoder_tied = dialogue_decoder_tied

        self.device = device
        # num_embedding, embedding_dim
        ''''Ps: dialogue_encoder, facts_encoder, and dialogue_decoder may have different
            word embedding.
        '''
        # Dialog Encoder
        self.dialogue_encoder = RNNEncoder(
            rnn_type=self.dialogue_encoder_rnn_type,
            bidirectional=self.dialogue_encoder_bidirectional,
            num_layers=self.dialogue_encoder_num_layers,
            hidden_size=self.dialogue_encoder_hidden_size,
            dropout=self.dialogue_encoder_dropout_probability,
            embedding=dialogue_encoder_embedding)

        if self.dialogue_encoder_rnn_type == 'LSTM':
            init_lstm_orth(self.dialogue_encoder.rnn)
        elif self.dialogue_encoder_rnn_type == 'GRU':
            init_gru_orth(self.dialogue_encoder.rnn)

        # facts encoder
        # mi = A * ri    fact_linearA(300, 512)
        self.fact_linearA = nn.Linear(self.dialogue_decoder_hidden_size,
                                      self.dialogue_decoder_hidden_size)
        # ci = C * ri
        self.fact_linearC = nn.Linear(self.dialogue_decoder_hidden_size,
                                       self.dialogue_decoder_hidden_size)

        self.fact_decoder_linear = nn.Linear(
            self.fact_embedding_size, self.dialogue_decoder_hidden_size)

        # Dialog Decoder with Attention
        self.dialogue_decoder = StdRNNDecoder(
            rnn_type=self.dialogue_decoder_rnn_type,
            bidirectional_encoder=self.dialogue_encoder_bidirectional,
            num_layers=self.dialogue_decoder_num_layers,
            hidden_size=self.dialogue_decoder_hidden_size,
            dropout=self.dialogue_decoder_dropout_probability,
            embedding=dialogue_decoder_embedding,  # maybe replace by dialogue_decoder_embedding
            attn_type=self.dialogue_decoder_attention_type)

        if self.dialogue_decoder == 'LSTM':
            init_lstm_orth(self.dialogue_decoder.rnn)
        elif self.dialogue_encoder_rnn_type == 'GRU':
            init_gru_orth(self.dialogue_decoder.rnn)

        self.dialogue_decoder_linear = nn.Linear(
            self.dialogue_decoder_hidden_size, self.dialogue_decoder_vocab_size)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if self.dialogue_decoder_tied:
            if self.dialogue_decoder_embedding_size != self.dialogue_decoder_hidden_size:
                raise ValueError(
                    'When using the tied flag, hidden_size must be equal to embedding_size.')
            print('using tied..................')
            self.dialogue_decoder_linear.weight = dialogue_decoder_embedding.get_embedding_weight()

        self.dialogue_decoder_softmax = nn.LogSoftmax(dim=1)

    '''
    KnowledgeGroundedModel forward
    '''

    def forward(self,
                dialogue_encoder_inputs,  # LongTensor
                dialogue_encoder_inputs_length,
                facts_inputs,
                dialogue_decoder_inputs,
                fact_top_k=20,
                teacher_forcing_ratio=0.5,
                batch_size=128):
        """
        Args:
            - dialogue_encoder_inputs: [max_length, batch_size]
            - dialogue_encoder_inputs_length: [max_length, batch_size]
            - facts_inputs: [facts_size, fact_embedding_size]
            - dialogue_decoder_inputs: [max_length, batch_size]
        """
        # init, [-sqrt(3/hidden_size), sqrt(3/hidden_size)]

        dialogue_encoder_state = self.dialogue_encoder.init_hidden(
            batch_size, self.device)

        '''dialogue_encoder forward'''
        dialogue_encoder_state, dialogue_encoder_memory_bank = self.dialogue_encoder(
            src=dialogue_encoder_inputs,
            lengths=dialogue_encoder_inputs_length,
            encoder_state=dialogue_encoder_state,
        )

        # init decoder sate
        dialogue_decoder_state = self.dialogue_decoder.init_decoder_state(
            encoder_final=dialogue_encoder_state)

        '''facts encoder forward'''
        new_hidden_tuple = self.facts_forward(
            facts_inputs, dialogue_decoder_state.hidden, batch_size)
        dialogue_decoder_state.update_state(new_hidden_tuple)

        '''dialogue_decoder forward'''
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        dialogue_decoder_outputs = torch.ones((self.dialogue_decoder_max_length,
                                             batch_size, self.dialogue_decoder_hidden_size),
                                            device=self.device) * self.dialogue_decoder_pad_id

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.dialogue_decoder_max_length):
                dialogue_decoder_state, dialogue_decoder_output, \
                    dialogue_decoder_attn = self.dialogue_decoder(
                        tgt=dialogue_decoder_inputs[di].view(1, -1),
                        memory_bank=dialogue_encoder_memory_bank,
                        state=dialogue_decoder_state,
                        memory_lengths=dialogue_encoder_inputs_length)
                dialogue_decoder_outputs[di] = dialogue_decoder_output.squeeze(0)
        else:
            # Without teacher forcing: use its own predictions as the next
            # input
            dialogue_decoder_input = dialogue_decoder_inputs[0]
            for di in range(self.dialogue_decoder_max_length):
                dialogue_decoder_state, dialogue_decoder_output, \
                    dialogue_decoder_attn = self.dialogue_decoder(
                        tgt=dialogue_decoder_input.view(1, -1),
                        memory_bank=dialogue_encoder_memory_bank,
                        state=dialogue_decoder_state,
                        memory_lengths=dialogue_encoder_inputs_length)
                dialogue_decoder_output = dialogue_decoder_output.detach().squeeze(0)
                dialogue_decoder_outputs[di] = dialogue_decoder_output
                dialogue_decoder_input = torch.argmax(
                    dialogue_decoder_output, dim=1)

                if dialogue_decoder_input[0].item() == self.dialogue_decoder_eos_id:
                    break

                    # beam search  dialogue_decoder_outputs -> [tgt_len x batch x hidden]
        dialogue_decoder_outputs = self.dialogue_decoder_linear(
            dialogue_decoder_outputs)

        # log softmax
        dialogue_decoder_outputs = self.dialogue_decoder_softmax(
            dialogue_decoder_outputs)

        return ((dialogue_encoder_state, dialogue_encoder_memory_bank),
                (dialogue_decoder_state, dialogue_decoder_outputs))

    def facts_forward(self,
                      facts_inputs,
                      hidden_tuple,
                      batch_size):
        """
        Args:
            - facts_inputs: [batch_size, top_k, embedding_size]
            - hidden_tuple: ([num_layers, batch_size, hidden_size], [])
            - batch_size
        """

        # [batch_size, topk, embedding_size] -> [batch_size, topk, hidden_size]
        if self.fact_embedding_size != self.dialogue_decoder_hidden_size:
                facts_inputs = self.fact_decoder_linear(facts_inputs)

        # M [batch_size, topk, hidden_size]
        fact_M = self.fact_linearA(facts_inputs)
        # C [batch_size, topk, hidden_size]
        fact_C = self.fact_linearC(facts_inputs)

        # hidden_tuple is a tuple object
        new_hidden_list = []
        for hidden_state in hidden_tuple[:1]:
            new_hidden_state = torch.zeros((self.dialogue_decoder_num_layers,
                                            batch_size,
                                            self.dialogue_decoder_hidden_size),
                                           device=self.device)
            for i in range(self.dialogue_decoder_num_layers):
                u = hidden_state[i]  # [batch_size, hidden_size]
                # batch product
                tmpP = torch.bmm(facts_inputs, u.unsqueeze(2))  # [batch_size, top_k, 1]
                P = F.softmax(tmpP.squeeze(2), dim=1)  # [batch_size, top_k]
                # [batch_size, hidden_size, 1]
                o = torch.bmm(facts_inputs.transpose(1, 2), P.unsqueeze(2))
                u_ = o.squeeze(2) + u  # [batch_size, hidden_size]
                new_hidden_state[i] = u_
                # new_hidden_state -> [num_layers, batch_size, hidden_size]
            new_hidden_list.append(new_hidden_state)

        new_hidden_tuple = tuple(new_hidden_list)
        return new_hidden_tuple

    def evaluate(self,
                 dialogue_encoder_inputs,  # LongTensor
                 dialogue_encoder_inputs_length,
                 facts_inputs,
                 dialogue_decoder_inputs,
                 batch_size=128):

        dialogue_encoder_state = self.dialogue_encoder.init_hidden(
            batch_size, self.device)

        '''dialogue_encoder forward'''
        dialogue_encoder_state, dialogue_encoder_memory_bank = self.dialogue_encoder(
            src=dialogue_encoder_inputs,
            lengths=dialogue_encoder_inputs_length,
            encoder_state=dialogue_encoder_state,
        )

        dialogue_decoder_state = self.dialogue_decoder.init_decoder_state(
            encoder_final=dialogue_encoder_state)

        '''facts encoder forward'''
        new_hidden_tuple = self.facts_forward(
            facts_inputs, dialogue_decoder_state.hidden, batch_size)
        dialogue_decoder_state.update_state(new_hidden_tuple)

        '''dialogue_decoder forward'''
        dialogue_decoder_outputs = torch.ones((self.dialogue_decoder_max_length,
                                             batch_size, self.dialogue_decoder_hidden_size),
                                            device=self.device) * self.dialogue_decoder_pad_id

        dialogue_decoder_input = dialogue_decoder_inputs[0]
        for di in range(self.dialogue_decoder_max_length):
            dialogue_decoder_state, dialogue_decoder_output, \
                dialogue_decoder_attn = self.dialogue_decoder(
                    tgt=dialogue_decoder_input.view(1, -1),
                    memory_bank=dialogue_encoder_memory_bank,
                    state=dialogue_decoder_state,
                    memory_lengths=dialogue_encoder_inputs_length)
            dialogue_decoder_output = dialogue_decoder_output.detach().squeeze(0)
            dialogue_decoder_outputs[di] = dialogue_decoder_output
            dialogue_decoder_input = torch.argmax(dialogue_decoder_output, dim=1)
            # greedy search

            if dialogue_decoder_input[0].item() == self.dialogue_decoder_eos_id:
                break

        # beam search  dialogue_decoder_outputs -> [tgt_len x batch x hidden]
        dialogue_decoder_outputs = self.dialogue_decoder_linear(
            dialogue_decoder_outputs)

        # log softmax
        dialogue_decoder_outputs = self.dialogue_decoder_softmax(
            dialogue_decoder_outputs)

        return ((dialogue_encoder_state, dialogue_encoder_memory_bank),
                (dialogue_decoder_state, dialogue_decoder_outputs))
