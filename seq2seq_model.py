# -*- coding: utf-8 -*-

import math
import random
import torch
import torch.nn as nn

from modules.encoder import RNNEncoder
from modules.decoder import StdRNNDecoder


class Seq2SeqModel(nn.Module):
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
                 dialog_encoder_clipnorm=1.0,
                 dialog_encoder_clipvalue=0.5,
                 dialog_encoder_bidirectional=True,
                 dialog_encoder_embedding=None,
                 dialog_encoder_pad_id=0,
                 dialog_encoder_tied=True,

                 dialog_decoder_embedding_size=300,
                 dialog_decoder_vocab_size=None,
                 dialog_decoder_hidden_size=300,
                 dialog_decoder_num_layers=2,
                 dialog_decoder_rnn_type='LSTM',
                 dialog_decoder_dropout_rate=0.5,
                 dialog_decoder_clipnorm=1.0,
                 dialog_decoder_clipvalue=0.5,
                 dialog_decoder_max_length=32,
                 dialog_decoder_bidirectional=True,
                 dialog_decoder_embedding=None,
                 dialog_decoder_pad_id=0,
                 dialog_decoder_eos_id=3,
                 dialog_decoder_attention_type='dot',
                 dialog_decoder_tied=True,
                 device=None
                 ):
        # super init
        super(Seq2SeqModel, self).__init__()

        '''Dialog encoder parameters'''
        self.dialog_encoder_vocab_size = dialog_encoder_vocab_size
        self.dialog_encoder_embedding_size = dialog_encoder_embedding_size
        self.dialog_encoder_hidden_size = dialog_encoder_hidden_size
        self.dialog_encoder_num_layers = dialog_encoder_num_layers
        self.dialog_encoder_rnn_type = dialog_encoder_rnn_type
        self.dialog_encoder_dropout_rate = dialog_encoder_dropout_rate
        self.dialog_encoder_max_length = dialog_encoder_max_length
        self.dialog_encoder_clipnorm = dialog_encoder_clipnorm
        self.dialog_encoder_clipvalue = dialog_encoder_clipvalue
        self.dialog_encoder_bidirectional = dialog_encoder_bidirectional
        self.dialog_encoder_pad_id = dialog_encoder_pad_id
        self.dialog_encoder_tied = dialog_encoder_tied

        '''Dialog decoder parameters'''
        self.dialog_decoder_vocab_size = dialog_decoder_vocab_size
        self.dialog_decoder_embedding_size = dialog_decoder_embedding_size
        self.dialog_decoder_hidden_size = dialog_decoder_hidden_size
        self.dialog_decoder_num_layers = dialog_decoder_num_layers
        self.dialog_decoder_rnn_type = dialog_decoder_rnn_type
        self.dialog_decoder_dropout_rate = dialog_decoder_dropout_rate
        self.dialog_decoder_max_length = dialog_decoder_max_length
        self.dialog_decoder_clipnorm = dialog_decoder_clipnorm
        self.dialog_decoder_clipvalue = dialog_decoder_clipvalue
        self.dialog_decoder_bidirectional = dialog_decoder_bidirectional
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
            embedding=dialog_encoder_embedding
        )

        # Dialog Decoder with Attention
        self.dialog_decoder = StdRNNDecoder(
            rnn_type=self.dialog_decoder_rnn_type,
            bidirectional_encoder=self.dialog_decoder_bidirectional,
            num_layers=self.dialog_decoder_num_layers,
            hidden_size=self.dialog_decoder_hidden_size,
            dropout=self.dialog_decoder_dropout_rate,
            embedding=dialog_decoder_embedding,  # maybe replace by dialog_decoder_embedding
            attn_type=self.dialog_decoder_attention_type
        )

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
                dialog_decoder_inputs,
                dialog_decoder_inputs_length,
                dialog_decoder_targets,
                teacher_forcing_ratio=0.5,
                batch_size=128):

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
        # tgt, memory_bank, state, memory_lengths=None
        dialog_decoder_state = self.dialog_decoder.init_decoder_state(
            encoder_final=dialog_encoder_state)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        dialog_decoder_outputs = torch.ones((self.dialog_decoder_max_length,
                                             batch_size, self.dialog_decoder_hidden_size),
                                            device=self.device) * self.dialog_decoder_pad_id

        dialog_decoder_attns_std = torch.zeros((self.dialog_decoder_max_length,
                                                batch_size, self.dialog_decoder_max_length-2),
                                               device=self.device,
                                               dtype=torch.float)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.dialog_decoder_max_length):
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
                dialog_decoder_state, dialog_decoder_output, \
                    dialog_decoder_attn = self.dialog_decoder(
                        tgt=dialog_decoder_input.view(1, -1),
                        memory_bank=dialog_encoder_memory_bank,
                        state=dialog_decoder_state,
                        memory_lengths=dialog_encoder_inputs_length)
                dialog_decoder_outputs[di] = dialog_decoder_output.squeeze(0)
                dialog_decoder_attns_std[di] = dialog_decoder_attn['std'].squeeze(0)
                print('dialog_decoder_output shape: {}'.dialog_decoder_output.shape)
                dialog_decoder_input = torch.argmax(dialog_decoder_output, dim=2).detach()

                if dialog_decoder_input[0].item() == self.dialog_decoder_eos_id:
                    break

        """
        dialog_decoder_state, dialog_decoder_outputs, \
            dialog_decoder_attns_std = self.dialog_decoder(
                tgt=dialog_decoder_inputs,
                memory_bank=dialog_encoder_memory_bank,
                state=dialog_decoder_state,
                memory_lengths=dialog_encoder_inputs_length)

        """

        # beam search  dialog_decoder_outputs -> [tgt_len x batch x hidden]
        dialog_decoder_outputs = self.dialog_decoder_linear(
            dialog_decoder_outputs)

        # log softmax
        dialog_decoder_outputs = self.dialog_decoder_softmax(
            dialog_decoder_outputs)

        return ((dialog_encoder_state, dialog_encoder_memory_bank),
                (dialog_decoder_state, dialog_decoder_outputs, dialog_decoder_attns_std))

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
            dialog_decoder_state, dialog_decoder_output, \
                dialog_decoder_attn = self.dialog_decoder(
                    tgt=dialog_decoder_input.view(1, -1),
                    memory_bank=dialog_encoder_memory_bank,
                    state=dialog_decoder_state,
                    memory_lengths=dialog_encoder_inputs_length
                )
            dialog_decoder_outputs[di] = dialog_decoder_output.squeeze(0)
            dialog_decoder_attns_std[di] = dialog_decoder_attn['std'].squeeze(
                0)
            # greedy search
            dialog_decoder_input = torch.argmax(
                dialog_decoder_output).detach().view(1, -1)

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

    # beam search  tensor.numpy()
    def beam_search_decoder(self, memory_bank, beam_size):

        if isinstance(memory_bank, torch.Tensor):
            # memory_bank = memory_bank.numpy()
            if memory_bank.is_cuda:
                memory_bank = memory_bank.cpu()
            memory_bank = memory_bank.detach().numpy()

        sequences = [[list(), 1.0]]
        # walk over each step in sequence
        for row in memory_bank:
            print("row shape: {}".format(row.shape))
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score * (- math.log(row[j]))]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(
                all_candidates, key=lambda tup: tup[1], reverse=False)
            # ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            # select k best
            sequences = ordered[:beam_size]

        outputs = [sequence[0] for sequence in sequences]
        return outputs
