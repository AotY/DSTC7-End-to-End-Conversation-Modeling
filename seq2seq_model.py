# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn

from modules.Encoder import RNNEncoder
from modules.Decoder import StdRNNDecoder
from modules.Embeddings import Embedding


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
                 dialog_encoder_pretrained_embedding_weight=None,
                 dialog_encoder_pad_id=1,
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
                 dialog_decoder_pretrained_embedding_weight=None,
                 dialog_decoder_pad_id=1,
                 dialog_decoder_attention_type='dot',
                 dialog_decoder_tied=True,
                 use_gpu=True
                 ):
        super(Seq2SeqModel, self).__init__()

        # init

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
        self.dialog_encoder_pretrained_embedding_weight = dialog_encoder_pretrained_embedding_weight
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
        self.dialog_decoder_pretrained_embedding_weight = dialog_decoder_pretrained_embedding_weight
        self.dialog_decoder_pad_id = dialog_decoder_pad_id
        self.dialog_decoder_attention_type = dialog_decoder_attention_type
        self.dialog_decoder_tied= dialog_decoder_tied


        self.use_gpu = use_gpu
        # num_embedding, embedding_dim
        ''''Ps: dialog_encoder, facts_encoder, and dialog_decoder may have different
            word embedding.
        '''
        # self.dialog_encoder_embedding = nn.Embedding(self.dialog_encoder_vocab_size + 1, self.dialog_encoder_hidden_size,)

        self.dialog_encoder_embedding = Embedding(embedding_size=self.dialog_encoder_embedding_size,
                                                   vocab_size=self.dialog_encoder_vocab_size,
                                                   padding_idx=self.dialog_encoder_pad_id,
                                                   dropout_ratio=self.dialog_encoder_dropout_rate)

        self.dialog_decoder_embedding = Embedding(embedding_size=self.dialog_decoder_embedding_size,
                                                   vocab_size=self.dialog_decoder_vocab_size,
                                                   padding_idx=self.dialog_decoder_pad_id,
                                                   dropout_ratio=self.dialog_decoder_dropout_rate)

        if self.dialog_encoder_pretrained_embedding_weight is not None:
            # pretrained_weight is a numpy matrix of shape (num_embedding, embedding_dim)
            self.dialog_encoder_embedding.set_pretrained_embedding(self.dialog_encoder_pretrained_embedding_weight, fixed=False)

            #  self.dialog_decoder_embedding = self.dialog_encoder_embedding
            self.dialog_decoder_embedding.set_pretrained_embedding(self.dialog_decoder_pretrained_embedding_weight, fixed=False)

        # Dialog Enocder
        self.dialog_encoder = RNNEncoder(
            rnn_type=self.dialog_encoder_rnn_type,
            bidirectional=self.dialog_encoder_bidirectional,
            num_layers=self.dialog_encoder_num_layers,
            hidden_size=self.dialog_encoder_hidden_size,
            dropout=self.dialog_encoder_dropout_rate,
            embedding=self.dialog_encoder_embedding
        )

        # Dialog Decoder with Attention
        # rnn_type,
        # bidirectional_encoder, num_layers,
        # hidden_size, attn_type = None,
        # dropout = 0.0, embedding = None
        self.dialog_decoder = StdRNNDecoder(
            rnn_type=self.dialog_decoder_rnn_type,
            bidirectional_encoder=self.dialog_decoder_bidirectional,
            num_layers=self.dialog_decoder_num_layers,
            hidden_size=self.dialog_decoder_hidden_size,
            dropout=self.dialog_decoder_dropout_rate,
            embedding=self.dialog_decoder_embedding,  # maybe replace by dialog_decoder_embedding
            attn_type=self.dialog_decoder_attention_type
        )

        self.dialog_decoder_linear = nn.Linear(self.dialog_decoder_hidden_size, self.dialog_decoder_vocab_size)
        # self.dialog_decoder_softmax = nn.LogSoftmax(dim=1)
        self.dialog_decoder_softmax = nn.Softmax(dim=1)

    '''
    Seq2SeqModel forward
    '''

    def forward(self,
                dialog_encoder_src,  # LongTensor
                dialog_encoder_src_lengths,
                dialog_decoder_tgt,
                dialog_decoder_tgt_lengths,
                use_teacher_forcing=True,
                teacher_forcing_ratio=0.5,
                batch_size=128
                ):

        # init, [-sqrt(3/hidden_size), sqrt(3/hidden_size)]
        print("dialog_encoder_src shape : {}".format(dialog_encoder_src.shape))

        dialog_encoder_initial_state = self.dialog_encoder.init_hidden(batch_size, self.use_gpu)
        print("dialog_encoder_initial_state[0] shape: {} ".format(dialog_encoder_initial_state[0].shape))

        print("dialog_encoder_src size : {}".format(dialog_encoder_src.shape))
        print("dialog_encoder_src_lengths size : {}".format(dialog_encoder_src_lengths.shape))


        '''dialog_encoder forward'''
        dialog_encoder_final_state, dialog_encoder_memory_bank = self.dialog_encoder.forward(
            src=dialog_encoder_src,
            lengths=dialog_encoder_src_lengths,
            encoder_state=dialog_encoder_initial_state,  # the source memory_bank lengths.
        )

        print('dialog_encoder_final_state[0] shape: {}'.format(dialog_encoder_final_state[0].shape))
        print('dialog_encoder_memory_bank shape: {}'.format(dialog_encoder_memory_bank.shape))

        '''dialog_decoder forward'''
        # tgt, memory_bank, state, memory_lengths=None
        # decoder_state = RNNDecoderState(self.dialog_decoder_hidden_size, dialog_encoder_final_state)
        decoder_state = self.dialog_decoder.init_decoder_state(encoder_final=dialog_encoder_final_state)


        print ("tgt shape : {}".format(dialog_decoder_tgt.shape))
        print("tgt: {}".format(dialog_decoder_tgt))

        print('dialog_encoder_src_lengths: {}'.format(dialog_encoder_src_lengths))
        print('dialog_decoder_tgt_lengths: {}'.format(dialog_decoder_tgt_lengths))

        dialog_decoder_memory_bank, dialog_decoder_final_state, \
        dialog_decoder_attns = self.dialog_decoder.forward(
            tgt=dialog_decoder_tgt,
            memory_bank=dialog_encoder_memory_bank,
            state=decoder_state,
            memory_lengths=dialog_encoder_src_lengths # dialog_decoder_tgt_lengths
        )


        # beam search  dialog_decoder_memory_bank -> [tgt_len x batch x hidden]
        # batch_size = dialog_decoder_memory_bank.shape[1]
        # dialog_decoder_outputs [tgt_len x batch x beam_size]?
        # dialog_decoder_outputs = torch.zeros((self.dialog_decoder_max_length, batch_size, self.dialog_decoder_hidden_size))
        # dialog_decoder_outputs = torch.zeros((self.dialog_decoder_max_length, batch_size, self.dialog_decoder_vocab_size))
        print("dialog_decoder_memory_bank shape: {}".format(dialog_decoder_memory_bank.shape))
        dialog_decoder_memory_bank = dialog_decoder_memory_bank[0]
        print("dialog_decoder_memory_bank shape: {}".format(dialog_decoder_memory_bank.shape))

        dialog_decoder_outputs = self.dialog_decoder_softmax(self.dialog_decoder_linear(dialog_decoder_memory_bank))
        print("dialog_decoder_outputs shape: {}".format(dialog_decoder_outputs.shape))

        '''
        beam_size = 100
        # dialog_decoder_outputs = torch.zeros((self.dialog_decoder_max_length, batch_size, beam_size))
        # 1. first strategy: use top1. 2, use beam search strategy
        dialog_decoder_outputs = torch.zeros((self.dialog_decoder_max_length, batch_size))


        for batch_index in range(batch_size):
            dialog_decoder_output = dialog_decoder_memory_bank[:, batch_index, :]
            print("dialog_decoder_output shape: {}".format(dialog_decoder_output.shape))
            decoder_linear_output = self.dialog_decoder_linear(dialog_decoder_memory_bank[:, batch_index, :])
            print("decoder_linear_output shape: {}".format(decoder_linear_output.shape))
            decoder_linear_output = self.dialog_decoder_softmax(decoder_linear_output)
            print("decoder_linear_output shape: {}".format(decoder_linear_output.shape))

            topv, topi = decoder_linear_output.topk(1)
            print("topv shape: {}".format(topv.shape))
            print("topi shape: {}".format(topi.shape))

            # beam_search_output = self.beam_search_decoder(decoder_linear_output, beam_size=beam_size)
            # top 1
            # beam_search_output = beam_search_output[0]
            # print("beam_search_output shape: {}".format(beam_search_output.shape))

            # beam_search_output = torch.Tensor(beam_search_output).view(-1)
            # dialog_decoder_outputs[:, batch_index] = beam_search_output
            dialog_decoder_outputs[:, batch_index] = topi.squeeze().detach()
        '''

        return ((dialog_encoder_final_state, dialog_encoder_memory_bank),
                (dialog_decoder_memory_bank, dialog_decoder_final_state, dialog_decoder_attns, dialog_decoder_outputs))

    def set_cuda(self):
        self.dialog_encoder.cuda()
        self.dialog_decoder.cuda()
        self.dialog_decoder_linear.cuda()
        self.dialog_decoder_softmax.cuda()

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
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=False)
            # ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            # select k best
            sequences = ordered[:beam_size]

        outputs = [sequence[0] for sequence in sequences]
        return outputs
