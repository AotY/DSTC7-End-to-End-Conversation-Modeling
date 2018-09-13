# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from modules.Encoder import RNNEncoder
from modules.Decoder import StdRNNDecoder


class KnowledgeGroundedModel(nn.Module):
    '''
    Knowledge-ground model
    Conditoning responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 dialog_encoder_vocab_size=None,
                 dialog_encoder_hidden_size=300,
                 dialog_encoder_num_layers=2,
                 dialog_encoder_rnn_type='LSTM',
                 dialog_encoder_dropout_rate=0.5,
                 dialog_encoder_max_length=32,
                 dialog_encoder_rnn_units=512,
                 dialog_encoder_clip_grads=1.0,
                 dialog_encoder_bidirectional=True,
                 dialog_encoder_pretrained_embedding_weight=None,

                 dialog_decoder_vocab_size=None,
                 dialog_decoder_hidden_size=300,
                 dialog_decoder_num_layers=2,
                 dialog_decoder_rnn_type='LSTM',
                 dialog_decoder_dropout_rate=0.5,
                 dialog_decoder_max_length=32,
                 dialog_decoder_rnn_units=512,
                 dialog_decoder_clip_grads=1.0,
                 dialog_decoder_bidirectional=True,
                 dialog_decoder_pretrained_embedding_weight=None,

                 facts_encoder_vocab_size=None,
                 facts_encoder_hidden_size=300,
                 facts_encoder_num_layers=2,
                 facts_encoder_rnn_type='LSTM',
                 facts_encoder_dropout_rate=0.5,
                 facts_encoder_max_length=32,
                 facts_encoder_rnn_units=512,
                 facts_encoder_clip_grads=1.0,
                 facts_encoder_bidirectional=True,
                 facts_encoder_pretrained_embedding_weight=None,
                 ):
        super(KnowledgeGroundedModel, self).__init__()

        # init


        '''Dialog encoder parameters'''
        self.dialog_encoder_vocab_size = dialog_encoder_vocab_size
        self.dialog_encoder_hidden_size = dialog_encoder_hidden_size
        self.dialog_encoder_num_layers = dialog_encoder_num_layers
        self.dialog_encoder_rnn_type = dialog_encoder_rnn_type
        self.dialog_encoder_dropout_rate = dialog_encoder_dropout_rate
        self.dialog_encoder_max_length = dialog_encoder_max_length
        self.dialog_encoder_rnn_units = dialog_encoder_rnn_units
        self.dialog_encoder_clip_grads = dialog_encoder_clip_grads
        self.dialog_encoder_bidirectional = dialog_encoder_bidirectional
        self.dialog_encoder_pretrained_embedding_weight = dialog_encoder_pretrained_embedding_weight

        '''Dialog decoder parameters'''
        self.dialog_decoder_vocab_size = dialog_decoder_vocab_size
        self.dialog_decoder_hidden_size = dialog_decoder_hidden_size
        self.dialog_decoder_num_layers = dialog_decoder_num_layers
        self.dialog_decoder_rnn_type = dialog_decoder_rnn_type
        self.dialog_decoder_dropout_rate = dialog_decoder_dropout_rate
        self.dialog_decoder_max_length = dialog_decoder_max_length
        self.dialog_decoder_rnn_units = dialog_decoder_rnn_units
        self.dialog_decoder_clip_grads = dialog_decoder_clip_grads
        self.dialog_decoder_bidirectional = dialog_decoder_bidirectional
        self.dialog_decoder_pretrained_embedding_weight = dialog_decoder_pretrained_embedding_weight

        '''Facts encoder parameters'''
        self.facts_encoder_vocab_size = facts_encoder_vocab_size
        self.facts_encoder_hidden_size = facts_encoder_hidden_size
        self.facts_encoder_num_layers = facts_encoder_num_layers
        self.facts_encoder_rnn_type = facts_encoder_rnn_type
        self.facts_encoder_dropout_rate = facts_encoder_dropout_rate
        self.facts_encoder_max_length = facts_encoder_max_length
        self.facts_encoder_rnn_units = facts_encoder_rnn_units
        self.facts_encoder_clip_grads = facts_encoder_clip_grads
        self.facts_encoder_bidirectional = facts_encoder_bidirectional
        self.facts_encoder_pretrained_embedding_weight = facts_encoder_pretrained_embedding_weight

        # num_embeddings, embedding_dim
        ''''Ps: dialog_encoder, facts_encoder, and dialog_decoder may have different
            word embeddings.
        '''
        self.dialog_encoder_embedding = nn.Embedding(self.dialog_encoder_vocab_size, self.dialog_encoder_hidden_size)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.dialog_encoder_embedding.weight.data.copy_(
            torch.from_numpy(self.dialog_encoder_pretrained_embedding_weight))

        '''
        self.dialog_decoder_embedding = nn.Embedding(self.dialog_decoder_vocab_size, self.dialog_decoder_hidden_size)
        self.dialog_decoder_embedding.weight.data.copy_(
            torch.from_numpy(self.dialog_decoder_pretrained_embedding_weight))
        
        self.facts_encoder_embedding = nn.Embedding(self.facts_encoder_vocab_size, self.facts_encoder_hidden_size)
        self.facts_encoder_embedding.weight.data.copy_(
            torch.from_numpy(self.facts_encoder_pretrained_embedding_weight))
        
        '''

        # Dialog Enocder
        self.dialog_encoder = RNNEncoder(
            rnn_type=self.dialog_encoder_rnn_type,
            bidirectional=self.dialog_encoder_bidirectional,
            num_layers=self.dialog_encoder_num_layers,
            hidden_size=self.dialog_encoder_hidden_size,
            dropout=self.dialog_encoder_dropout_rate,
            embeddings=self.dialog_encoder_embedding
        )

        # Search facts, Facts Encoder
        self.facts_encoder = StdRNNDecoder(
            rnn_type=self.facts_encoder_rnn_type,
            bidirectional_encoder=self.facts_encoder_bidirectional,
            num_layers=self.facts_encoder_num_layers,
            hidden_size=self.facts_encoder_hidden_size,
            dropout=self.facts_encoder_dropout_rate,
            embeddings=self.dialog_encoder_embedding  # maybe replace by facts_encoder_embedding
        )

        # Dialog Decoder with Attention
        # rnn_type,
        # bidirectional_encoder, num_layers,
        # hidden_size, attn_type = None,
        # dropout = 0.0, embeddings = None
        self.dialog_decoder = StdRNNDecoder(
            rnn_type=self.dialog_decoder_rnn_type,
            bidirectional_encoder=self.dialog_decoder_bidirectional,
            num_layers=self.dialog_decoder_num_layers,
            hidden_size=self.dialog_decoder_hidden_size,
            dropout=self.dialog_decoder_dropout_rate,
            embeddings=self.dialog_encoder_embedding  # maybe replace by dialog_decoder_embedding
        )

    '''
    
    '''

    def forward(self,
                dialog_encoder_src,  # LongTensor
                dialog_encoder_src_lengths,
                dialog_encoder_initial_state,

                facts_encoder_tgt,
                facts_encoder_initial_state,

                dialog_decoder_tgt,

                ):

        '''dialog_encoder forward'''
        dialog_encoder_final_state, dialog_encoder_memory_bank = self.dialog_encoder.forward(
            dialog_encoder_src,
            lengths=dialog_encoder_src_lengths,
            encoder_state=dialog_encoder_initial_state,  # the source memory_bank lengths.
        )

        '''facts_encoder forward'''
        facts_encoder_memory_bank, facts_encoder_final_stae, facts_encoder_attns = self.facts_encoder.forward(
            tgt=facts_encoder_tgt,  # sequences of padded tokens
            memory_bank=dialog_encoder_memory_bank,
            state=facts_encoder_initial_state,
            memory_lengths=dialog_encoder_src_lengths,  # # the source memory_bank lengths.
        )

        '''concat dialog_encoder memeory bank and fatcs_encoder memeory bank'''
        add_encoder_memory_bank = torch.Tensor.add(dialog_encoder_memory_bank, facts_encoder_memory_bank)
        add_encoder_final_state = torch.Tensor.add(dialog_encoder_initial_state, facts_encoder_final_stae)

        '''dialog_decoder forward'''
        # tgt, memory_bank, state, memory_lengths=None
        dialog_decoder_memory_bank, dialog_decoder_final_stae, dialog_decoder_attns = self.dialog_decoder.forward(
            tgt=dialog_decoder_tgt,
            memory_bank=add_encoder_memory_bank,
            state=add_encoder_final_state,
            memory_lengths=dialog_encoder_src_lengths
        )


        return (
            (dialog_encoder_final_state, dialog_encoder_memory_bank),
            (facts_encoder_memory_bank, facts_encoder_final_stae, facts_encoder_attns),
            (dialog_decoder_memory_bank, dialog_decoder_final_stae, dialog_decoder_attns)
        )


