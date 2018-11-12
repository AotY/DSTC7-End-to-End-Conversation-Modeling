# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import Encoder
from modules.simple_encoder import SimpleEncoder
import modules.transformer as transformer
from modules.session_encoder import SessionEncoder
#  from modules.session_decoder import SessionDecoder
from modules.decoder import Decoder
from modules.reduce_state import ReduceState
from modules.bahdanau_attn_decoder import BahdanauAttnDecoder
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.utils import init_lstm_orth, init_gru_orth
from modules.utils import init_linear_wt, init_wt_normal
from modules.utils import sequence_mask
#  from modules.beam_search import beam_decode
from modules.beam_search_original import beam_decode

"""
KGModel
"""


class KGModel(nn.Module):
    '''
    generating responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 model_type,
                 vocab_size,
                 c_max_len,
                 pre_embedding_size,
                 embedding_size,
                 share_embedding,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 encoder_num_layers,
                 decoder_num_layers,
                 bidirectional,
                 turn_num,
                 turn_type,
                 decoder_type,
                 attn_type,
                 dropout,
                 padding_idx,
                 tied,
                 device,
                 pre_trained_weight=None):
        super(KGModel, self).__init__()

        self.vocab_size = vocab_size
        self.model_type = model_type
        self.embedding_size = embedding_size
        self.pre_embedding_size = pre_embedding_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.turn_num = turn_num
        self.turn_type = turn_type
        self.decoder_type = decoder_type
        self.device = device

        encoder_embedding = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx
        )

        if pre_trained_weight is not None:
            encoder_embedding.weight.data.copy_(pre_trained_weight)
        else:
            init_wt_normal(encoder_embedding.weight)

        # h_encoder
        if turn_type != 'transformer':
            self.simple_encoder = SimpleEncoder(vocab_size,
                                                encoder_embedding,
                                                rnn_type,
                                                hidden_size,
                                                encoder_num_layers,
                                                bidirectional,
                                                dropout)
        else:
            self.transformer_encoder = transformer.models.Encoder(
                vocab_size,
                c_max_len,
                embedding_size,
                6,
                8,
                64,
                64,
                hidden_size,
                1024,
                padding_idx,
                dropout
            )

        if turn_type != 'none' or turn_type != 'concat':
            if turn_type == 'c_concat':
                self.c_concat_linear = nn.Linear(hidden_size * turn_num, hidden_size)
                init_linear_wt(self.c_concat_linear)
            elif turn_type == 'sequential' or turn_type == 'weight':
                self.session_encoder = SessionEncoder(
                    rnn_type,
                    hidden_size,
                    num_layers,
                    bidirectional,
                    dropout,
                )
            else:
                pass

        # fact encoder
        if model_type == 'kg':
            # mi = A * ri    fact_linearA(300, 512)
            self.fact_linearA = nn.Linear(hidden_size, hidden_size)
            init_linear_wt(self.fact_linearA)
            # ci = C * ri
            self.fact_linearC = nn.Linear(hidden_size, hidden_size)
            init_linear_wt(self.fact_linearC)

            if pre_embedding_size != hidden_size:
                self.fact_linear = nn.Linear(pre_embedding_size, hidden_size)
                init_linear_wt(self.fact_linear)

        # encoder hidden_state -> decoder hidden_state
        self.reduce_state = ReduceState(rnn_type)

        if share_embedding:
            decoder_embedding = encoder_embedding
        else:
            decoder_embedding = nn.Embedding(
                vocab_size,
                embedding_size,
                padding_idx
            )
            if pre_trained_weight is not None:
                decoder_embedding.weight.data.copy_(pre_trained_weight)
            else:
                init_wt_normal(decoder_embedding.weight)

        self.decoder = self.build_decoder(
            decoder_type,
            vocab_size,
            decoder_embedding,
            rnn_type,
            hidden_size,
            num_layers,
            dropout,
            tied,
            turn_type,
            attn_type,
            device
        )

    def forward(self,
                h_inputs,
                h_turns_length,
                h_inputs_lengths,
                h_inputs_position,
                decoder_inputs,
                f_inputs,
                f_inputs_length,
                batch_size,
                r_max_len,
                teacher_forcing_ratio):
        '''
        input:
            h_inputs: # [max_len, batch_size, turn_num]
            h_turns_length: [batch_size]
            h_inputs_lengths: [batch_size, turn_num]

            decoder_inputs: [r_max_len, batch_size], first step: [sos * batch_size]

            f_inputs: [batch_size, r_max_len, topk]
            f_inputs_length: [batch_size]
        '''
        h_encoder_outputs, h_encoder_hidden_state, h_decoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_lengths,
            h_inputs_position,
            batch_size
        )

        if h_encoder_hidden_state is None:
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_inputs,
                                                  f_inputs_length,
                                                  decoder_hidden_state,
                                                  batch_size)

        # decoder
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_outputs, decoder_hidden_state, attn_weights = self.decoder(decoder_inputs,
                                                                               decoder_hidden_state,
                                                                               h_encoder_outputs,
                                                                               h_decoder_lengths)
            return decoder_outputs

        else:
            decoder_outputs = []
            decoder_input = decoder_inputs[0].view(1, -1)
            for i in range(r_max_len):
                decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                  decoder_hidden_state,
                                                                                  h_encoder_outputs,
                                                                                  h_decoder_lengths)

                decoder_outputs.append(decoder_output)
                decoder_input = torch.argmax(
                    decoder_output, dim=2).detach().view(1, -1)

            # [r_max_len, batch_size, vocab_size]
            decoder_outputs = torch.cat(decoder_outputs, dim=0)

            return decoder_outputs

    '''evaluate'''

    def evaluate(self,
                 h_inputs,
                 h_turns_length,
                 h_inputs_lengths,
                 h_inputs_position,
                 decoder_input,
                 f_inputs,
                 f_inputs_length,
                 r_max_len,
                 batch_size):
        '''
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        h_encoder_outputs, h_encoder_hidden_state, h_decoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_lengths,
            h_inputs_position,
            batch_size
        )

        if h_encoder_hidden_state is None:
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_inputs,
                                                  f_inputs_length,
                                                  decoder_hidden_state,
                                                  batch_size)

        # decoder
        decoder_outputs = []
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, _ = self.decoder(decoder_input,
                                                                   decoder_hidden_state,
                                                                   h_encoder_outputs,
                                                                   h_decoder_lengths)

            decoder_input = torch.argmax(
                decoder_output, dim=2).detach()  # [1, batch_size]
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        return decoder_outputs

    '''decode'''

    def decode(self,
               h_inputs,
               h_turns_length,
               h_inputs_lengths,
               h_inputs_position,
               decoder_input,
               f_inputs,
               f_inputs_length,
               decode_type,
               r_max_len,
               eosid,
               batch_size,
               beam_width,
               best_n):

        h_encoder_outputs, h_encoder_hidden_state, h_decoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_lengths,
            h_inputs_position,
            batch_size
        )

        if h_encoder_hidden_state is None:
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_inputs,
                                                  f_inputs_length,
                                                  decoder_hidden_state,
                                                  batch_size)

        # decoder
        greedy_outputs = None
        beam_outputs = None
        #  if decode_type == 'greedy':
        greedy_outputs = []
        input = decoder_input
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, attn_weights = self.decoder(input,
                                                                              decoder_hidden_state,
                                                                              h_encoder_outputs,
                                                                              h_decoder_lengths)

            input = torch.argmax(
                decoder_output, dim=2).detach()  # [1, batch_size]
            greedy_outputs.append(input)

            if input[0][0].item() == eosid:
                break

        greedy_outputs = torch.cat(greedy_outputs, dim=0)
        # [len, batch_size]  -> [batch_size, len]
        greedy_outputs.transpose_(0, 1)

        #  elif decode_type == 'beam_search':
        input = decoder_input
        beam_outputs = beam_decode(
            self.decoder,
            h_encoder_outputs,
            h_encoder_inputs_length,
            decoder_hidden_state,
            input,
            batch_size,
            beam_width,
            best_n,
            eosid,
            r_max_len,
            self.vocab_size,
            self.device
        )

        return greedy_outputs, beam_outputs

    def h_forward(self,
                  h_inputs,
                  h_turns_length,
                  h_inputs_lengths,
                  h_inputs_position,
                  batch_size):
        """history forward
        Args:
            h_inputs: # [max_len, batch_size, turn_num]
            h_turns_length: [batch_size]
            h_inputs_lengths: [batch_size, turn_num]
        turn_type:
        """
        if self.turn_type == 'concat' or self.turn_type == 'none':
            inputs = h_inputs[:, :, 0] # [max_len, batch_size]
            inputs_length = h_inputs_lengths[:, 0]

            # [max_len, batch_size, hidden_size]
            outputs, hidden_state = self.simple_encoder(inputs, inputs_length)
            return outputs, hidden_state, inputs_length
        else:
            stack_outputs = []
            #  stack_hidden_states = []
            for ti in range(self.turn_num):
                inputs = h_inputs[:, :, ti] # [max_len, batch_size]
                if self.turn_type == 'transformer':
                    inputs_position = h_inputs_position[:, :, ti]
                    outputs = self.transformer_encoder(inputs, inputs_position)
                    print('transformer: ', outputs.shape)
                else:
                    inputs_length = h_inputs_lengths[:, ti] # [batch_size]
                    # [max_len, batch_size, hidden_size] , [num_layers * bidirection_num, batch_size, hidden_size]
                    outputs, hidden_state = self.simple_encoder(inputs, inputs_length)
                stack_outputs.append(outputs[-1].unsqueeze(0))

            if self.turn_type == 'sum':
                stack_outputs = torch.cat(stack_outputs, dim=0) # [turn_num, batch_size, hidden_size]
                return stack_outputs.sum(dim=0).unsqueeze(0), None, None # [1, batch_size, hidden_size]
            elif self.turn_type == 'c_concat':
                c_concat_outputs = torch.cat(stack_outputs, dim=2) # [1, hidden_size * turn_num]
                return self.c_concat_linear(c_concat_outputs), None, None # [1, batch_size, hidden_size]
            elif self.turn_type == 'sequential':
                stack_outputs = torch.cat(stack_outputs, dim=0) # [turn_num, batch_size, hidden_size]
                session_outputs, session_hidden_state = self.session_encoder(stack_outputs, h_inputs_lengths) # [1, batch_size, hidden_size]
                return session_outputs[1].unsqueeze(0), session_hidden_state, h_turns_length
            elif self.turn_type == 'weight':
                stack_outputs = torch.cat(stack_outputs, dim=0) # [turn_num, batch_size, hidden_size]
                session_outputs, session_hidden_state = self.session_encoder(stack_outputs, h_inputs_lengths) # [1, batch_size, hidden_size]
                return session_outputs, session_hidden_state, h_turns_length
            elif self.turn_type == 'hran':
                # TODO self attention
                pass


    def f_forward(self,
                  f_inputs,
                  f_inputs_length,
                  hidden_state,
                  batch_size):
        """
        Args:
            - f_inputs: [batch_size, top_k, embedding_size]
            - hidden_state: [num_layers, batch_size, hidden_size]
            - batch_size
        """

        # [batch_size, topk, embedding_size] -> [batch_size, topk, hidden_size]
        if self.pre_embedding_size != self.hidden_size:
            f_inputs = self.fact_linear(f_inputs)

        # M [batch_size, topk, hidden_size]
        fact_M = self.fact_linearA(f_inputs)

        # C [batch_size, topk, hidden_size]
        fact_C = self.fact_linearC(f_inputs)

        # [batch_size, num_layers, topk]
        tmpP = torch.bmm(hidden_state.transpose(0, 1), fact_M.transpose(1, 2))

        mask = sequence_mask(f_inputs_length, max_len=tmpP.size(-1))
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        tmpP.masked_fill_(1 - mask, -float('inf'))

        P = F.softmax(tmpP, dim=2)

        o = torch.bmm(P, fact_C)  # [batch_size, num_layers, hidden_size]
        u_ = torch.add(o, hidden_state.transpose(0, 1))

        # [num_layers, batch_size, hidden_size]
        u_ = u_.transpose(0, 1).contiguous()
        return u_

    """build decoder"""

    def build_decoder(self,
                    decoder_type,
                    vocab_size,
                    embedding,
                    rnn_type,
                    hidden_size,
                    num_layers,
                    dropout,
                    tied,
                    turn_type,
                    attn_type,
                    device):

        if decoder_type == 'normal':
            decoder = Decoder(
                vocab_size,
                embedding,
                rnn_type,
                hidden_size,
                num_layers,
                dropout,
                tied,
                turn_type
            )
        elif decoder_type == 'luong':
            decoder = LuongAttnDecoder(vocab_size,
                                       embedding,
                                       rnn_type,
                                       hidden_size,
                                       num_layers,
                                       dropout,
                                       tied,
                                       turn_type,
                                       attn_type,
                                       device)

        else:
            decoder = BahdanauAttnDecoder(
                vocab_size,
                embedding,
                rnn_type,
                hidden_size,
                num_layers,
                dropout,
                tied,
                turn_type,
                attn_type,
                device
            )

        return decoder

