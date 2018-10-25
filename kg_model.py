# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import Encoder
from modules.simple_encoder import SimpleEncoder
from modules.decoder import Decoder
from modules.reduce_state import ReduceState
#  from modules.bahdanau_attn_decoder import BahdanauAttnDecoder
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.utils import init_lstm_orth, init_gru_orth
from modules.utils import init_linear_wt

"""
KGModel
1. h_encoder (history_conversations)
2. c_encoder  (conversation)
3. fact_encoder (facts)
4. decoder

"""


class KGModel(nn.Module):
    '''
    generating responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 model_type,
                 vocab_size,
                 embedding_size,
                 rnn_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 turn_num,
                 turn_type,
                 decoder_type,
                 attn_type,
                 dropout,
                 padding_idx,
                 tied,
                 device):
        super(KGModel, self).__init__()

        self.model_type = model_type
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.turn_num = turn_num
        self.turn_type = turn_type
        self.decoder_type = decoder_type
        self.device = device

        if turn_num > 1 and turn_type != 'concat':
                        # history_encoder
            self.history_encoder = SimpleEncoder(vocab_size,
                                                 embedding_size,
                                                 rnn_type,
                                                 hidden_size,
                                                 num_layers,
                                                 bidirectional,
                                                 dropout,
                                                 padding_idx)

        # c_encoder (conversation encoder)
        self.c_encoder = Encoder(vocab_size,
                                 embedding_size,
                                 rnn_type,
                                 hidden_size,
                                 num_layers,
                                 bidirectional,
                                 dropout,
                                 padding_idx)

        # fact encoder
        if model_type == 'kg':
            # mi = A * ri    fact_linearA(300, 512)
            self.fact_linearA = nn.Linear(hidden_size, hidden_size)
            init_linear_wt(self.fact_linearA)
            # ci = C * ri
            self.fact_linearC = nn.Linear(hidden_size, hidden_size)
            init_linear_wt(self.fact_linearC)

            self.fact_linear = nn.Linear(embedding_size, hidden_size)
            init_linear_wt(self.fact_linear)

        # encoder hidden_state -> decoder hidden_state
        self.reduce_state = ReduceState(
            hidden_size, num_layers, self.c_encoder.bidirection_num)

        # decoder
        if decoder_type == 'luong':
            self.decoder = LuongAttnDecoder(vocab_size,
                                            embedding_size,
                                            rnn_type,
                                            hidden_size,
                                            num_layers,
                                            dropout,
                                            padding_idx,
                                            tied,
                                            attn_type,
                                            device)
        elif decoder_type == 'normal':
            self.decoder = Decoder(
                vocab_size,
                embedding_size,
                rnn_type,
                hidden_size,
                num_layers,
                dropout,
                padding_idx,
                tied)

    def forward(self,
                h_encoder_inputs,
                c_encoder_inputs,
                c_encoder_inputs_length,
                decoder_inputs,
                f_encoder_inputs,
                batch_size,
                r_max_len,
                teacher_forcing_ratio):
        '''
        input:
            h_encoder_inputs: [[num, h_max], ...]
            c_encoder_inputs_length: [seq_len, batch_size]
            c_encoder_inputs_length: [batch_size]
            decoder_inputs: [r_max_len, batch_size], first step: [sos * batch_size]
            dialogue_decoder_targets: [r_max_len, batch_size]

            f_encoder_inputs: [batch_size, r_max_len, topk]
        '''
        # h encoder
        h_encoder_outputs = None
        h_encoder_hidden_state = None
        if self.turn_num > 1 and len(h_encoder_inputs) > 0:
            h_encoder_outputs, h_encoder_hidden_state = self.h_forward(h_encoder_inputs, batch_size)

        # dialogue encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(
            batch_size, self.device)
        c_encoder_outputs, c_encoder_hidden_state, c_encoder_max_output = self.c_encoder(c_encoder_inputs,
                                                                                         c_encoder_inputs_length,
                                                                                         c_encoder_hidden_state)
        # dialogue encoder reduce
        # [num_layers * num_directions, batch, hidden_size] -> [num_layers, batch, hidden_size]
        if h_encoder_hidden_state is not None:
            c_encoder_hidden_state += h_encoder_hidden_state

        decoder_hidden_state = self.reduce_state(
            c_encoder_hidden_state, batch_size)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.fact_forward(f_encoder_inputs,
                                                     decoder_hidden_state,
                                                     batch_size)

        # dialogue decoder
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        dialogue_decoder_outputs = []
        decoder_input = decoder_inputs[0].view(
            1, -1)  # sos [1, batch_size]
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for i in range(r_max_len):
                dialogue_decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                           decoder_hidden_state,
                                                                                           c_encoder_max_output,
                                                                                           c_encoder_outputs,
                                                                                           h_encoder_outputs)
                dialogue_decoder_outputs.append(dialogue_decoder_output)
                decoder_input = decoder_inputs[i].view(
                    1, -1).contiguous()
        else:
            # Without teacher forcing: use its own predictions as the next input
            for i in range(r_max_len):
                dialogue_decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                           decoder_hidden_state,
                                                                                           c_encoder_max_output,
                                                                                           c_encoder_outputs,
                                                                                           h_encoder_outputs)

                dialogue_decoder_outputs.append(dialogue_decoder_output)
                decoder_input = torch.argmax(
                    dialogue_decoder_output, dim=2).detach().view(1, -1).contiguous()

        dialogue_decoder_outputs = torch.cat(dialogue_decoder_outputs, dim=0)

        return dialogue_decoder_outputs

    '''evaluate'''

    def evaluate(self,
                 h_encoder_inputs,
                 c_encoder_inputs,
                 c_encoder_inputs_length,
                 decoder_input,
                 f_encoder_inputs,
                 r_max_len,
                 batch_size):
        '''
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        h_encoder_outputs = None
        h_encoder_hidden_state = None
        if self.turn_num > 1 and len(h_encoder_inputs) > 0:
            h_encoder_outputs, h_encoder_hidden_state = self.h_forward(
                h_encoder_inputs, batch_size)

        # encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(batch_size, self.device)
        c_encoder_outputs, c_encoder_hidden_state, c_encoder_max_output = self.c_encoder(c_encoder_inputs,
                                                                                         c_encoder_inputs_length,
                                                                                         c_encoder_hidden_state)

        if h_encoder_hidden_state is not None:
            c_encoder_hidden_state += h_encoder_hidden_state

        decoder_hidden_state = self.reduce_state(c_encoder_hidden_state, batch_size)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.fact_forward(f_encoder_inputs,
                                                     decoder_hidden_state,
                                                     batch_size)

        # decoder
        dialogue_decoder_outputs = []
        for i in range(r_max_len):
            dialogue_decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                       decoder_hidden_state,
                                                                                       c_encoder_max_output,
                                                                                       c_encoder_outputs,
                                                                                       h_encoder_outputs)

            decoder_input = torch.argmax(dialogue_decoder_output, dim=2).detach()  # [1, batch_size]
            dialogue_decoder_outputs.append(dialogue_decoder_output)

        dialogue_decoder_outputs = torch.cat(dialogue_decoder_outputs, dim=0)

        return dialogue_decoder_outputs

    '''decode'''

    def decode(self,
               h_encoder_inputs,
               c_encoder_inputs,
               c_encoder_inputs_length,
               decoder_input,
               f_encoder_inputs,
               decode_type,
               r_max_len,
               eosid,
               batch_size,
               beam_width,
               best_n):

        h_encoder_outputs = None
        h_encoder_hidden_state = None
        if self.turn_num > 1 and len(h_encoder_inputs) > 0:
            h_encoder_outputs, h_encoder_hidden_state = self.h_forward(
                h_encoder_inputs, batch_size)

        '''
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        # encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(
            batch_size, self.device)
        c_encoder_outputs,  \
            c_encoder_hidden_state, \
            c_encoder_max_output = self.c_encoder(c_encoder_inputs,
                                                  c_encoder_inputs_length,
                                                  c_encoder_hidden_state)

        if h_encoder_hidden_state is not None:
            c_encoder_hidden_state += h_encoder_hidden_state
        decoder_hidden_state = self.reduce_state(
            c_encoder_hidden_state, batch_size)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.fact_forward(f_encoder_inputs,
                                                     decoder_hidden_state,
                                                     batch_size)

        # dialogue decoder
        if decode_type == 'greedy':
            dialogue_decode_outputs = []
            for i in range(r_max_len):
                dialogue_decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                           decoder_hidden_state,
                                                                                           c_encoder_max_output,
                                                                                           c_encoder_outputs,
                                                                                           h_encoder_outputs)

                decoder_input = torch.argmax(
                    dialogue_decoder_output, dim=2).detach()  # [1, batch_size]
                dialogue_decode_outputs.append(decoder_input)

                ni = decoder_input[0][0].item()
                if ni == eosid:
                    break

            dialogue_decode_outputs = torch.cat(dialogue_decode_outputs, dim=0)
            # [len, batch_size]  -> [batch_size, len]
            dialogue_decode_outputs.transpose_(0, 1)
        elif decode_type == 'beam_search':
            pass

        return dialogue_decode_outputs

    def h_forward(self, h_encoder_inputs, batch_size):
        """history forward
        Args:
            h_encoder_inputs: [tensor_ids, ...]
        turn_type:
            concat
            dcgm1
            dcgm2
            hred
        """
        h_encoder_outputs = []
        h_encoder_hidden_state = []
        for history_input in h_encoder_inputs:
            final_outputs = torch.zeros(
                (self.turn_num - 1, self.hidden_size), device=self.device)
            final_hidden_states = []
            for i, item in enumerate(history_input):
                # item: [id, ...] is a tensor object
                # [num_layers * bidirection_num, 1, hidden_size]
                hidden_state = self.hisotry_encoder.init_hidden()
                outputs, hidden_state = self.history_encoder(
                    item.view(-1, 1), hidden_state)
                # outputs: [len, 1, hidden_size]
                fianl_outputs[i] = outputs[-1]  # [1, hidden_size]
                final_hidden_states.append(hidden_state)
            h_encoder_outputs.append(fianl_outputs)
            h_encoder_hidden_state.append(
                torch.stack(final_hidden_states, dim=0).mean(0))
        # [batch_size, turn_num - 1, hidden_size]
        h_encoder_outputs = torch.stack(h_encoder_outputs, dim=0)
        # [turn_num - 1, batch_size, hidden_size]
        h_encoder_outputs.transpose_(0, 1)
        # [num_layers * bidirection_num, batch_size, hidden_size]
        h_encoder_hidden_state = torch.cat(h_encoder_hidden_state, dim=1)

        return h_encoder_outputs, h_encoder_hidden_state

    def fact_forward(self,
                     f_encoder_inputs,
                     hidden_state,
                     batch_size):
        """
        Args:
            - f_encoder_inputs: [batch_size, top_k, embedding_size]
            - decoder_hidden_state
            - batch_size
        """
        # [batch_size, topk, embedding_size] -> [batch_size, topk, hidden_size]
        if self.embedding_size != self.hidden_size:
            f_encoder_inputs = self.fact_linear(f_encoder_inputs)

        # M [batch_size, topk, hidden_size]
        fact_M = self.fact_linearA(f_encoder_inputs)

        # C [batch_size, topk, hidden_size]
        fact_C = self.fact_linearC(f_encoder_inputs)

        # hidden_tuple is a tuple object
        new_hidden_list = []
        for cur_hidden_state in hidden_state:
            # cur_hidden_state: [num_layers, batch_size, hidden_size]
            # [batch_size, num_layers, hidden_size]
            cur_hidden_state.transpose_(0, 1)
            # [batch_size, num_layers, topk]
            tmpP = torch.bmm(cur_hidden_state, fact_M.transpose(1, 2))
            P = F.softmax(tmpP, dim=2)

            o = torch.bmm(P, fact_C)  # [batch_size, num_layers, hidden_size]
            u_ = o + cur_hidden_state
            # [num_layers, batch_size, hidden_size]
            u_ = u_.transpose(0, 1).contiguous()

            new_hidden_list.append(u_)

        new_hidden_state = tuple(new_hidden_list)
        return new_hidden_state
