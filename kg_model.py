# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import Encoder
from modules.simple_encoder import SimpleEncoder
from modules.decoder import Decoder
from modules.reduce_state import ReduceState
from modules.bahdanau_attn_decoder import BahdanauAttnDecoder
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.utils import init_lstm_orth, init_gru_orth
from modules.utils import init_linear_wt
from modules.beam_search import beam_decode

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

        self.model_type = model_type
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.turn_num = turn_num
        self.turn_type = turn_type
        self.decoder_type = decoder_type
        self.device = device

        if turn_type != 'concat' or turn_type != 'none':
            self.h_encoder = SimpleEncoder(vocab_size,
                                            embedding_size,
                                            pre_trained_weight,
                                            rnn_type,
                                            hidden_size,
                                            encoder_num_layers,
                                            bidirectional,
                                            dropout,
                                            padding_idx)

        # c_encoder (conversation encoder)
        self.c_encoder = Encoder(vocab_size,
                                 embedding_size,
                                 pre_trained_weight,
                                 rnn_type,
                                 hidden_size,
                                 encoder_num_layers,
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
        self.reduce_state = ReduceState(rnn_type,
                                        hidden_size,
                                        num_layers,
                                        self.c_encoder.bidirection_num)

        self.combine_c_h_linear = nn.Linear(hidden_size * 2, hidden_size)
        init_linear_wt(self.combine_c_h_linear)

        # decoder
        if decoder_type == 'normal':
            self.decoder = Decoder(
                vocab_size,
                embedding_size,
                rnn_type,
                hidden_size,
                decoder_num_layers,
                dropout,
                padding_idx,
                tied
            )
        elif decoder_type == 'luong':
            self.decoder = LuongAttnDecoder(vocab_size,
                                            embedding_size,
                                            rnn_type,
                                            hidden_size,
                                            decoder_num_layers,
                                            dropout,
                                            padding_idx,
                                            turn_type,
                                            tied,
                                            attn_type,
                                            device)
        else:
            self.decoder = BahdanauAttnDecoder(
                vocab_size,
                embedding_size,
                hidden_size,
                decoder_num_layers,
                dropout,
                padding_idx,
                tied,
                attn_type,
                device
            )

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
            h_encoder_inputs: [[[ids], ...], ...]
            c_encoder_inputs: [seq_len, batch_size]
            c_encoder_inputs_length: [batch_size]
            decoder_inputs: [r_max_len, batch_size], first step: [sos * batch_size]

            f_encoder_inputs: [batch_size, r_max_len, topk]
        '''
        # h encoder
        h_encoder_outputs, h_encoder_hidden_state = self.h_forward(h_encoder_inputs, batch_size)

        # conversation encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(batch_size, self.device)
        c_encoder_outputs, c_encoder_hidden_state = self.c_encoder(c_encoder_inputs,
                                                                   c_encoder_inputs_length,
                                                                   c_encoder_hidden_state)

        # cat h_encoder and c_encoder, rnn -> GRU
        if h_encoder_hidden_state is not None:
            c_encoder_hidden_state = self.combine_c_h_state(c_encoder_hidden_state, h_encoder_hidden_state)

        decoder_hidden_state = self.reduce_state(c_encoder_hidden_state, batch_size)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_encoder_inputs,
                                                  decoder_hidden_state,
                                                  batch_size)

        # decoder
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        decoder_input = decoder_inputs[0].view(1, -1)
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for i in range(r_max_len):
                decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                  decoder_hidden_state,
                                                                                  c_encoder_outputs,
                                                                                  h_encoder_outputs)
                decoder_outputs.append(decoder_output)
                decoder_input = decoder_inputs[i].view(1, -1).contiguous()
        else:
            # Without teacher forcing: use its own predictions as the next input
            for i in range(r_max_len):
                decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                  decoder_hidden_state,
                                                                                  c_encoder_outputs,
                                                                                  h_encoder_outputs)

                decoder_outputs.append(decoder_output)
                decoder_input = torch.argmax(decoder_output, dim=2).detach().view(1, -1).contiguous()

        # [r_max_len, batch_size, vocab_size]
        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        return decoder_outputs

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
        # h encoder
        h_encoder_outputs, h_encoder_hidden_state = self.h_forward(h_encoder_inputs, batch_size)

        # c encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(batch_size, self.device)
        c_encoder_outputs, c_encoder_hidden_state = self.c_encoder(c_encoder_inputs,
                                                                   c_encoder_inputs_length,
                                                                   c_encoder_hidden_state)

        # cat h_encoder and c_encoder, rnn -> GRU
        if h_encoder_hidden_state is not None:
            c_encoder_hidden_state = self.combine_c_h_state(c_encoder_hidden_state, h_encoder_hidden_state)

        decoder_hidden_state = self.reduce_state(c_encoder_hidden_state, batch_size)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_encoder_inputs,
                                                     decoder_hidden_state,
                                                     batch_size)

        # decoder
        decoder_outputs = []
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                              decoder_hidden_state,
                                                                              c_encoder_outputs,
                                                                              h_encoder_outputs)

            decoder_input = torch.argmax(decoder_output, dim=2).detach()  # [1, batch_size]
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        return decoder_outputs

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

        # h encoder
        h_encoder_outputs, h_encoder_hidden_state = self.h_forward(h_encoder_inputs, batch_size)

        '''
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        # encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(
            batch_size, self.device)
        c_encoder_outputs,  c_encoder_hidden_state = self.c_encoder(c_encoder_inputs,
                                                                    c_encoder_inputs_length,
                                                                    c_encoder_hidden_state)

        if h_encoder_hidden_state is not None:
            c_encoder_hidden_state = self.combine_c_h_state(c_encoder_hidden_state, h_encoder_hidden_state)

        decoder_hidden_state = self.reduce_state(c_encoder_hidden_state, batch_size)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_encoder_inputs,
                                                     decoder_hidden_state,
                                                     batch_size)

        # decoder
        if decode_type == 'greedy':
            decode_outputs = []
            for i in range(r_max_len):
                decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                  decoder_hidden_state,
                                                                                  c_encoder_outputs,
                                                                                  h_encoder_outputs)

                decoder_input = torch.argmax(decoder_output, dim=2).detach()  # [1, batch_size]
                decode_outputs.append(decoder_input)

                ni = decoder_input[0][0].item()
                if ni == eosid:
                    break

            decode_outputs = torch.cat(decode_outputs, dim=0)
            # [len, batch_size]  -> [batch_size, len]
            decode_outputs.transpose_(0, 1)
            return decode_outputs
        elif decode_type == 'beam_search':
            batch_utterances = beam_decode(
                self.decoder,
                c_encoder_outputs,
                h_encoder_outputs,
                decoder_hidden_state,
                decoder_input,
                batch_size,
                beam_width,
                best_n,
                eosid,
                r_max_len
            )
            return batch_utterances


    def h_forward(self, h_encoder_inputs, batch_size):
        """history forward
        Args:
            h_encoder_inputs: [tensor_ids, ...]
        turn_type:
            concat
            dcgm
            attention
            hred
        """
        if self.turn_type == 'concat' or self.turn_type == 'none':
            return None, None
        elif self.turn_type == 'dcgm':
            # [[h], [h]....]
            h_encoder_hidden_states = []
            for h_encoder_input in h_encoder_inputs:
                if len(h_encoder_input) == 0:
                    h_encoder_hidden_state = torch.zeros((self.encoder_num_layers * self.h_encoder.bidirection_num, 1, self.hidden_size), device=self.device)
                else:
                    # h_encoder_input: [h]
                    h_encoder_hidden_state = self.h_encoder.init_hidden(1, self.device)
                    _, h_encoder_hidden_state = self.h_encoder(h_encoder_input[0].view(-1, 1), h_encoder_hidden_state) # h_encoder_hidden_state: [num_layers * bidirection_num, 1, hidden_size]
                h_encoder_hidden_states.append(h_encoder_hidden_state)
            h_encoder_hidden_states = torch.cat(h_encoder_hidden_states, dim=1) # [num_layers * bidirection_num, batch_size, hidden_size]
            return None, h_encoder_hidden_states
        elif self.turn_type == 'attention':
            h_outputs = []
            h_encoder_hidden_states = [] # [batch_size, hidden_size]
            for h_encoder_input in h_encoder_inputs:
                h_attn_hidden_states = []
                for i, h in enumerate(h_encoder_input):
                    # h: h is a tensor object
                    h_encoder_hidden_state = self.h_encoder.init_hidden(1, self.device)
                    _, h_encoder_hidden_state = self.h_encoder(h.view(-1, 1), h_encoder_hidden_state)
                    h_encoder_hidden_states.append(h_encoder_hidden_state.squeeze(1))
                h_attn_hidden_states = torch.stack(h_attn_hidden_states, dim=0) #[num, num_layers * bidirection_num, hidden_size]

                #  h_encoder_hidden_state.append(torch.stack(h_encoder_hidden_states, dim=0).mean(0))
            #  # [batch_size, turn_num - 1, hidden_size]
            #  h_outputs = torch.stack(h_outputs, dim=0)
            #  # [turn_num - 1, batch_size, hidden_size]
            #  h_outputs.transpose_(0, 1)
            #  # [num_layers * bidirection_num, batch_size, hidden_size]
            #  h_encoder_hidden_state = torch.cat(h_encoder_hidden_state, dim=1)

            return h_outputs, h_encoder_hidden_states
        elif self.turn_type == 'hred':
            pass

        return None, None


    def f_forward(self,
                     f_encoder_inputs,
                     hidden_state,
                     batch_size):
        """
        Args:
            - f_encoder_inputs: [batch_size, top_k, embedding_size]
            - hidden_state: [num_layers, batch_size, hidden_size]
            - batch_size
        """
        # [batch_size, topk, embedding_size] -> [batch_size, topk, hidden_size]
        if self.embedding_size != self.hidden_size:
            f_encoder_inputs = self.fact_linear(f_encoder_inputs)

        # M [batch_size, topk, hidden_size]
        fact_M = self.fact_linearA(f_encoder_inputs)

        # C [batch_size, topk, hidden_size]
        fact_C = self.fact_linearC(f_encoder_inputs)

        # [batch_size, num_layers, topk]
        tmpP = torch.bmm(hidden_state.transpose(0, 1), fact_M.transpose(1, 2))
        P = F.softmax(tmpP, dim=2)

        o = torch.bmm(P, fact_C)  # [batch_size, num_layers, hidden_size]
        u_ = torch.add(o, hidden_state.transpose(0, 1))
        # [num_layers, batch_size, hidden_size]
        u_ = u_.transpose(0, 1).contiguous()
        return u_

    def combine_c_h_state(self, c_encoder_hidden_state, h_encoder_hidden_state):
        #  tmp_encoder_hidden_state = torch.add(c_encoder_hidden_state, h_encoder_hidden_state)
        # or
        tmp_encoder_hidden_state = torch.cat((c_encoder_hidden_state, h_encoder_hidden_state), dim=2)
        tmp_encoder_hidden_state = torch.relu(self.combine_c_h_linear(tmp_encoder_hidden_state))
        return tmp_encoder_hidden_state

