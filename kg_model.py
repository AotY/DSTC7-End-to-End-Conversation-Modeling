# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import Encoder
from modules.simple_encoder import SimpleEncoder
from modules.session_encoder import SessionEncoder
from modules.session_decoder import SessionDecoder
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
        if turn_type != 'concat' or turn_type != 'none':
            self.simple_encoder = SimpleEncoder(vocab_size,
                                                encoder_embedding,
                                                rnn_type,
                                                hidden_size,
                                                encoder_num_layers,
                                                bidirectional,
                                                dropout)
        if turn_type == 'hred' or turn_type == 'hred_attn':
            self.session_encoder = SessionDecoder(
                rnn_type,
                hidden_size,
                num_layers,
                dropout,
                attn_type
            )

        # c_encoder (conversation encoder)
        self.c_encoder = Encoder(vocab_size,
                                 encoder_embedding,
                                 rnn_type,
                                 hidden_size,
                                 encoder_num_layers,
                                 bidirectional,
                                 dropout)

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

        if self.rnn_type == 'GRU':
            self.combine_c_h_linear = nn.Linear(hidden_size, hidden_size)
        else:
            self.combine_c_h_linear_1 = nn.Linear(hidden_size, hidden_size)
            self.combine_c_h_linear_2 = nn.Linear(hidden_size, hidden_size)


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
                h_encoder_inputs,
                h_encoder_inputs_length,
                c_encoder_inputs,
                c_encoder_inputs_length,
                decoder_inputs,
                f_encoder_inputs,
                f_encoder_inputs_length,
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

        # [turn_num, batch_size, hidden_size], [num_layers, batch_size, hidden_size]
        h_encoder_outputs, h_encoder_hidden_state = self.h_forward(
            h_encoder_inputs,
            h_encoder_inputs_length,
            batch_size
        )

        # conversation encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(batch_size, self.device)

        # [c_len, batch_size, hidden_size],  [num_layers * bidirection_num, batch_size, hidden_size//2]
        c_encoder_outputs, c_encoder_hidden_state = self.c_encoder(c_encoder_inputs,
                                                                   c_encoder_inputs_length,
                                                                   c_encoder_hidden_state)

        # [num_layers, batch_size, hidden_size]
        decoder_hidden_state = self.reduce_state(c_encoder_hidden_state)

        if h_encoder_hidden_state is not None:
            decoder_hidden_state = self.combine_c_h_state(decoder_hidden_state, h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_encoder_inputs,
                                                  f_encoder_inputs_length,
                                                  decoder_hidden_state,
                                                  batch_size)

        # decoder
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_outputs, decoder_hidden_state, attn_weights = self.decoder(decoder_inputs,
                                                                               decoder_hidden_state,
                                                                               c_encoder_outputs,
                                                                               c_encoder_inputs_length,
                                                                               h_encoder_outputs,
                                                                               h_encoder_inputs_length)
            return decoder_outputs

        else:
            decoder_outputs = []
            decoder_input = decoder_inputs[0].view(1, -1)

            for i in range(r_max_len):
                decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                  decoder_hidden_state,
                                                                                  c_encoder_outputs,
                                                                                  c_encoder_inputs_length,
                                                                                  h_encoder_outputs,
                                                                                  h_encoder_inputs_length)

                decoder_outputs.append(decoder_output)
                decoder_input = torch.argmax(
                    decoder_output, dim=2).detach().view(1, -1)

            # [r_max_len, batch_size, vocab_size]
            decoder_outputs = torch.cat(decoder_outputs, dim=0)

            return decoder_outputs

    '''evaluate'''

    def evaluate(self,
                 h_encoder_inputs,
                 h_encoder_inputs_length,
                 c_encoder_inputs,
                 c_encoder_inputs_length,
                 decoder_input,
                 f_encoder_inputs,
                 f_encoder_inputs_length,
                 r_max_len,
                 batch_size):
        '''
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        # h encoder
        h_encoder_outputs, h_encoder_hidden_state = self.h_forward(
            h_encoder_inputs,
            h_encoder_inputs_length,
            batch_size
        )

        # c encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(batch_size, self.device)

        c_encoder_outputs, c_encoder_hidden_state = self.c_encoder(c_encoder_inputs,
                                                                   c_encoder_inputs_length,
                                                                   c_encoder_hidden_state)

        # [num_layers, batch_size, hidden_size]
        decoder_hidden_state = self.reduce_state(c_encoder_hidden_state)

        if h_encoder_hidden_state is not None:
            decoder_hidden_state = self.combine_c_h_state(decoder_hidden_state, h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_encoder_inputs,
                                                  f_encoder_inputs_length,
                                                  decoder_hidden_state,
                                                  batch_size)

        # decoder
        decoder_outputs = []
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, _ = self.decoder(decoder_input,
                                                                   decoder_hidden_state,
                                                                   c_encoder_outputs,
                                                                   c_encoder_inputs_length,
                                                                   h_encoder_outputs,
                                                                   h_encoder_inputs_length)

            decoder_input = torch.argmax(
                decoder_output, dim=2).detach()  # [1, batch_size]
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        return decoder_outputs

    '''decode'''

    def decode(self,
               h_encoder_inputs,
               h_encoder_inputs_length,
               c_encoder_inputs,
               c_encoder_inputs_length,
               decoder_input,
               f_encoder_inputs,
               f_encoder_inputs_length,
               decode_type,
               r_max_len,
               eosid,
               batch_size,
               beam_width,
               best_n):

        # h encoder
        h_encoder_outputs, h_encoder_hidden_state = self.h_forward(
            h_encoder_inputs,
            h_encoder_inputs_length,
            batch_size
        )

        '''
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        # encoder
        c_encoder_hidden_state = self.c_encoder.init_hidden(batch_size, self.device)

        c_encoder_outputs,  c_encoder_hidden_state = self.c_encoder(c_encoder_inputs,
                                                                    c_encoder_inputs_length,
                                                                    c_encoder_hidden_state)

        # [num_layers, batch_size, hidden_size]
        decoder_hidden_state = self.reduce_state(c_encoder_hidden_state)

        if h_encoder_hidden_state is not None:
            decoder_hidden_state = self.combine_c_h_state(decoder_hidden_state, h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_encoder_inputs,
                                                  f_encoder_inputs_length,
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
                                                                              c_encoder_outputs,
                                                                              c_encoder_inputs_length,
                                                                              h_encoder_outputs,
                                                                              h_encoder_inputs_length)

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
            c_encoder_outputs,
            c_encoder_inputs_length,
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

    def h_forward(self, h_encoder_inputs, h_encoder_inputs_length, batch_size):
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
                    h_encoder_hidden_state = torch.zeros(
                        (self.encoder_num_layers * self.simple_encoder.bidirection_num, 1, self.hidden_size), device=self.device)
                else:
                    # h_encoder_input: [h]
                    h_encoder_hidden_state = self.simple_encoder.init_hidden(
                        1, self.device)
                    # h_encoder_hidden_state: [num_layers * bidirection_num, 1, hidden_size]
                    _, h_encoder_hidden_state = self.simple_encoder(
                        h_encoder_input[0].view(-1, 1), h_encoder_hidden_state)

                h_encoder_hidden_states.append(h_encoder_hidden_state)
            # [num_layers * bidirection_num, batch_size, hidden_size]
            h_encoder_hidden_states = torch.cat(h_encoder_hidden_states, dim=1)
            return None, h_encoder_hidden_states
        elif self.turn_type == 'hred' or self.turn_type == 'hred_attn':
            h_encoder_hidden_states = []
            h_encoder_outputs = []

            for h_encoder_input in h_encoder_inputs:
                # [num_layers, 1, hidden_size]
                session_encoder_hidden_state = self.session_encoder.init_hidden(1, self.device)

                # [turn_num, 1, hidden_size]
                tmp_sesseion_encocer_outputs = torch.zeros((self.turn_num, 1, self.hidden_size), device=self.device)

                for i, ids in enumerate(h_encoder_input):
                    # [num_layers * bidirection_num, 1, hidden_size // bidirection_num]
                    simple_encoder_hidden_state = self.simple_encoder.init_hidden(1, self.device)
                    simple_encoder_outputs, simple_encoder_hidden_state = self.simple_encoder(ids.view(-1, 1),
                                                                                              simple_encoder_hidden_state)

                    # session update
                    session_encoder_output, session_encoder_hidden_state = self.session_encoder(
                        simple_encoder_outputs[-1].unsqueeze(0),
                        session_encoder_hidden_state,
                        simple_encoder_outputs
                    )

                    #  session_encoder_output, session_encoder_hidden_state = self.session_encoder(simple_encoder_outputs[-1].unsqueeze(0),
                                                                                                 #  session_encoder_hidden_state)

                    tmp_sesseion_encocer_outputs[i, :, :] = session_encoder_output

                h_encoder_outputs.append(tmp_sesseion_encocer_outputs)

                # [num_layers, 1, hidden_size]
                h_encoder_hidden_states.append(session_encoder_hidden_state)


            # [turn_num, batch_size, hidden_size]
            h_encoder_outputs = torch.cat(h_encoder_outputs, dim=1)

            # [num_layers, batch_size, hidden_size]
            if self.rnn_type == 'GRU':
                h_encoder_hidden_states = torch.cat(h_encoder_hidden_states, dim=1)
            else:
                tmp_hs, tmp_cs = list(), list()
                for h, c in h_encoder_hidden_states:
                    tmp_hs.append(h)
                    tmp_cs.append(c)

                tmp_hs = torch.cat(tmp_hs, dim=1)
                tmp_cs = torch.cat(tmp_cs, dim=1)
                #  h_encoder_hidden_states = tuple(list[torch.cat(item, dim=1) for item in [tmp_hs, tmp_cs]])
                h_encoder_hidden_states = tuple([tmp_hs, tmp_cs])

            return h_encoder_outputs, h_encoder_hidden_states

    def f_forward(self,
                  f_encoder_inputs,
                  f_encoder_inputs_length,
                  hidden_state,
                  batch_size):
        """
        Args:
            - f_encoder_inputs: [batch_size, top_k, embedding_size]
            - hidden_state: [num_layers, batch_size, hidden_size]
            - batch_size
        """

        # [batch_size, topk, embedding_size] -> [batch_size, topk, hidden_size]
        if self.pre_embedding_size != self.hidden_size:
            f_encoder_inputs = self.fact_linear(f_encoder_inputs)

        # M [batch_size, topk, hidden_size]
        fact_M = self.fact_linearA(f_encoder_inputs)

        # C [batch_size, topk, hidden_size]
        fact_C = self.fact_linearC(f_encoder_inputs)

        # [batch_size, num_layers, topk]
        tmpP = torch.bmm(hidden_state.transpose(0, 1), fact_M.transpose(1, 2))

        mask = sequence_mask(f_encoder_inputs_length, max_len=tmpP.size(-1))
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        tmpP.masked_fill_(1 - mask, -float('inf'))

        P = F.softmax(tmpP, dim=2)

        o = torch.bmm(P, fact_C)  # [batch_size, num_layers, hidden_size]
        u_ = torch.add(o, hidden_state.transpose(0, 1))

        # [num_layers, batch_size, hidden_size]
        u_ = u_.transpose(0, 1).contiguous()
        return u_

    def combine_c_h_state(self, c_encoder_hidden_state, h_encoder_hidden_state):
        if self.rnn_type == 'GRU':
            #  tmp_encoder_hidden_state = c_encoder_hidden_state + h_encoder_hidden_state
            tmp_encoder_hidden_state = torch.cat((c_encoder_hidden_state, h_encoder_hidden_state), dim=2)
            tmp_encoder_hidden_state = torch.tanh(self.combine_c_h_linear(tmp_encoder_hidden_state))
        else:
            #  tmp_encoder_hidden_state = tuple([item1 + item2 for (item1, item2) in zip(c_encoder_hidden_state, h_encoder_hidden_state)])
            tmp_encoder_hidden_state_1 = torch.cat((c_encoder_hidden_state[0], h_encoder_hidden_state[0]), dim=2)
            tmp_encoder_hidden_state_1 = torch.tanh(self.combine_c_h_linear_1(tmp_encoder_hidden_state_1))
            tmp_encoder_hidden_state_2 = torch.cat((c_encoder_hidden_state[1], h_encoder_hidden_state[1]), dim=2)
            tmp_encoder_hidden_state_2 = torch.tanh(self.combine_c_h_linear_2(tmp_encoder_hidden_state_2))
            return tuple([tmp_encoder_hidden_state_1, tmp_encoder_hidden_state_2])

        return tmp_encoder_hidden_state

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

