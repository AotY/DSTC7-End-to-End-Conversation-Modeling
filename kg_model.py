# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import Encoder
#  from modules.decoder import Decoder
from modules.reduce_state import ReduceState
#  from modules.bahdanau_attn_decoder import BahdanauAttnDecoder
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.utils import init_lstm_orth, init_gru_orth
from modules.utils import init_linear_wt

"""
KGModel
1. dialogue_encoder
2. facts_encoder
3. dialogue_decoder

"""

class KGModel(nn.Module):
    '''
    conditoning responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 model_type,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 attn_type,
                 dropout,
                 padding_idx,
                 tied,
                 device):
        super(KGModel, self).__init__()

        self.model_type = model_type
        self.device = device

        # dialogue_encoder
        self.dialogue_encoder = Encoder(vocab_size,
                                        embedding_size,
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
        self.reduce_state = ReduceState(hidden_size)

        # dialogue_decoder
        self.dialogue_decoder = LuongAttnDecoder(vocab_size,
                                               embedding_size,
                                               hidden_size,
                                               num_layers,
                                               dropout,
                                               padding_idx,
                                               tied,
                                               attn_type,
                                               device)

    def forward(self,
                dialogue_encoder_inputs,
                dialogue_encoder_inputs_length,
                dialogue_decoder_inputs,
                facts_inputs,
                batch_size,
                max_len,
                teacher_forcing_ratio):
        '''
        input:
            dialogue_encoder_inputs: [seq_len, batch_size]
            dialogue_encoder_inputs_length: [batch_size]
            decoder_inputs: [max_len, batch_size], first step: [sos * batch_size]
            dialogue_decoder_targets: [max_len, batch_size]

            fact_inputs: [batch_size, max_len, topk]
        '''
        # dialogue encoder
        dialogue_encoder_hidden_state = self.dialogue_encoder.init_hidden(batch_size, self.device)
        dialogue_encoder_outputs,  \
        dialogue_encoder_hidden_state, \
        dialogue_encoder_max_output = self.dialogue_encoder(dialogue_encoder_inputs,
                                                            dialogue_encoder_inputs_length,
                                                            dialogue_encoder_hidden_state)

        # dialogue_encoder_hidden_state -> [num_layers * num_directions, batch, hidden_size]
        #  dialogue_decoder_hidden_state = tuple([item[:2, :, :] + item[2:, :, :] for item in dialogue_encoder_hidden_state])
        dialogue_decoder_hidden_state = self.reduce_state(dialogue_encoder_hidden_state, batch_size)

        # fact encoder
        if self.model_type == 'kg':
            dialogue_decoder_hidden_state = self.fact_forward(facts_inputs,
                                                                dialogue_decoder_hidden_state,
                                                                batch_size)

        # dialogue decoder
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        dialogue_decoder_outputs = []
        dialogue_decoder_input = dialogue_decoder_inputs[0].view(1, -1) # sos [1, batch_size]
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for i in range(max_len):
                dialogue_decoder_output, dialogue_decoder_hidden_state, attn_weights = self.dialogue_decoder(dialogue_decoder_input,
                                                                                                    dialogue_decoder_hidden_state,
                                                                                                    dialogue_encoder_max_output,
                                                                                                    dialogue_encoder_outputs)
                dialogue_decoder_outputs.append(dialogue_decoder_output)
                dialogue_decoder_input = dialogue_decoder_inputs[i].view(1, -1)
        else:
            # Without teacher forcing: use its own predictions as the next input
            for i in range(max_len):
                dialogue_decoder_output, dialogue_decoder_hidden_state, attn_weights = self.decoder(dialogue_decoder_input,
                                                                                                    dialogue_decoder_hidden_state,
                                                                                                    dialogue_encoder_max_output,
                                                                                                    dialogue_encoder_outputs)
                dialogue_decoder_outputs.append(dialogue_decoder_output)
                dialogue_decoder_input = torch.argmax(dialogue_decoder_output, dim=2).detach().view(1, -1)

        dialogue_decoder_outputs = torch.cat(dialogue_decoder_outputs, dim=0)

        return dialogue_decoder_outputs

    '''evaluate'''

    def evaluate(self,
                 dialogue_encoder_inputs,
                 dialogue_encoder_inputs_length,
                 dialogue_decoder_input,
                 facts_inputs,
                 max_len,
                 batch_size):
        '''
        dialogue_encoder_inputs: [seq_len, batch_size], maybe [max_len, 1]
        dialogue_decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        # encoder
        dialogue_encoder_hidden_state = self.dialogue_encoder.init_hidden(batch_size)
        dialogue_encoder_outputs, dialogue_encoder_hidden_state = self.dialogue_encoder(dialogue_encoder_inputs,
                                                                                        dialogue_encoder_inputs_length,
                                                                                        dialogue_encoder_hidden_state)

        #  dialogue_decoder_hidden_state = tuple([item[:2, :, :] + item[2:, :, :] for item in dialogue_encoder_hidden_state])
        dialogue_decoder_hidden_state = self.reduce_state(dialogue_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            dialogue_decoder_hidden_state = self.fact_forward(facts_inputs,
                                                                dialogue_decoder_hidden_state,
                                                                batch_size)


        # decoder
        dialogue_decoder_outputs = []
        for i in range(max_len):
            dialogue_decoder_output, dialogue_decoder_hidden_state, attn_weights = self.decoder(dialogue_decoder_input,
                                                                                                dialogue_decoder_hidden_state,
                                                                                                dialogue_encoder_outputs)

            dialogue_decoder_input = torch.argmax(dialogue_decoder_output, dim=2).detach() #[1, batch_size]
            dialogue_decoder_outputs.append(dialogue_decoder_output)

        dialogue_decoder_outputs = torch.cat(dialogue_decoder_outputs, dim=0)

        return dialogue_decoder_outputs



    '''decode'''

    def decode(self,
                 dialogue_encoder_inputs,
                 dialogue_encoder_inputs_length,
                 dialogue_decoder_input,
                 facts_inputs,
                 decode_type,
                 max_len,
                 eosid,
                 batch_size,
                 beam_width,
                 best_n):
        '''
        dialogue_encoder_inputs: [seq_len, batch_size], maybe [max_len, 1]
        dialogue_decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        # encoder
        dialogue_encoder_hidden_state = self.dialogue_encoder.init_hidden(batch_size)
        dialogue_encoder_outputs, dialogue_encoder_hidden_state = self.dialogue_encoder(dialogue_encoder_inputs,
                                                                                        dialogue_encoder_inputs_length,
                                                                                        dialogue_encoder_hidden_state)

        #  dialogue_decoder_hidden_state = tuple([item[:1, :, :] + item[2:, :, :] for item in dialogue_encoder_hidden_state])
        dialogue_decoder_hidden_state = self.reduce_state(dialogue_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            dialogue_decoder_hidden_state = self.fact_forward(facts_inputs,
                                                                dialogue_decoder_hidden_state,
                                                                batch_size)

        # dialogue decoder
        if decode_type == 'greedy':
            dialogue_decode_outputs = []
            for i in range(max_len):
                dialogue_decoder_output, dialogue_decoder_hidden_state, attn_weights = self.decoder(dialogue_decoder_input,
                                                                                                    dialogue_decoder_hidden_state,
                                                                                                    dialogue_encoder_outputs)

                dialogue_decoder_input = torch.argmax(dialogue_decoder_output, dim=2).detach() #[1, batch_size]
                dialogue_decode_outputs.append(dialogue_decoder_input)

                ni = dialogue_decoder_input[0][0].item()
                if ni == eosid:
                    break

            # [len, batch_size]  -> [batch_size, len]
            dialogue_decode_outputs = torch.cat(dialogue_decode_outputs, dim=0)
            dialogue_decode_outputs.transpose_(0, 1)
        elif decode_type == 'beam_search':
            pass

        return dialogue_decode_outputs



    def fact_forward(self,
                     facts_inputs,
                     hidden_state,
                     batch_size):
        """
        Args:
            - facts_inputs: [batch_size, top_k, embedding_size]
            - dialogue_decoder_hidden_state
            - batch_size
        """
        # [batch_size, topk, embedding_size] -> [batch_size, topk, hidden_size]
        if self.embedding_size != self.hidden_size:
            facts_inputs = self.fact_linear(facts_inputs)

        # M [batch_size, topk, hidden_size]
        fact_M = self.fact_linearA(facts_inputs)

        # C [batch_size, topk, hidden_size]
        fact_C = self.fact_linearC(facts_inputs)

        # hidden_tuple is a tuple object
        new_hidden_list = []
        for cur_hidden_state in hidden_state:
            # cur_hidden_state: [num_layers, batch_size, hidden_size]
            cur_hidden_state.transpose_(0, 1) # [batch_size, num_layers, hidden_size]
            tmpP = torch.bmm(cur_hidden_state, fact_M.transpose(1, 2)) # [batch_size, num_layers, topk]
            P = F.softmax(tmpP, dim=2)

            o = torch.bmm(P, fact_C) # [batch_size, num_layers, hidden_size]
            u_ = o + cur_hidden_state
            u_.transpose_(0, 1) # [num_layers, batch_size, hidden_size]

            new_hidden_list.append(u_)

        new_hidden_state = tuple(new_hidden_list)
        return new_hidden_state

