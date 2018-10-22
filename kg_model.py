# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as f

from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.reduce_state import ReduceState
from modules.bahdanau_attn_decoder import BahdanauAttnDecoder
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.utils import init_lstm_orth, init_gru_orth

"""
KGModel
1. dialogue_encoder
2. facts_encoder
3. dialogue_decoder

"""

class KGModel(nn.module):
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
                                        padding_idx,
                                        device)

        # fact encoder
        if model_type == 'kg':
            self.fact_encoder = None



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
                fact_inputs,
                batch_size,
                max_len,
                teacher_forcing_ratio):
        '''
        input:
            dialogue_encoder_inputs: [seq_len, batch_size]
            dialogue_encoder_inputs_length: [batch_size]
            decoder_inputs: [max_len, batch_size], first step: [sos * batch_size]
            dialogue_decoder_targets: [max_len, batch_size]

            fact_inputs
            fact_inputs_length
        '''
        # dialogue encoder
        dialogue_encoder_hidden_state = self.dialogue_encoder.init_hidden(batch_size, self.device)
        dialogue_encoder_outputs,  \
        dialogue_encoder_hidden_state, \
        dialogue_encoder_max_output = self.dialogue_encoder(dialogue_encoder_inputs,
                                                            dialogue_encoder_inputs_length,
                                                            dialogue_encoder_hidden_state)
        # fact encoder
        if self.model_type == 'kg':
            pass


        # dialogue decoder
        # dialogue_encoder_hidden_state -> [num_layers * num_directions, batch, hidden_size]
        #  dialogue_decoder_hidden_state = tuple([item[:2, :, :] + item[2:, :, :] for item in dialogue_encoder_hidden_state])
        dialogue_decoder_hidden_state = self.reduce_state(dialogue_encoder_hidden_state)

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
                dialogue_decoder_outputs.append(dialogue_encoder_output)
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

        # decoder
        #  dialogue_decoder_hidden_state = tuple([item[:2, :, :] + item[2:, :, :] for item in dialogue_encoder_hidden_state])
        dialogue_decoder_hidden_state = self.reduce_state(dialogue_encoder_hidden_state)

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
                 decode_type='greedy', # or beam search
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

        # decoder
        dialogue_decoder_hidden_state = tuple([item[:2, :, :] + item[2:, :, :] for item in dialogue_encoder_hidden_state])

        if decode_type == 'greedy':
            dialogue_decode_outputs = []
            for i in range(max_len):
                dialogue_decoder_output, dialogue_decoder_hidden_state, attn_weights = self.decoder(dialogue_decoder_input,
                                                                                                    dialogue_decoder_hidden_state,
                                                                                                    dialogue_encoder_outputs)

                dialogue_decoder_input = torch.argmax(dialogue_decoder_output, dim=2).detach() #[1, batch_size]
                dialogue_decoder_outputs.append(dialogue_decoder_input)

                ni = dialogue_decoder_input[0][0].item()
                if ni == eosid:
                    break

            # [len, batch_size]
            dialogue_decode_outputs = torch.cat(dialogue_decode_outputs, dim=0)
            dialogue_decode_outputs.transpose_(0, 1)
        elif decode_type == 'beam_search':
            pass

        return dialogue_decode_outputs


