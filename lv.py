# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.simple_encoder import SimpleEncoder
from modules.self_attn import SelfAttentive
from modules.session_encoder import SessionEncoder
from modules.reduce_state import ReduceState
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.utils import init_linear_wt, init_wt_normal
from modules.utils import sequence_mask
from modules.beam_search_original import beam_decode

"""
Latent Variable Dialogue Models and their Diversity
"""


class LV(nn.Module):
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
                 latent_size,
                 num_layers,
                 encoder_num_layers,
                 decoder_num_layers,
                 bidirectional,
                 turn_num,
                 turn_type,
                 decoder_type,
                 attn_type,
                 dropout,
                 padid,
                 tied,
                 device,
                 pre_trained_weight=None):
        super(LV, self).__init__()

        self.vocab_size = vocab_size
        self.model_type = model_type
        self.embedding_size = embedding_size
        self.pre_embedding_size = pre_embedding_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.latent_size = latent_size
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
            padid
        )

        if pre_trained_weight is not None:
            encoder_embedding.weight.data.copy_(pre_trained_weight)
        else:
            init_wt_normal(encoder_embedding.weight,
                           encoder_embedding.embedding_dim)

        # h_encoder
        self.self_attn_encoder = SelfAttentive(
            encoder_embedding,
            rnn_type,
            num_layers,
            bidirectional,
            hidden_size,
            dropout=dropout
        )

        self.simple_encoder = SimpleEncoder(vocab_size,
                                            encoder_embedding,
                                            rnn_type,
                                            hidden_size,
                                            encoder_num_layers,
                                            bidirectional,
                                            dropout)

        self.session_encoder = SessionEncoder(
            rnn_type,
            hidden_size,
            num_layers,
            bidirectional,
            dropout,
        )

        self.mean_linear = nn.Linear(hidden_size * 2, latent_size)
        self.logvar_linear = nn.Linear(hidden_size * 2, latent_size)
        init_linear_wt(self.mean_linear)
        init_linear_wt(self.logvar_linear)

        # fact encoder
        if model_type == 'kg':
            if pre_embedding_size != hidden_size:
                self.f_embedded_linear = nn.Linear(
                    pre_embedding_size, hidden_size)
                init_linear_wt(self.f_embedded_linear)

            # mi = A * ri    fact_linearA(300, 512)
            self.fact_linearA = nn.Linear(hidden_size, hidden_size)
            init_linear_wt(self.fact_linearA)
            # ci = C * ri
            self.fact_linearC = nn.Linear(hidden_size, hidden_size)
            init_linear_wt(self.fact_linearC)

        # encoder hidden_state -> decoder hidden_state
        self.reduce_state = ReduceState(rnn_type)

        if share_embedding:
            decoder_embedding = encoder_embedding
        else:
            decoder_embedding = nn.Embedding(
                vocab_size,
                embedding_size,
                padid
            )
            if pre_trained_weight is not None:
                decoder_embedding.weight.data.copy_(pre_trained_weight)
            else:
                init_wt_normal(decoder_embedding.weight,
                               decoder_embedding.embedding_dim)

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
                h_inputs_length,
                h_inputs_position,
                decoder_inputs,
                decoder_targets,
                decoder_inputs_length,
                f_embedded_inputs,
                f_embedded_inputs_length,
                f_ids_inputs,
                f_ids_inputs_length,
                batch_size,
                r_max_len,
                teacher_forcing_ratio):
        '''
        input:
            h_inputs: # [max_len, batch_size, turn_num]
            h_turns_length: [batch_size]
            h_inputs_length: [batch_size, turn_num]

            decoder_inputs: [r_max_len, batch_size], first step: [sos * batch_size]

            f_embedded_inputs: [batch_size, r_max_len, topk]
            f_embedded_inputs_length: [batch_size]
        '''

        h_encoder_outputs, h_encoder_hidden_state, h_decoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_position,
            batch_size
        )
        h_encoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            h_encoder_hidden_state = self.f_forward(f_embedded_inputs,
                                                    f_embedded_inputs_length,
                                                    f_ids_inputs,
                                                    f_ids_inputs_length,
                                                    h_encoder_hidden_state,
                                                    batch_size)

        r_encoder_outputs, r_encoder_hidden_state = self.simple_encoder(
            decoder_targets,
            decoder_inputs_length
        )
        r_encoder_hidden_state = self.reduce_state(r_encoder_hidden_state)

        # [num_layers, batch_size, hidden_size * 2]
        concat_encoder_hidden_state = torch.cat(
            (h_encoder_hidden_state, r_encoder_hidden_state), dim=2)

        # [num_layers, batch_size, latent_size]
        mean = self.mean_linear(concat_encoder_hidden_state)
        # [num_layers, batch_size, latent_size]
        logvar = self.logvar_linear(concat_encoder_hidden_state)

        std = torch.exp(0.5 * logvar)

        # reparameterize
        z = torch.randn_like(mean)

        z = z * std + mean

        # decoder

        decoder_outputs, decoder_hidden_state, attn_weights = self.decoder(inputs=decoder_inputs,
                                                                           hidden_state=h_encoder_hidden_state,
                                                                           inputs_length=decoder_inputs_length,
                                                                           z=z)
        return decoder_outputs, mean, logvar

    '''inference'''

    def inference(self,
                  h_inputs,
                  h_turns_length,
                  h_inputs_length,
                  h_inputs_position,
                  decoder_input,
                  f_embedded_inputs,
                  f_embedded_inputs_length,
                  f_ids_inputs,
                  f_ids_inputs_length,
                  decode_type,
                  r_max_len,
                  eosid,
                  batch_size,
                  beam_width,
                  best_n):

        h_encoder_outputs, h_encoder_hidden_state, h_decoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_position,
            batch_size
        )

        h_encoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            h_encoder_hidden_state = self.f_forward(f_embedded_inputs,
                                                    f_embedded_inputs_length,
                                                    f_ids_inputs,
                                                    f_ids_inputs_length,
                                                    h_encoder_hidden_state,
                                                    batch_size)

        z = torch.randn((self.decoder_num_layers, batch_size,
                         self.latent_size), device=self.device)

        # decoder
        greedy_outputs = None
        beam_outputs = None
        #  if decode_type == 'greedy':
        greedy_outputs = []
        input = decoder_input
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, attn_weights = self.decoder(inputs=input,
                                                                              hidden_state=h_encoder_hidden_state,
                                                                              z=z)

            input = torch.argmax(
                decoder_output, dim=2).detach()  # [1, batch_size]
            greedy_outputs.append(input)

            if input[0][0].item() == eosid:
                break

        greedy_outputs = torch.cat(greedy_outputs, dim=0)
        greedy_outputs.transpose_(0, 1)

        input = decoder_input
        beam_outputs = beam_decode(
            self.decoder,
            None,
            None,
            h_encoder_hidden_state,
            input,
            batch_size,
            beam_width,
            best_n,
            eosid,
            r_max_len,
            self.vocab_size,
            self.device,
            z=z
        )

        return greedy_outputs, beam_outputs

    def h_forward(self,
                  h_inputs,
                  h_turns_length,
                  h_inputs_length,
                  h_inputs_position,
                  batch_size):
        """history forward
        Args:
            h_inputs: # [max_len, batch_size, turn_num]
            h_turns_length: [batch_size]
            h_inputs_length: [batch_size, turn_num]
        turn_type:
        """
        stack_outputs = []
        #  stack_hidden_states = []
        for ti in range(self.turn_num):
            inputs = h_inputs[:, :, ti]  # [max_len, batch_size]
            inputs_length = h_inputs_length[:, ti]  # [batch_size]
            outputs, hidden_state = self.self_attn_encoder(
                inputs, inputs_length)
            stack_outputs.append(outputs[-1].unsqueeze(0))

        # [turn_num, batch_size, hidden_size]
        stack_outputs = torch.cat(stack_outputs, dim=0)
        session_outputs, session_hidden_state = self.session_encoder(
            stack_outputs, h_turns_length)  # [1, batch_size, hidden_size]
        return session_outputs, session_hidden_state, h_turns_length

    def f_forward(self,
                  f_embedded_inputs,
                  f_embedded_inputs_length,
                  f_ids_inputs,
                  f_ids_inputs_length,
                  hidden_state,
                  batch_size):
        """
        Args:
            - f_embedded_inputs: [batch_size, topk, embedding_size]
            - hidden_state: [num_layers, batch_size, hidden_size]
            -f_ids_inputs: [max_len, batch_size, topk]
            -f_ids_inputs_length: [batch_size, topk]
            -hidden_state: [num_layers, batch_size, hidden_size]
        """

        # [batch_size, topk, embedding_size] -> [batch_size, topk, hidden_size]
        if self.pre_embedding_size != self.hidden_size:
            f_embedded_inputs = self.f_embedded_linear(f_embedded_inputs)

        # M [batch_size, topk, hidden_size]
        fM = self.fact_linearA(f_embedded_inputs)

        # C [batch_size, topk, hidden_size]
        fC = self.fact_linearC(f_embedded_inputs)

        # [batch_size, num_layers, topk]
        tmpP = torch.bmm(hidden_state.transpose(0, 1), fM.transpose(1, 2))

        mask = sequence_mask(f_embedded_inputs_length, max_len=tmpP.size(-1))
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        tmpP.masked_fill_(1 - mask, -float('inf'))

        P = F.softmax(tmpP, dim=2)

        o = torch.bmm(P, fC)  # [batch_size, num_layers, hidden_size]
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

        return decoder
