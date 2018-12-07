# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.normal_cnn import NormalCNN
from modules.normal_encoder import NormalEncoder
from modules.self_attn import SelfAttentive
from modules.session_encoder import SessionEncoder
from modules.reduce_state import ReduceState
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.beam import Beam
from modules.attention import Attention
from modules.utils import init_linear_wt

from misc.vocab import PAD_ID, SOS_ID, EOS_ID

"""
KGModel
"""


class KGModel(nn.Module):
    '''
    generating responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 config,
                 device='cuda'):
        super(KGModel, self).__init__()

        self.config = config
        self.device = device

        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.forward_step = 0

        enc_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            PAD_ID
        )

        dec_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            PAD_ID
        )

        # c_encoder
        self.c_encoder = None
        if config.turn_type == 'self_attn':
            self.c_encoder = SelfAttentive(
                config,
                enc_embedding
            )
        elif config.turn_type in ['none', 'concat']:
            pass
        elif config.turn_type == 'cnn':
            self.c_encoder = NormalCNN(
                config,
                enc_embedding,
                type='context'
            )
        else:
            self.c_encoder = NormalEncoder(
                config,
                enc_embedding
            )

        # q encoder
        self.q_encoder = NormalEncoder(
            config,
            enc_embedding
        )

        self.f_encoder = None
        if config.model_type == 'kg':
            self.f_encoder = NormalCNN(
                config,
                enc_embedding
            )

        # encoder hidden_state -> decoder hidden_state
        self.reduce_state = ReduceState(config.rnn_type)

        # session encoder
        if config.turn_type == 'session':
            self.session_encoder = SessionEncoder(config)

        if config.turn_type not in ['none', 'concat']:
            self.c_attn = Attention(config.hidden_size)
            self.c_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
            init_linear_wt(self.c_linear)

        # decoder
        self.decoder = LuongAttnDecoder(config, dec_embedding)

        # encoder embedding share
        if self.c_encoder is not None:
            self.c_encoder.embedding.weight = self.q_encoder.embedding.weight

        if self.f_encoder is not None:
            self.f_encoder.embedding.weight = self.q_encoder.embedding.weight

        # encoder, decode embedding share
        if config.share_embedding:
            self.decoder.embedding.weight = self.q_encoder.embedding.weight

    def forward(self,
                q_inputs,
                q_inputs_length,
                c_inputs,
                c_inputs_length,
                c_turn_length,
                dec_inputs,
                f_inputs,
                f_inputs_length,
                f_topk_length):
        '''
        Args:
            q_inputs: [max_len, batch_size]

            c_inputs: # [turn_num-1, max_len, batch_size] [c1, c2, ..., q]
            c_inputs_length: [turn_num, batch_size]
            c_turn_length: [batch_size]

            dec_inputs: [r_max_len, batch_size], first step: [sos * batch_size]

            f_inputs: [f_max_len, batch_size, topk]
            f_inputs_length: [f_max_len, batch_size, topk]
            f_topk_length: [batch_size]
            f_embedded_inputs_length: [batch_size]
        '''
        c_enc_outputs = None
        if self.config.turn_type not in ['none', 'concat']:
            # [turn_num-1, batch_size, hidden_size]
            c_enc_outputs = self.c_forward(
                c_inputs,
                c_inputs_length,
                c_turn_length,
            )

        # fact encoder
        f_enc_outputs = None
        if self.config.model_type == 'kg':
            f_enc_outputs = self.f_forward(f_inputs, f_inputs_length)

        # [max_len, batch_size, hidden_size]
        q_enc_outputs, q_encoder_hidden = self.q_encoder(
            q_inputs,
            q_inputs_length
        )

        # init decoder hidden [num_layers, batch_size, hidden_size]
        dec_hidden = self.reduce_state(q_encoder_hidden)

        if c_enc_outputs is not None:
            # q_enc_outputs, c_enc_outputs attention
            c_context, _ = self.c_attn(dec_hidden, c_enc_outputs)
            #  print('c_context: ', c_context.shape)
            #  print('dec_hidden: ', dec_hidden.shape)

            #  dec_hidden = torch.cat((dec_hidden, c_context), dim=2)
            #  dec_hidden = self.c_linear(dec_hidden) # [num_layers, batch_size, hidden_size]

            #  dec_hidden += c_context
            dec_hidden = torch.add(dec_hidden, c_context)

            c_enc_outputs = None

        # decoder
        dec_outputs, dec_hidden, _ = self.decoder(
            dec_inputs,
            dec_hidden,
            q_enc_outputs=q_enc_outputs,
            q_enc_length=q_inputs_length,
            c_enc_outputs=c_enc_outputs,
            c_enc_length=c_turn_length,
            f_enc_outputs=f_enc_outputs,
            f_enc_length=f_topk_length
        )

        # [max_len * batch_, vocab_size]
        dec_outputs = dec_outputs.view(-1, dec_outputs.size(-1)).contiguous()
        #  print('dec_outputs: ', dec_outputs)

        return dec_outputs

    '''decode'''

    def decode(self,
               q_inputs,
               q_inputs_length,
               c_inputs,
               c_inputs_length,
               c_turn_length,
               f_inputs,
               f_inputs_length,
               f_topk_length):

        c_enc_outputs = None
        if self.config.turn_type not in ['none', 'concat']:
            # [turn_num-1, batch_size, hidden_size]
            c_enc_outputs = self.c_forward(
                c_inputs,
                c_inputs_length,
                c_turn_length,
            )

        # fact encoder
        f_enc_outputs = None
        if self.config.model_type == 'kg':
            f_enc_outputs = self.f_forward(f_inputs, f_inputs_length)

        # query forward
        q_enc_outputs, q_encoder_hidden = self.q_encoder(
            q_inputs,
            q_inputs_length
        )

        # init decoder hidden
        dec_hidden = self.reduce_state(q_encoder_hidden)

        if c_enc_outputs is not None:
            # q_enc_outputs, c_enc_outputs attention
            c_context, _ = self.c_attn(dec_hidden, c_enc_outputs)

            #  dec_hidden = torch.cat((dec_hidden, c_context), dim=2)
            #  dec_hidden = self.c_linear(dec_hidden) # [num_layers, batch_size, hidden_size]

            dec_hidden = torch.add(dec_hidden, c_context)

            c_enc_outputs = None

        # decoder
        beam_outputs, beam_score, beam_length = self.beam_decode(
            dec_hidden,
            q_enc_outputs,
            q_inputs_length,
            c_enc_outputs,
            c_turn_length,
            f_enc_outputs,
            f_topk_length
        )

        greedy_outputs = self.greedy_decode(
            dec_hidden,
            q_enc_outputs,
            q_inputs_length,
            c_enc_outputs,
            c_turn_length,
            f_enc_outputs,
            f_topk_length
        )

        return greedy_outputs, beam_outputs, beam_length

    def greedy_decode(self,
                      dec_hidden,
                      q_enc_outputs,
                      q_inputs_length,
                      c_enc_outputs,
                      c_turn_length,
                      f_enc_outputs,
                      f_topk_length):

        greedy_outputs = []
        dec_input = torch.ones((1, self.config.batch_size),
                               dtype=torch.long, device=self.device) * SOS_ID

        for i in range(self.config.r_max_len):
            output, dec_hidden,  _ = self.decoder(
                dec_input,
                dec_hidden,
                q_enc_outputs=q_enc_outputs,
                q_enc_length=q_inputs_length,
                c_enc_outputs=c_enc_outputs,
                c_enc_length=c_turn_length,
                f_enc_outputs=f_enc_outputs,
                f_enc_length=f_topk_length
            )
            output = F.log_softmax(output, dim=2)
            dec_input = torch.argmax(output, dim=2).detach().view(1, -1)  # [1, batch_size]
            greedy_outputs.append(dec_input)

        # [len, batch_size]  -> [batch_size, len]
        greedy_outputs = torch.cat(greedy_outputs, dim=0).transpose(0, 1)

        return greedy_outputs

    def beam_decode(self,
                    dec_hidden,
                    q_enc_outputs,
                    q_inputs_length,
                    c_enc_outputs,
                    c_turn_length,
                    f_enc_outputs,
                    f_topk_length):
        '''
        Args:
            dec_hidden : [num_layers, batch_size, hidden_size] (optional)
            c_enc_outputs : [max_len, batch_size, hidden_size]
            c_encoder_lengths : [batch_size] (optional)

        Return:
            prediction: [batch_size, beam, max_len]
        '''
        batch_size, beam_size = self.config.batch_size, self.config.beam_size

        # [1, batch_size x beam_size]
        dec_input = torch.ones(1, batch_size * beam_size,
                               dtype=torch.long,
                               device=self.device) * SOS_ID

        # [num_layers, batch_size * beam_size, hidden_size]
        dec_hidden = dec_hidden.repeat(1, beam_size, 1)

        q_enc_outputs = q_enc_outputs.repeat(1, beam_size, 1)
        q_inputs_length = q_inputs_length.repeat(beam_size)

        if c_enc_outputs is not None:
            c_enc_outputs = c_enc_outputs.repeat(1, beam_size, 1)
            c_turn_length = c_turn_length.repeat(beam_size)

        if f_enc_outputs is not None:
            f_enc_outputs = f_enc_outputs.repeat(1, beam_size, 1)
            f_topk_length = f_topk_length.repeat(beam_size)

        # [batch_size] [0, beam_size * 1, ..., beam_size * (batch_size - 1)]
        batch_position = torch.arange(
            0, batch_size, dtype=torch.long, device=self.device) * beam_size

        score = torch.ones(batch_size * beam_size,
                           device=self.device) * -float('inf')
        score.index_fill_(0, torch.arange(
            0, batch_size, dtype=torch.long, device=self.device) * beam_size, 0.0)

        # Initialize Beam that stores decisions for backtracking
        beam = Beam(
            batch_size,
            beam_size,
            self.config.r_max_len,
            batch_position,
            EOS_ID
        )

        for i in range(self.config.r_max_len):
            output, dec_hidden, _ = self.decoder(
                dec_input.view(1, -1),
                dec_hidden,
                q_enc_outputs=q_enc_outputs,
                q_enc_length=q_inputs_length,
                c_enc_outputs=c_enc_outputs,
                c_enc_length=c_turn_length,
                f_enc_outputs=f_enc_outputs,
                f_enc_length=f_topk_length
            )

            # output: [1, batch_size * beam_size, vocab_size]
            # -> [batch_size * beam_size, vocab_size]
            log_prob = F.log_softmax(output.squeeze(0), dim=1)
            #  print('log_prob: ', log_prob.shape)

            # score: [batch_size * beam_size, vocab_size]
            score = score.view(-1, 1) + log_prob

            # score [batch_size, beam_size]
            score, top_k_idx = score.view(
                batch_size, -1).topk(beam_size, dim=1)

            # dec_input: [batch_size x beam_size]
            dec_input = (top_k_idx % self.config.vocab_size).view(-1)

            # beam_idx: [batch_size, beam_size]
            # [batch_size, beam_size]
            beam_idx = top_k_idx / self.config.vocab_size

            # top_k_pointer: [batch_size * beam_size]
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            # [num_layers, batch_size * beam_size, hidden_size]
            dec_hidden = dec_hidden.index_select(1, top_k_pointer)

            # Update sequence scores at beam
            beam.update(score.clone(), top_k_pointer, dec_input)

            # Erase scores for EOS so that they are not expanded
            # [batch_size, beam_size]
            eos_idx = dec_input.data.eq(EOS_ID).view(
                batch_size, beam_size)

            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length

    def c_forward(self,
                  c_inputs,
                  c_inputs_length,
                  c_turn_length):
        """
        Args:
            c_inputs: # [turn_num-1, max_len, batch_size]  [c1, c2, ..., q]
            c_inputs_length: [turn_num-1, batch_size]
            c_turn_length: [batch_size]
        turn_type: [max_len, turn_num-1, hidden_size]
        """
        stack_outputs = list()
        # query encode separately.
        for ti in range(self.config.turn_num - 1):
            inputs = c_inputs[ti, :, :]  # [max_len, batch_size]
            inputs_length = c_inputs_length[ti, :]  # [batch_size]

            if self.config.turn_type == 'cnn':
                inputs = inputs.transpose(0, 1)
            outputs, hidden_state = self.c_encoder(inputs, inputs_length, sort=False)

            stack_outputs.append(outputs[-1].unsqueeze(0))

        # [turn_num-1, batch_size, hidden_size]
        stack_outputs = torch.cat(stack_outputs, dim=0)
        #  print('stack_outputs: ', stack_outputs.shape)

        """
        if self.config.turn_type == 'session':
            # [turn_num-1, batch_size, hidden_size]
            session_outputs, session_hidden_state = self.session_encoder(stack_outputs, c_turn_length)
            return session_outputs
        """

        return stack_outputs


    def f_forward(self,
                  f_inputs,
                  f_inputs_length):
        """
        Args:
            -f_inputs: [topk, batch_size, max_len]
            -f_inputs_length: [topk, batch_size]
        """
        f_outputs = list()
        for i in range(f_inputs.size(0)):
            f_input = f_inputs[i, :, :]  # [batch_size, max_len]
            f_input_length = f_inputs_length[i, :]  # [batch_size]
            # [1, batch_size, hidden_size]
            output, _ = self.f_encoder(f_input, f_input_length)

            f_outputs.append(output)

        # [topk, batch_size, hidden_size]
        f_outputs = torch.cat(f_outputs, dim=0)
        return f_outputs
