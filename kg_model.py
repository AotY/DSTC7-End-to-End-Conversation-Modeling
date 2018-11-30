# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn

from modules.normal_cnn import NormalCNN
from modules.normal_encoder import NormalEncoder
from modules.self_attn import SelfAttentive
from modules.session_encoder import SessionEncoder
from modules.reduce_state import ReduceState
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.utils import init_linear_wt, init_wt_normal
from modules.beam import Beam

import modules.transformer.models as transformer_models

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
                 vocab,
                 device='cuda',
                 pre_trained_weight=None):
        super(KGModel, self).__init__()

        self.config = config
        self.vocab = vocab
        self.device = device

        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        self.forward_step = 0

        if pre_trained_weight is not None:
            encoder_embedding = nn.Embedding.from_pretrained(
                pre_trained_weight)
            encoder_embedding.padding_idx = vocab.padid
        else:
            encoder_embedding = nn.Embedding(
                config.vocab_size,
                config.embedding_size,
                vocab.padid
            )

        decoder_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            vocab.padid
        )

        # c_encoder
        if config.turn_type == 'transformer':
            self.c_encoder = transformer_models.Encoder(
                config.c_max_len,
                encoder_embedding,
                num_layers=6,
                num_head=8,
                k_dim=64,
                v_dim=64,
                model_dim=config.hidden_size,
                inner_dim=1024,
                padid=vocab.padid,
                dropout=config.dropout
            )
        elif config.turn_type == 'self_attn':
            self.c_encoder = SelfAttentive(
                config,
                encoder_embedding
            )
        else:
            self.c_encoder = NormalEncoder(
                config,
                encoder_embedding,
            )

        # q encoder
        self.q_encoder = NormalEncoder(
            config,
            encoder_embedding
        )

        self.f_encoder = NormalCNN(
            config,
            encoder_embedding
        )

        # encoder hidden_state -> decoder hidden_state
        self.reduce_state = ReduceState(config.rnn_type)

        # encoder embedding share
        self.c_encoder.embedding.weight.data = self.q_encoder.embedding.weight.data
        self.f_encoder.embedding.weight.data = self.q_encoder.embedding.weight.data

        self.decoder = LuongAttnDecoder(config, decoder_embedding)
        if config.share_embedding:  # encoder, decode embedding share
            self.decoder.embedding.weight.data = self.q_encoder.embedding.weight.data

        if config.turn_type != 'none' or config.turn_type != 'concat':
            if config.turn_type == 'c_concat':
                self.c_concat_linear = nn.Linear(
                    config.hidden_size * config.turn_num, config.hidden_size)
                init_linear_wt(self.c_concat_linear)
            elif config.turn_type in ['sequential', 'weight', 'transformer', 'self_attn']:
                self.session_encoder = SessionEncoder(config)

    def forward(self,
                h_inputs,
                h_inputs_length,
                h_inputs_position,
                h_turns_length,
                decoder_inputs,
                decoder_inputs_length,
                f_inputs,
                f_inputs_length,
                f_topks_length):
        '''
        input:
            h_inputs: # [turn_num, max_len, batch_size] [c1, c2, ..., q]
            h_inputs_length: [turn_num, batch_size]
            h_turns_length: [batch_size]
            h_inputs_position: # [turn_num, max_len, batch_size] [c1, c2, ..., q]

            decoder_inputs: [r_max_len, batch_size], first step: [sos * batch_size]

            f_inputs: [f_max_len, batch_size, topk]
            f_inputs_length: [f_max_len, batch_size, topk]
            f_topks_length: [batch_size]
            f_embedded_inputs_length: [batch_size]
        '''
        # c forward
        c_inputs, c_turn_length, c_inputs_length, c_inputs_position = (h_inputs[:-1], h_turns_length - 1, h_inputs_length[:-1], h_inputs_position[:-1])

        c_encoder_outputs = self.c_forward(
            c_inputs,
            c_inputs_length,
            c_turn_length,
            c_inputs_position
        )

        # query forward
        q_input, q_input_length, q_input_position = h_inputs[-1], h_inputs_length[-1], h_inputs_position[-1:]

        q_encoder_outputs, q_encoder_hidden = self.q_encoder(
            q_input,
            q_input_length
        )

        # init decoder hidden
        decoder_hidden = self.reduce_state(q_encoder_hidden)

        # fact encoder
        f_encoder_outputs = None
        if self.config.model_type == 'kg':
            f_encoder_outputs = self.f_forward(f_inputs, f_inputs_length)

        # decoder
        decoder_outputs = []
        decoder_output = None
        self.update_teacher_forcing_ratio()
        for i in range(0, self.config.r_max_len):
            if i == 0:
                decoder_input = decoder_inputs[i].view(1, -1)
            else:
                use_teacher_forcing = True if random.random(
                ) < self.teacher_forcing_ratio else False
                if use_teacher_forcing:
                    decoder_input = decoder_inputs[i].view(1, -1)
                else:
                    decoder_input = torch.argmax(
                        decoder_output, dim=2).detach().view(1, -1)

            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input,
                decoder_hidden,
                q_encoder_outputs,
                q_input_length,
                c_encoder_outputs,
                c_turn_length,
                f_encoder_outputs,
                f_topks_length
            )

            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        return decoder_outputs

    def reset_teacher_forcing_ratio(self):
        self.forward_step = 0
        self.teacher_forcing_ratio = 1.0

    def update_teacher_forcing_ratio(self, eplison=0.0001, min_t=0.8):
        self.forward_step += 1
        if (self.teacher_forcing_ratio == min_t):
            return
        update_t = self.teacher_forcing_ratio - \
            eplison * (self.forward_step)
        self.teacher_forcing_ratio = max(update_t, min_t)

    '''evaluate'''

    def evaluate(self,
                 h_inputs,
                 h_inputs_length,
                 h_inputs_position,
                 h_turns_length,
                 f_inputs,
                 f_inputs_length,
                 f_topks_length):
        '''
        h_inputs: # [turn_num, max_len, batch_size]
        h_inputs_length: [turn_num, batch_size]
        h_turns_length: [batch_size]
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        # c forward
        c_inputs, c_turn_length, c_inputs_length, c_inputs_position = (h_inputs[:-1], h_turns_length - 1, h_inputs_length[:-1], h_inputs_position[:-1])

        c_encoder_outputs = self.c_forward(
            c_inputs,
            c_inputs_length,
            c_turn_length,
            c_inputs_position
        )

        # query forward
        q_input, q_input_length, q_input_position = h_inputs[-1], h_inputs_length[-1], h_inputs_position[-1:]

        q_encoder_outputs, q_encoder_hidden = self.q_encoder(
            q_input,
            q_input_length
        )

        # init decoder hidden
        decoder_hidden = self.reduce_state(q_encoder_hidden)

        # fact encoder
        f_encoder_outputs = None
        if self.config.model_type == 'kg':
            f_encoder_outputs = self.f_forward(f_inputs, f_inputs_length)

        # decoder
        decoder_outputs = []
        decoder_input = torch.ones(
            1, self.config.batch_size, dtype=torch.long, device=self.device) * self.vocab.sosid
        for i in range(0, self.config.r_max_len):
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input,
                decoder_hidden,
                q_encoder_outputs,
                q_input_length,
                c_encoder_outputs,
                c_turn_length,
                f_encoder_outputs,
                f_topks_length
            )
            decoder_input = torch.argmax(
                decoder_output, dim=2).detach().view(1, -1)

            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        return decoder_outputs

    '''decode'''

    def decode(self,
               h_inputs,
               h_inputs_length,
               h_inputs_position,
               h_turns_length,
               f_inputs,
               f_inputs_length,
               f_topks_length):

        # c forward
        c_inputs, c_turn_length, c_inputs_length, c_inputs_position = (h_inputs[:-1], h_turns_length - 1, h_inputs_length[:-1], h_inputs_position[:-1])

        c_encoder_outputs = self.c_forward(
            c_inputs,
            c_inputs_length,
            c_turn_length,
            c_inputs_position
        )

        # query forward
        q_input, q_input_length, q_input_position = h_inputs[-1], h_inputs_length[-1], h_inputs_position[-1:]

        q_encoder_outputs, q_encoder_hidden = self.q_encoder(
            q_input,
            q_input_length
        )

        # init decoder hidden
        decoder_hidden = self.reduce_state(q_encoder_hidden)

        # fact encoder
        f_encoder_outputs = None
        if self.config.model_type == 'kg':
            f_encoder_outputs = self.f_forward(f_inputs, f_inputs_length)

        # decoder
        beam_outputs, beam_score, beam_length = self.beam_decode(
            decoder_hidden,
            q_encoder_outputs,
            q_input_length,
            c_encoder_outputs,
            c_turn_length,
            f_encoder_outputs,
            f_topks_length
        )

        greedy_outputs = self.greedy_decode(
            decoder_hidden,
            q_encoder_outputs,
            q_input_length,
            c_encoder_outputs,
            c_turn_length,
            f_encoder_outputs,
            f_topks_length
        )

        return greedy_outputs, beam_outputs, beam_length

    def greedy_decode(self,
                      hidden_state,
                      q_encoder_outputs,
                      q_input_length,
                      c_encoder_outputs,
                      c_encoder_length,
                      f_encoder_outputs,
                      f_encoder_length):

        greedy_outputs = []
        input = torch.ones((1, self.config.batch_size),
                           dtype=torch.long, device=self.device) * self.vocab.sosid
        for i in range(self.config.r_max_len):
            output, hidden_state, _ = self.decoder(
                input,
                hidden_state,
                q_encoder_outputs,
                q_input_length,
                c_encoder_outputs,
                c_encoder_length,
                f_encoder_outputs,
                f_encoder_length
            )
            input = torch.argmax(output, dim=2).detach().view(
                1, -1)  # [1, batch_size]
            greedy_outputs.append(input)

            # eos problem
            #  if input[0][0].item() == self.vocab.eosid:
            #  break
            #  eos_index = input[0].eq(self.vocab.eosid)

        # [len, batch_size]  -> [batch_size, len]
        greedy_outputs = torch.cat(greedy_outputs, dim=0).transpose(0, 1)

        return greedy_outputs

    def beam_decode(self,
                    hidden_state,
                    q_encoder_outputs,
                    q_input_length,
                    c_encoder_outputs,
                    c_encoder_length,
                    f_encoder_outputs,
                    f_encoder_length):
        '''
        Args:
            hidden_state : [num_layers, batch_size, hidden_size] (optional)
            c_encoder_outputs : [max_len, batch_size, hidden_size]
            c_encoder_lengths : [batch_size] (optional)

        Return:
            prediction: [batch_size, beam, max_len]
        '''
        batch_size, beam_size = self.config.batch_size, self.config.beam_size

        # [1, batch_size x beam_size]
        input = torch.ones(1, batch_size * beam_size,
                           dtype=torch.long,
                           device=self.device) * self.vocab.sosid

        # [num_layers, batch_size x beam_size, hidden_size]
        hidden_state = hidden_state.repeat(1, beam_size, 1)

        if q_encoder_outputs is not None:
            q_encoder_outputs = q_encoder_outputs.repeat(1, beam_size, 1)
            q_input_length = q_input_length.repeat(beam_size)

        if c_encoder_outputs is not None:
            c_encoder_outputs = c_encoder_outputs.repeat(1, beam_size, 1)
            c_encoder_length = c_encoder_length.repeat(beam_size)

        if f_encoder_outputs is not None:
            f_encoder_outputs = f_encoder_outputs.repeat(1, beam_size, 1)
            f_encoder_length = f_encoder_length.repeat(beam_size)

        # [batch_size] [0, beam_size * 1, ..., beam_size * (batch_size - 1)]
        batch_position = torch.arange(0, batch_size, dtype=torch.long, device=self.device) * beam_size

        score = torch.ones(batch_size * beam_size, device=self.device) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size, dtype=torch.long, device=self.device) * beam_size, 0.0)

        # Initialize Beam that stores decisions for backtracking
        beam = Beam(
            batch_size,
            beam_size,
            self.config.r_max_len,
            batch_position,
            self.vocab.eosid
        )

        for i in range(self.config.r_max_len):
            output, hidden_state, _ = self.decoder(input.view(1, -1),
                                                   hidden_state,
                                                   q_encoder_outputs,
                                                   q_input_length,
                                                   c_encoder_outputs,
                                                   c_encoder_length,
                                                   f_encoder_outputs,
                                                   f_encoder_length)

            # output: [1, batch_size * beam_size, vocab_size]
            # -> [batch_size * beam_size, vocab_size]
            log_prob = output.squeeze(0)
            #  print('log_prob: ', log_prob.shape)

            # score: [batch_size * beam_size, vocab_size]
            score = score.view(-1, 1) + log_prob

            # score [batch_size, beam_size]
            score, top_k_idx = score.view(
                batch_size, -1).topk(beam_size, dim=1)

            # input: [batch_size x beam_size]
            input = (top_k_idx % self.config.vocab_size).view(-1)

            # beam_idx: [batch_size, beam_size]
            # [batch_size, beam_size]
            beam_idx = top_k_idx / self.config.vocab_size

            # top_k_pointer: [batch_size * beam_size]
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            # [num_layers, batch_size * beam_size, hidden_size]
            hidden_state = hidden_state.index_select(1, top_k_pointer)

            # Update sequence scores at beam
            beam.update(score.clone(), top_k_pointer, input)

            # Erase scores for EOS so that they are not expanded
            # [batch_size, beam_size]
            eos_idx = input.data.eq(self.vocab.eosid).view(
                batch_size, beam_size)

            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length

    def c_forward(self,
                  c_inputs,
                  c_inputs_length,
                  c_turns_length,
                  c_inputs_position):
        """history forward
        Args:
            c_inputs: # [turn_num, max_len, batch_size]  [c1, c2, ..., q]
            c_inputs_length: [turn_num, batch_size]
            c_turns_length: [batch_size]
        turn_type:
        """
        if self.config.turn_type == 'concat' or self.config.turn_type == 'none':
            inputs = c_inputs[0, :, :]  # [max_len, batch_size]
            inputs_length = c_inputs_length[0, :]

            # [max_len, batch_size, hidden_size]
            outputs, hidden_state = self.c_encoder(inputs, inputs_length)
            return outputs, hidden_state, inputs_length
        else:
            stack_outputs = list()
            # query encode separately.
            for ti in range(self.config.turn_num - 1):
                inputs = c_inputs[ti, :, :]  # [max_len, batch_size]
                if self.config.turn_type == 'transformer':
                    # [max_len, batch_size]
                    inputs_position = c_inputs_position[ti, :, :]
                    outputs = self.c_encoder(inputs.transpose(
                        0, 1), inputs_position.transpose(0, 1))
                    # [batch_size, max_len, hidden_size]
                    outputs = outputs.transpose(0, 1)
                elif self.config.turn_type == 'self_attn':
                    inputs_length = c_inputs_length[ti, :]  # [batch_size]
                    outputs, hidden_state = self.c_encoder(
                        inputs, inputs_length)
                    # [1, batch_size, hidden_size]
                    outputs = outputs.unsqueeze(0)
                else:
                    inputs_length = c_inputs_length[ti, :]  # [batch_size]
                    outputs, hidden_state = self.c_encoder(
                        inputs, inputs_length)

                stack_outputs.append(outputs[-1].unsqueeze(0))

            if self.config.turn_type == 'sum':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                # [1, batch_size, hidden_size]
                return stack_outputs.sum(dim=0)
            elif self.config.turn_type == 'c_concat':
                # [1, hidden_size * turn_num]
                c_concat_outputs = torch.cat(stack_outputs, dim=2)
                # [1, batch_size, hidden_size]
                return self.c_concat_linear(c_concat_outputs)
            elif self.config.turn_type == 'sequential':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, c_turns_length)  # [1, batch_size, hidden_size]
                return session_outputs[-1].unsqueeze(0)
            elif self.config.turn_type == 'weight':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, c_turns_length)  # [1, batch_size, hidden_size]
                return session_outputs
            elif self.config.turn_type == 'transformer':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, _ = self.session_encoder(
                    stack_outputs, c_turns_length)  # [1, batch_size, hidden_size]
                return session_outputs
            elif self.config.turn_type == 'self_attn':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                # session_hidden_state: [num_layers, batch_size, hidden_size]
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, c_turns_length)
                return session_outputs

    def f_forward(self,
                  f_inputs,
                  f_inputs_length):
        """
        Args:
            -f_inputs: [topk, max_len, batch_size]
            -f_inputs_length: [topk, batch_size]
            -f_topks_length: [batch_size]
            -hidden_state: [num_layers, batch_size, hidden_size]
        """
        f_outputs = list()
        for i in range(f_inputs.size(0)):
            f_input = f_inputs[i, :, :]  # [max_len, batch_size]
            f_input_length = f_inputs_length[i, :]  # [batch_size]

            #  outputs, hidden_state = self.f_encoder(f_input, f_input_length)

            output, _ = self.f_encoder(f_input, f_input_length)

            """
            # outputs: [hidden_size, batch_size, max_len]
            _, outputs, _ = self.f_encoder(f_input, f_input_length)
            #  print('outputs: ', outputs.shape)
            outputs = outputs.permute(2, 1, 0)
            """

            f_outputs.append(output)

        # [topk, batch_size, hidden_size]
        f_outputs = torch.cat(f_outputs, dim=0)
        return f_outputs
