# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import random

from modules.normal_cnn import NormalCNN
from modules.normal_encoder import NormalEncoder
from modules.self_attn import SelfAttentive
from modules.session_encoder import SessionEncoder
from modules.beam import Beam
from modules.utils import init_linear_wt, init_wt_normal
import modules.transformer as transformer

from misc.vocab import PAD_ID, SOS_ID, EOS_ID

"""
Conversation model,
including encoder and decoder.
"""


class ConversationModel(nn.Module):
    def __init__(self,
                 config,
                 device='cuda'):
        super(Conversation, self).__init__()

        self.config = config
        self.device = device

        enc_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            PADID
        )

        dec_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            PAD_ID
        )

        self.c_encoder = SelfAttentive(
            config,
            encoder_embedding
        )

        self.f_encoder = NormalCNN(
            config,
            encoder_embedding
        )

        self.q_encoder = transformer.Encoder(
            config,
            encoder_embedding
        )

        self.decoder = transformer.Decoder(
            config,
            dec_embedding
        )

        self.c_encoder.embedding.weight = self.c_encoder.embedding.weight
        self.f_encoder.embedding.weight = self.c_encoder.embedding.weight

    self.decoder.embedding.weight = self.c_encoder.embedding.weight

        self.output_linear = nn.Linear(
            config.transformer_size + config.hidden_size * 2,
            config.vocab_size,
            bias=False
        )
        nn.init.xavier_normal_(self.output_linear.weight)

        # [batch_size, max_len, vocab_size]
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self,
                h_inputs,
                h_inputs_length,
                h_inputs_pos,
                h_turns_length,
                dec_inputs,
                dec_inputs_length,
                dec_inputs_pos,
                f_inputs,
                f_inputs_length,
                f_topks_length):
        '''
        input:
            h_inputs: # [turn_num, max_len, batch_size]
            h_inputs_length: [turn_num, batch_size]
            h_turns_length: [batch_size]

            dec_inputs: [r_max_len, batch_size], first step: [sos * batch_size]

            f_inputs: [topk, batch_size, f_max_len]
            f_inputs_length: [batch_size, f_max_len]
            f_topks_length: [batch_size]
        '''
        # c forward
        c_inputs, c_turn_length, c_inputs_length, c_inputs_pos = (
            h_inputs[:-1], h_turns_length - 1, h_inputs_length[:-1], h_inputs_position[:-1])

        c_encoder_outputs = self.c_forward(
            c_inputs,
            c_inputs_length,
            c_turn_length,
            c_inputs_pos
        )

        # fact encoder
        f_encoder_outputs = self.f_forward(f_inputs, f_inputs_length)

        # query forward
        q_input, q_input_length, q_input_pos = h_inputs[-1], h_inputs_length[-1], h_inputs_position[-1:]

        q_enc_outputs = self.q_encoder(q_input, q_input_pos)

        # decoder [batch_size, max_len, transformer_size]
        dec_outputs = self.decoder(
            dec_inputs,
            dec_inputs_pos,
            q_input,
            q_enc_outputs
        )


        return dec_outputs

    '''evaluate'''

    def evaluate(self,
                 h_inputs,
                 h_turns_length,
                 h_inputs_length,
                 h_inputs_pos,
                 f_inputs,
                 f_inputs_length,
                 f_topks_length):
        '''
        h_inputs: # [turn_num, max_len, batch_size]
        h_inputs_length: [turn_num, batch_size]
        h_turns_length: [batch_size]
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        dec_input: [1, batch_size], maybe: [sos * 1]
        '''
        h_encoder_outputs, h_encoder_hidden_state, h_encoder_lengths = self.c_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_pos,
        )

        if h_encoder_hidden_state is None:
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(
                0).repeat(self.config.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        f_encoder_outputs, f_encoder_lengths = None, f_topks_length
        if self.config.model_type == 'kg':
            """
            f_encoder_outputs = self.f_forward(
                f_inputs,
                f_inputs_length,
                f_topks_length,
            )
            decoder_hidden_state += f_encoder_hidden_state
            """
            #  f_encoder_outputs = self.f_embedding_forward(f_inputs)
            f_encoder_outputs = self.f_forward(f_inputs, f_inputs_length)

        # decoder
        dec_outputs = []
        dec_input = torch.ones((1, self.config.batch_size),
                               dtype=torch.long, device=self.device) * SOS_ID
        for i in range(self.config.r_max_len):
            decoder_output, decoder_hidden_state, _ = self.decoder(dec_input,
                                                                   decoder_hidden_state,
                                                                   h_encoder_outputs,
                                                                   h_encoder_lengths,
                                                                   f_encoder_outputs,
                                                                   f_encoder_lengths)

            dec_input = torch.argmax(
                decoder_output, dim=2).detach()  # [1, batch_size]
            dec_outputs.append(decoder_output)

        dec_outputs = torch.cat(dec_outputs, dim=0)

        return dec_outputs

    '''decode'''

    def decode(self,
               h_inputs,
               h_turns_length,
               h_inputs_length,
               h_inputs_pos,
               f_inputs,
               f_inputs_length,
               f_topks_length):

        h_encoder_outputs, h_encoder_hidden_state, h_encoder_lengths = self.c_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_pos,
        )

        if h_encoder_hidden_state is None:
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(
                0).repeat(self.config.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # f encoder
        f_encoder_outputs, f_encoder_lengths = None, f_topks_length
        if self.config.model_type == 'kg':
            """
            f_encoder_outputs = self.f_forward(
                f_inputs,
                f_inputs_length,
                f_topks_length,
            )
            decoder_hidden_state += f_encoder_hidden_state
            """
            #  f_encoder_outputs = self.f_embedding_forward(f_inputs)
            f_encoder_outputs = self.f_forward(f_inputs, f_inputs_length)

        # decoder
        beam_outputs, beam_score, beam_length = self.beam_decode(
            decoder_hidden_state,
            h_encoder_outputs,
            h_encoder_lengths,
            f_encoder_outputs,
            f_encoder_lengths
        )

        greedy_outputs = self.greedy_decode(
            decoder_hidden_state,
            h_encoder_outputs,
            h_encoder_lengths,
            f_encoder_outputs,
            f_encoder_lengths
        )

        return greedy_outputs, beam_outputs, beam_length

    def greedy_decode(self,
                      hidden_state,
                      h_encoder_outputs,
                      h_encoder_lengths,
                      f_encoder_outputs,
                      f_encoder_lengths):

        greedy_outputs = []
        input = torch.ones((1, self.config.batch_size),
                           dtype=torch.long, device=self.device) * SOS_ID
        for i in range(self.config.r_max_len):
            output, hidden_state, _ = self.decoder(input,
                                                   hidden_state,
                                                   h_encoder_outputs,
                                                   h_encoder_lengths,
                                                   f_encoder_outputs,
                                                   f_encoder_lengths)

            input = torch.argmax(output, dim=2).detach().view(
                1, -1)  # [1, batch_size]
            greedy_outputs.append(input)

            if input[0][0].item() == EOS_ID
              break

        # [len, batch_size]  -> [batch_size, len]
        greedy_outputs = torch.cat(greedy_outputs, dim=0).transpose(0, 1)

        return greedy_outputs

    def beam_decode(self,
                    hidden_state=None,
                    h_encoder_outputs=None,
                    h_encoder_lengths=None,
                    f_encoder_outputs=None,
                    f_encoder_lengths=None):
        '''
        Args:
            hidden_state : [num_layers, batch_size, hidden_size] (optional)
            h_encoder_outputs : [max_len, batch_size, hidden_size]
            h_encoder_lengths : [batch_size] (optional)

        Return:
            prediction: [batch_size, beam, max_len]
        '''
        batch_size, beam_size = self.config.batch_size, self.config.beam_size
        # [1, batch_size x beam_size]
        input = torch.ones(batch_size * beam_size,
                           dtype=torch.long, device=self.device) * SOS_ID

        # [num_layers, batch_size x beam_size, hidden_size]
        hidden_state = hidden_state.repeat(1, beam_size, 1)

        if h_encoder_outputs is not None:
            h_encoder_outputs = h_encoder_outputs.repeat(1, beam_size, 1)
            h_encoder_lengths = h_encoder_lengths.repeat(beam_size)

        if f_encoder_outputs is not None:
            f_encoder_outputs = f_encoder_outputs.repeat(1, beam_size, 1)
            f_encoder_lengths = f_encoder_lengths.repeat(beam_size)

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
            self.config.beam_size,
            self.config.r_max_len,
            batch_position,
            EOS_ID
        )

        for i in range(self.config.r_max_len):
            output, hidden_state, _ = self.decoder(input.unsqueeze(0).contiguous(),
                                                   hidden_state,
                                                   h_encoder_outputs,
                                                   h_encoder_lengths,
                                                   f_encoder_outputs,
                                                   f_encoder_lengths)

            # output: [1, batch_size * beam_size, vocab_size]
            # -> [batch_size * beam_size, vocab_size]
            log_prob = output[0]
            print('log_prob: ', log_prob.shape)

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
            eos_idx = input.data.eq(EOS_ID).view(batch_size, beam_size)

            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length

    def c_forward(self,
                  h_inputs,
                  h_turns_length,
                  h_inputs_length,
                  h_inputs_pos):
        """history forward
        Args:
            h_inputs: # [turn_num, max_len, batch_size]
            h_inputs_length: [turn_num, batch_size]
            h_turns_length: [batch_size]
            h_inputs_pos: [batch_s]
        turn_type:
        """
          stack_outputs = list()
           for ti in range(self.config.turn_num):
                inputs = h_inputs[ti, :, :]  # [batch_size, max_len]
                # [batch_size, max_len]
                inputs_position = h_inputs_pos[ti, :, :]
                # [batch_size, max_len, transformer_size]
                outputs = self.c_encoder(inputs, inputs_position)

                # [batch_size, max_len, transformer_size]
                stack_outputs.append(outputs)

            # [turn_num, batch_size, transformer_size * turn_num]
            stack_outputs = torch.cat(stack_outputs, dim=2)
            return stack_outputs

    def f_forward(self,
                  f_inputs,
                  f_inputs_length):
        """
        Args:
            -f_inputs: [batch_sizes, topk, max_len]
            -f_inputs_length: [batch_size, topk]
        """
        f_outputs = list()
        for i in range(f_inputs.size(1)):
            f_input = f_inputs[:, i, :]  # [batch_size, max_len]
            f_input_length = f_inputs_length[:, i]  # [batch_size]

            # [batch_size, 1, max_len]
            output, _ = self.f_encoder(f_input, f_input_length)

            f_outputs.append(output)

        # [batch_size, topk, hidden_size]
        f_outputs = torch.cat(f_outputs, dim=1)
        return f_outputs
