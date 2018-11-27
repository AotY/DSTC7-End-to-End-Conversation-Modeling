# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn

from modules.cnn_encoder import CNNEncoder
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
            self.encoder_embedding = nn.Embedding.from_pretrained(pre_trained_weight)
            self.encoder_embedding.padding_idx = vocab.padid
        else:
            self.encoder_embedding = nn.Embedding(
                config.vocab_size,
                config.embedding_size,
                vocab.padid
            )
            init_wt_normal(self.encoder_embedding.weight,
                           self.encoder_embedding.embedding_dim)


        # h_encoder
        if config.turn_type == 'transformer':
            self.h_encoder = transformer_models.Encoder(
                config.c_max_len,
                self.encoder_embedding,
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
            self.h_encoder = SelfAttentive(
                config,
                self.encoder_embedding
            )
        else:
            self.h_encoder = NormalEncoder(
                config,
                self.encoder_embedding,
            )

        #  self.f_encoder = NormalEncoder(
            #  config,
            #  self.encoder_embedding,
        #  )

        self.h_encoder = SelfAttentive(
            config,
            self.encoder_embedding,
        )

        #  self.cnn_encoder = CNNEncoder(
            #  config,
            #  self.encoder_embedding
        #  )

        if config.turn_type != 'none' or config.turn_type != 'concat':
            if config.turn_type == 'c_concat':
                self.c_concat_linear = nn.Linear(config.hidden_size * config.turn_num, config.hidden_size)
                init_linear_wt(self.c_concat_linear)
            elif config.turn_type in ['sequential', 'weight', 'transformer', 'self_attn']:
                self.session_encoder = SessionEncoder(config)

        # encoder hidden_state -> decoder hidden_state
        self.reduce_state = ReduceState(config.rnn_type)

        if config.share_embedding:
            decoder_embedding = self.encoder_embedding
        else:
            decoder_embedding = nn.Embedding(
                config.vocab_size,
                config.embedding_size,
                vocab.padid
            )
            init_wt_normal(decoder_embedding.weight, config.embedding_size)

        self.decoder = LuongAttnDecoder(config, decoder_embedding)

    def forward(self,
                h_inputs,
                h_turns_length,
                h_inputs_length,
                h_inputs_position,
                decoder_inputs,
                decoder_inputs_length,
                f_inputs,
                f_inputs_length,
                f_topks_length):
        '''
        input:
            h_inputs: # [turn_num, max_len, batch_size]
            h_inputs_length: [turn_num, batch_size]
            h_turns_length: [batch_size]

            decoder_inputs: [r_max_len, batch_size], first step: [sos * batch_size]

            f_inputs: [f_max_len, batch_size, topk]
            f_inputs_length: [f_max_len, batch_size, topk]
            f_topks_length: [batch_size]
            f_embedded_inputs_length: [batch_size]
        '''

        h_encoder_outputs, h_encoder_hidden_state, h_encoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_position,
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
        decoder_outputs = []
        decoder_output = None
        self.update_teacher_forcing_ratio()
        for i in range(0, self.config.r_max_len):
            if i == 0:
                decoder_input = decoder_inputs[i].view(1, -1)
            else:
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
                if use_teacher_forcing:
                    decoder_input = decoder_inputs[i].view(1, -1)
                else:
                    decoder_input = torch.argmax(
                        decoder_output, dim=2).detach().view(1, -1)

            decoder_output, decoder_hidden_state, _ = self.decoder(decoder_input,
                                                                   decoder_hidden_state,
                                                                   h_encoder_outputs,
                                                                   h_encoder_lengths,
                                                                   f_encoder_outputs,
                                                                   f_encoder_lengths)

            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)
        return decoder_outputs

    def update_teacher_forcing_ratio(self, eplison=0.0001, min_t=0.2):
        self.forward_step += 1
        update_t = self.teacher_forcing_ratio - \
            eplison * (self.forward_step * eplison)
        self.teacher_forcing_ratio = max(update_t, min_t)

    '''evaluate'''

    def evaluate(self,
                 h_inputs,
                 h_turns_length,
                 h_inputs_length,
                 h_inputs_position,
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
        h_encoder_outputs, h_encoder_hidden_state, h_encoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_position,
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
        decoder_outputs = []
        decoder_input = torch.ones((1, self.config.batch_size), dtype=torch.long, device=self.device) * self.vocab.sosid
        for i in range(self.config.r_max_len):
            decoder_output, decoder_hidden_state, _ = self.decoder(decoder_input,
                                                                   decoder_hidden_state,
                                                                   h_encoder_outputs,
                                                                   h_encoder_lengths,
                                                                   f_encoder_outputs,
                                                                   f_encoder_lengths)

            decoder_input = torch.argmax(decoder_output, dim=2).detach()  # [1, batch_size]
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        return decoder_outputs

    '''decode'''

    def decode(self,
               h_inputs,
               h_turns_length,
               h_inputs_length,
               h_inputs_position,
               f_inputs,
               f_inputs_length,
               f_topks_length):

        h_encoder_outputs, h_encoder_hidden_state, h_encoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_position,
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
        input = torch.ones((1, self.config.batch_size), dtype=torch.long, device=self.device) * self.vocab.sosid
        for i in range(self.config.r_max_len):
            output, hidden_state, _ = self.decoder(input,
                                                    hidden_state,
                                                    h_encoder_outputs,
                                                    h_encoder_lengths,
                                                    f_encoder_outputs,
                                                    f_encoder_lengths)

            input = torch.argmax(output, dim=2).detach().view(1, -1)  # [1, batch_size]
            greedy_outputs.append(input)

            if input[0][0].item() == self.vocab.eosid:
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
                           dtype=torch.long, device=self.device) * self.vocab.sosid

        # [num_layers, batch_size x beam_size, hidden_size]
        hidden_state = hidden_state.repeat(1, beam_size, 1)

        if h_encoder_outputs is not None:
            h_encoder_outputs = h_encoder_outputs.repeat(1, beam_size, 1)
            h_encoder_lengths = h_encoder_lengths.repeat(beam_size)

        if f_encoder_outputs is not None:
            f_encoder_outputs = f_encoder_outputs.repeat(1, beam_size, 1)
            f_encoder_lengths = f_encoder_lengths.repeat(beam_size)

        # [batch_size] [0, beam_size * 1, ..., beam_size * (batch_size - 1)]
        batch_position = torch.arange(0, batch_size, dtype=torch.long, device=self.device) * beam_size

        score = torch.ones(batch_size * beam_size, device=self.device) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size, dtype=torch.long, device=self.device) * beam_size, 0.0)

        # Initialize Beam that stores decisions for backtracking
        beam = Beam(
            batch_size,
            self.config.beam_size,
            self.config.r_max_len,
            batch_position,
            self.vocab.eosid
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
            score, top_k_idx = score.view(batch_size, -1).topk(beam_size, dim=1)

            # input: [batch_size x beam_size]
            input = (top_k_idx % self.config.vocab_size).view(-1)

            # beam_idx: [batch_size, beam_size]
            beam_idx = top_k_idx / self.config.vocab_size  # [batch_size, beam_size]

            # top_k_pointer: [batch_size * beam_size]
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            # [num_layers, batch_size * beam_size, hidden_size]
            hidden_state = hidden_state.index_select(1, top_k_pointer)

            # Update sequence scores at beam
            beam.update(score.clone(), top_k_pointer, input)

            # Erase scores for EOS so that they are not expanded
            # [batch_size, beam_size]
            eos_idx = input.data.eq(self.vocab.eosid).view(batch_size, beam_size)

            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length

    def h_forward(self,
                  h_inputs,
                  h_turns_length,
                  h_inputs_length,
                  h_inputs_position):
        """history forward
        Args:
            h_inputs: # [turn_num, max_len, batch_size]
            h_inputs_length: [turn_num, batch_size]
            h_turns_length: [batch_size]
        turn_type:
        """
        if self.config.turn_type == 'concat' or self.config.turn_type == 'none':
            inputs = h_inputs[0, :, :]  # [max_len, batch_size]
            inputs_length = h_inputs_length[0, :]

            # [max_len, batch_size, hidden_size]
            outputs, hidden_state = self.h_encoder(inputs, inputs_length)
            return outputs, hidden_state, inputs_length
        else:
            stack_outputs = list()
            for ti in range(self.config.turn_num):
                inputs = h_inputs[ti, :, :]  # [max_len, batch_size]
                if self.config.turn_type == 'transformer':
                    inputs_position = h_inputs_position[ti, :, :] # [max_len, batch_size]
                    outputs = self.h_encoder(inputs.transpose(0, 1), inputs_position.transpose(0, 1))
                    # [batch_size, max_len, hidden_size]
                    outputs = outputs.transpose(0, 1)
                elif self.config.turn_type == 'self_attn':
                    inputs_length = h_inputs_length[ti, :]  # [batch_size]
                    outputs, hidden_state = self.h_encoder(inputs, inputs_length)
                    outputs = outputs.unsqueeze(0) # [1, batch_size, hidden_size]
                else:
                    inputs_length = h_inputs_length[ti, :]  # [batch_size]
                    outputs, hidden_state = self.h_encoder(inputs, inputs_length)

                stack_outputs.append(outputs[-1].unsqueeze(0))

            if self.config.turn_type == 'sum':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                # [1, batch_size, hidden_size]
                return stack_outputs.sum(dim=0).unsqueeze(0), None, None
            elif self.config.turn_type == 'c_concat':
                # [1, hidden_size * turn_num]
                c_concat_outputs = torch.cat(stack_outputs, dim=2)
                # [1, batch_size, hidden_size]
                return self.c_concat_linear(c_concat_outputs), None, None
            elif self.config.turn_type == 'sequential':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, h_turns_length)  # [1, batch_size, hidden_size]
                return session_outputs[-1].unsqueeze(0), session_hidden_state, h_turns_length
            elif self.config.turn_type == 'weight':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, h_turns_length)  # [1, batch_size, hidden_size]
                return session_outputs, session_hidden_state, h_turns_length
            elif self.config.turn_type == 'transformer':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, h_turns_length)  # [1, batch_size, hidden_size]
                return session_outputs, session_hidden_state, h_turns_length
            elif self.config.turn_type == 'self_attn':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                # session_hidden_state: [num_layers, batch_size, hidden_size]
                session_outputs, session_hidden_state = self.session_encoder(stack_outputs, h_turns_length)
                return session_outputs, session_hidden_state, h_turns_length

    def f_forward(self,
                  f_inputs,
                  f_inputs_length,
                  f_topks_length=None):
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

            outputs, hidden_state = self.f_encoder(f_input, f_input_length)
            outputs = outputs.unsqueeze(0) # [1, batch_size, hidden_size]

            """
            # outputs: [hidden_size, batch_size, max_len]
            _, outputs, _ = self.f_encoder(f_input, f_input_length)
            #  print('outputs: ', outputs.shape)
            outputs = outputs.permute(2, 1, 0)
            """

            output = outputs[-1]
            f_outputs.append(output)

        # [topk, batch_size, hidden_size]
        f_outputs = torch.stack(f_outputs, dim=0)
        return f_outputs

    def f_embedding_forward(self, f_inputs):
        """
        f_inputs: [topk, max_len, batch_size]
        f_inputs_length: [topk, batch_size]
        f_topks_length: [batch_size]
        ignore padding_idx
        """
        f_embedded = list()  # [topk, batch_size, embedding_size]
        for i in range(f_inputs.size(0)):
            batch_embedded = list()
            for j in range(f_inputs.size(2)):
                ids = f_inputs[i, :, j].contiguous()  # [max_len]
                nonzero_count = ids.nonzero().numel()
                embedded = self.encoder_embedding(ids)
                embedded = embedded.sum(dim=0)
                if nonzero_count != 0:
                    embedded = embedded / nonzero_count  # [embedding_size]

                batch_embedded.append(embedded)

            # [batch_size, embedding_size]
            batch_embedded = torch.stack(batch_embedded, dim=0)

            f_embedded.append(batch_embedded)

        # [topk, batch_size, embedding_size]
        f_embedded = torch.stack(f_embedded, dim=0)

        return f_embedded
