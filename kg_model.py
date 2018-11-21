# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn

from modules.simple_encoder import SimpleEncoder
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
                 model_type,
                 vocab_size,
                 c_max_len,
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
                 padid,
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
            padid
        )

        if pre_trained_weight is not None:
            encoder_embedding.weight.data.copy_(pre_trained_weight)
        else:
            init_wt_normal(encoder_embedding.weight,
                           encoder_embedding.embedding_dim)

        # h_encoder
        if turn_type == 'transformer':
            self.transformer_encoder = transformer_models.Encoder(
                c_max_len,
                encoder_embedding,
                num_layers=6,
                num_head=8,
                k_dim=64,
                v_dim=64,
                model_dim=hidden_size,
                inner_dim=1024,
                padid=padid,
                dropout=dropout
            )
        elif turn_type == 'self_attn':
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

        if turn_type != 'none' or turn_type != 'concat':
            if turn_type == 'c_concat':
                self.c_concat_linear = nn.Linear(
                    hidden_size * turn_num, hidden_size)
                init_linear_wt(self.c_concat_linear)
            elif turn_type in ['sequential', 'weight', 'transformer', 'self_attn']:
                self.session_encoder = SessionEncoder(
                    rnn_type,
                    hidden_size,
                    num_layers,
                    bidirectional,
                    dropout,
                )

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

        self.decoder = LuongAttnDecoder(
            vocab_size,
            decoder_embedding,
            rnn_type,
            hidden_size,
            num_layers,
            dropout,
            tied,
            device
        )

    def forward(self,
                h_inputs,
                h_turns_length,
                h_inputs_length,
                h_inputs_position,
                decoder_inputs,
                decoder_inputs_length,
                f_inputs,
                f_inputs_length,
                f_topks_length,
                batch_size,
                r_max_len,
                teacher_forcing_ratio):
        '''
        input:
            h_inputs: # [max_len, batch_size, turn_num]
            h_turns_length: [batch_size]
            h_inputs_length: [batch_size, turn_num]

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
            batch_size
        )

        if h_encoder_hidden_state is None:
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        f_encoder_outputs, f_encoder_hidden_state, f_encoder_lengths = None, None, None
        if self.model_type == 'kg':
            f_encoder_outputs, f_encoder_hidden_state, f_encoder_lengths = self.f_forward(
                f_inputs,
                f_inputs_length,
                f_topks_length,
            )
            decoder_hidden_state += f_encoder_hidden_state

        # decoder
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            decoder_outputs, decoder_hidden_state, attn_weights = self.decoder(decoder_inputs,
                                                                               decoder_hidden_state,
                                                                               decoder_inputs_length,
                                                                               h_encoder_outputs,
                                                                               h_encoder_lengths,
                                                                               f_encoder_outputs,
                                                                               f_encoder_lengths)
            return decoder_outputs

        else:
            decoder_outputs = []
            decoder_input = decoder_inputs[0].view(1, -1)
            for i in range(r_max_len):
                decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                  decoder_hidden_state,
                                                                                  None,
                                                                                  h_encoder_outputs,
                                                                                  h_encoder_lengths,
                                                                                  f_encoder_outputs,
                                                                                  f_encoder_lengths)

                decoder_outputs.append(decoder_output)
                decoder_input = torch.argmax(
                    decoder_output, dim=2).detach().view(1, -1)

            # [r_max_len, batch_size, vocab_size]
            decoder_outputs = torch.cat(decoder_outputs, dim=0)

            return decoder_outputs

    '''evaluate'''

    def evaluate(self,
                 h_inputs,
                 h_turns_length,
                 h_inputs_length,
                 h_inputs_position,
                 decoder_input,
                 f_inputs,
                 f_inputs_length,
                 f_topks_length,
                 r_max_len,
                 batch_size):
        '''
        c_encoder_inputs: [seq_len, batch_size], maybe [r_max_len, 1]
        decoder_input: [1, batch_size], maybe: [sos * 1]
        '''
        h_encoder_outputs, h_encoder_hidden_state, h_encoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_position,
            batch_size
        )

        if h_encoder_hidden_state is None:
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        f_encoder_outputs, f_encoder_hidden_state, f_encoder_lengths = None, None, None
        if self.model_type == 'kg':
            f_encoder_outputs, f_encoder_hidden_state, f_encoder_lengths = self.f_forward(
                f_inputs,
                f_inputs_length,
                f_topks_length,
            )

        # decoder
        decoder_outputs = []
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, _ = self.decoder(decoder_input,
                                                                   decoder_hidden_state,
                                                                   None,
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
               f_topks_length,
               decode_type,
               r_max_len,
               sosid,
               eosid,
               batch_size,
               beam_width,
               best_n):

        h_encoder_outputs, h_encoder_hidden_state, h_encoder_lengths = self.h_forward(
            h_inputs,
            h_turns_length,
            h_inputs_length,
            h_inputs_position,
            batch_size
        )

        if h_encoder_hidden_state is None:
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(
                0).repeat(self.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        f_encoder_outputs, f_encoder_hidden_state, f_encoder_lengths = None, None, None
        if self.model_type == 'kg':
            f_encoder_outputs, f_encoder_hidden_state, f_encoder_lengths = self.f_forward(
                f_inputs,
                f_inputs_length,
                f_topks_length,
            )

        # decoder
        greedy_outputs = self.greedy_decode(decoder_hidden_state,
                                            h_encoder_outputs,
                                            h_encoder_lengths,
                                            f_encoder_outputs,
                                            f_encoder_lengths,
                                            r_max_len,
                                            batch_size,
                                            sosid,
                                            eosid)

        beam_outputs, _, _ = self.beam_decode(
            decoder_hidden_state,
            h_encoder_outputs,
            h_encoder_lengths,
            f_encoder_outputs,
            f_encoder_lengths,
            r_max_len,
            beam_width,
            batch_size,
            sosid,
            eosid
        )

        beam_outputs = beam_outputs.tolist()

        return greedy_outputs, beam_outputs

    def greedy_decode(self,
                      decoder_hidden_state,
                      h_encoder_outputs,
                      h_encoder_lengths,
                      f_encoder_outputs,
                      f_encoder_lengths,
                      r_max_len,
                      batch_size,
                      sosid,
                      eosid):

        greedy_outputs = []
        input = torch.ones((1, batch_size), dtype=torch.long, device=self.device) * sosid
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, attn_weights = self.decoder(input,
                                                                              decoder_hidden_state,
                                                                              None,
                                                                              h_encoder_outputs,
                                                                              h_encoder_lengths,
                                                                              f_encoder_outputs,
                                                                              f_encoder_lengths)

            input = torch.argmax(decoder_output, dim=2).detach()  # [1, batch_size]
            greedy_outputs.append(input)

            if input[0][0].item() == eosid:
                break

        greedy_outputs = torch.cat(greedy_outputs, dim=0)
        # [len, batch_size]  -> [batch_size, len]
        greedy_outputs.transpose_(0, 1)

        return greedy_outputs


    def beam_decode(self,
                    hidden_state=None,
                    h_encoder_outputs=None,
                    h_encoder_lengths=None,
                    f_encoder_outputs=None,
                    f_encoder_lengths=None,
                    r_max_len=35,
                    beam_width=8,
                    batch_size=128,
                    sosid=2,
                    eosid=3):
        '''
        Args:
            hidden_state : [num_layers, batch_size, hidden_size] (optional)
            h_encoder_outputs : [max_len, batch_size, hidden_size]
            h_encoder_lengths : [batch_size] (optional)

        Return:
            prediction: [batch_size, beam, max_len]
        '''
        # [1, batch_size x beam_width]
        input = torch.ones(batch_size * beam_width, dtype=torch.long, device=self.device) * sosid

        # [num_layers, batch_size x beam_width, hidden_size]
        hidden_state = hidden_state.repeat(1, beam_width, 1)

        if h_encoder_outputs is not None:
            h_encoder_outputs = h_encoder_outputs.repeat(1, beam_width, 1)
            h_encoder_lengths = h_encoder_lengths.repeat(beam_width)

        # batch_position [batch_size]
        #   [0, 1 * beam_width, 2 * 2 * beam_width, .., (batch_size-1) * beam_width]
        #   Points where batch_size starts in [batch_size x beam_width] tensors
        #   Ex. position_idx[5]: when 5-th batch_size starts
        batch_position = torch.arange(
            0, batch_size, dtype=torch.long, device=self.device) * beam_width

        # Initialize scores of sequence
        # [batch_size x beam_width]
        # Ex. batch_size: 5, beam_width: 3
        # [0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf]
        score = torch.ones(batch_size * beam_width, device=self.device) * -float('inf')

        score.index_fill_(0, torch.arange(
            0, batch_size, dtype=torch.long, device=self.device) * beam_width, 0.0)

        # Initialize Beam that stores decisions for backtracking
        beam = Beam(
            batch_size,
            self.hidden_size,
            self.vocab_size,
            beam_width,
            r_max_len,
            batch_position,
            eosid
        )

        for i in range(r_max_len):
            # x: [1, batch_size x beam_width]
            # =>
            # output: [1, batch_size x beam_width, vocab_size]
            # h: [num_layers, batch_size x beam_width, hidden_size]
            output, hidden_state, _ = self.decoder(input.view(1, -1).contiguous(),
                                                   hidden_state,
                                                   None,
                                                   h_encoder_outputs,
                                                   h_encoder_lengths,
                                                   f_encoder_outputs,
                                                   f_encoder_lengths)

            # output: [1, batch_size * beam_width, vocab_size]
            # [batch_size * beam_width, vocab_size]
            log_prob = output.squeeze(0)
            # [batch_size * beam_width, vocab_size]
            score = score.view(-1, 1) + log_prob
            #  print('score: ', score.shape)

            # Select `beam size` transitions out of `vocab size` combinations

            # [batch_size x beam_width, vocab_size]
            # => [batch_size, beam_width x vocab_size]
            # Cutoff and retain candidates with top-k scores
            # score: [batch_size, beam_width]
            # top_k_idx: [batch_size, beam_width]
            #       each element of top_k_idx [0 ~ beam x vocab)

            score, top_k_idx = score.view(
                batch_size, -1).topk(beam_width, dim=1)

            # Get token ids with remainder after dividing by top_k_idx
            # Each element is among [0, vocab_size)
            # Ex. Index of token 3 in beam 4
            # (4 * vocab size) + 3 => 3
            # input: [1, batch_size x beam_width]
            input = (top_k_idx % self.vocab_size).view(-1)

            # top-k-pointer [batch_size x beam_width]
            #       Points top-k beam that scored best at current step
            #       Later used as back-pointer at backtracking
            #       Each element is beam index: 0 ~ beam_width
            #                     + position index: 0 ~ beam_width x (batch_size-1)
            beam_idx = top_k_idx / self.vocab_size  # [batch_size, beam_width]
            #  print('beam_idx: ', beam_idx.shape)

            # beam_idx: [batch_size, beam_width], batch_position: [batch_size]
            #  print('beam_idx: ', beam_idx.shape)
            #  print('batch_position: ', batch_position.shape)
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)
            #  print('top_k_pointer: ', top_k_pointer.shape)

            # Select next h (size doesn't change)
            # [num_layers, batch_size * beam_width, hidden_size]
            hidden_state = hidden_state.index_select(1, top_k_pointer)
            #  print('hidden_state: ', hidden_state.shape)

            # Update sequence scores at beam
            beam.update(score.clone(), top_k_pointer, input)

            # Erase scores for EOS so that they are not expanded
            # [batch_size, beam_width]
            eos_idx = input.data.eq(eosid).view(batch_size, beam_width)
            #  print('eos_idx: ', eos_idx.shape)

            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        # prediction ([batch_size, k, r_max_len])
        #     A list of Tensors containing predicted sequence
        # final_score [batch_size, k]
        #     A list containing the final scores for all top-k sequences
        # length [batch_size, k]
        #     A list specifying the length of each sequence in the top-k candidates
        prediction, final_score, length = beam.backtrack()

        return prediction, final_score, length

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
        if self.turn_type == 'concat' or self.turn_type == 'none':
            inputs = h_inputs[:, :, 0]  # [max_len, batch_size]
            inputs_length = h_inputs_length[:, 0]

            # [max_len, batch_size, hidden_size]
            outputs, hidden_state = self.simple_encoder(inputs, inputs_length)
            return outputs, hidden_state, inputs_length
        else:
            stack_outputs = []
            #  stack_hidden_states = []
            for ti in range(self.turn_num):
                inputs = h_inputs[:, :, ti]  # [max_len, batch_size]

                if self.turn_type == 'transformer':
                    inputs_position = h_inputs_position[:, :, ti]
                    outputs = self.transformer_encoder(
                        inputs.transpose(0, 1), inputs_position.transpose(0, 1))
                    #  print('transformer: ', outputs.shape) # [batch_size, max_len, hidden_size]
                    outputs = outputs.transpose(0, 1)
                elif self.turn_type == 'self_attn':
                    inputs_length = h_inputs_length[:, ti]  # [batch_size]
                    outputs, hidden_state = self.self_attn_encoder(
                        inputs, inputs_length)
                    outputs = outputs.unsqueeze(0)
                else:
                    inputs_length = h_inputs_length[:, ti]  # [batch_size]
                    outputs, hidden_state = self.simple_encoder(
                        inputs, inputs_length)

                stack_outputs.append(outputs[-1].unsqueeze(0))

            if self.turn_type == 'sum':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                # [1, batch_size, hidden_size]
                return stack_outputs.sum(dim=0).unsqueeze(0), None, None
            elif self.turn_type == 'c_concat':
                # [1, hidden_size * turn_num]
                c_concat_outputs = torch.cat(stack_outputs, dim=2)
                # [1, batch_size, hidden_size]
                return self.c_concat_linear(c_concat_outputs), None, None
            elif self.turn_type == 'sequential':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, h_turns_length)  # [1, batch_size, hidden_size]
                return session_outputs[-1].unsqueeze(0), session_hidden_state, h_turns_length
            elif self.turn_type == 'weight':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, h_turns_length)  # [1, batch_size, hidden_size]
                return session_outputs, session_hidden_state, h_turns_length
            elif self.turn_type == 'transformer':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                #  print('stack_outputs shape: ', stack_outputs.shape)
                #  print(h_turns_length)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, h_turns_length)  # [1, batch_size, hidden_size]
                #  print(session_outputs.shape)
                #  print(session_hidden_state.shape)
                return session_outputs, session_hidden_state, h_turns_length
            elif self.turn_type == 'self_attn':
                # [turn_num, batch_size, hidden_size]
                stack_outputs = torch.cat(stack_outputs, dim=0)
                session_outputs, session_hidden_state = self.session_encoder(
                    stack_outputs, h_turns_length)  # session_hidden_state: [num_layers, batch_size, hidden_size]
                return session_outputs, session_hidden_state, h_turns_length

    def f_forward(self,
                  f_inputs,
                  f_inputs_length,
                  f_topks_length):
        """
        Args:
            -f_inputs: [topk, max_len, batch_size]
            -f_inputs_length: [topk, batch_size]
            -f_topks_length: [batch_size]
            -hidden_state: [num_layers, batch_size, hidden_size]
        """
        f_outputs = list()
        f_hidden_states = list()
        for i in range(f_inputs.size(0)):
            f_input = f_inputs[i, :, :]  # [max_len, batch_size]
            f_input_length = f_inputs_length[i, :]  # [batch_size]

            #  output: [1, batch_size, hidden_size]
            #  hidden_state: [num_layers * bidirection_num, batch_size, hidden_size // 2]

            """
            output, hidden_state = self.self_attn_encoder(
                f_input,
                f_input_length
            )
            """

            outputs, hidden_state = self.simple_encoder(f_input, f_input_length)
            output = outputs[-1]

            hidden_state = self.reduce_state(hidden_state)
            f_hidden_states.append(hidden_state)
            f_outputs.append(output)

        f_outputs = torch.stack(f_outputs, dim=0)  # [topk, batch_size, hidden_size]
        f_hidden_states = torch.stack(f_hidden_states, dim=0).mean(0) # [num_layes, batch_size, hidden_size]

        return f_outputs, f_hidden_states, f_topks_length
