# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.simple_encoder import SimpleEncoder
from modules.self_attn import SelfAttentive
from modules.session_encoder import SessionEncoder
from modules.decoder import Decoder
from modules.reduce_state import ReduceState
from modules.bahdanau_attn_decoder import BahdanauAttnDecoder
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.utils import init_linear_wt, init_wt_normal
from modules.utils import sequence_mask
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
            init_wt_normal(encoder_embedding.weight, encoder_embedding.embedding_dim)

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
            self.self_attn_encoder =  SelfAttentive(
                encoder_embedding,
                rnn_type,
                num_layers,
                bidirectional,
                hidden_size,
                dropout=dropout
            )
        else:
            self.simple_encoder = SimpleEncoder(vocab_size,
                                                encoder_embedding,
                                                rnn_type,
                                                hidden_size,
                                                encoder_num_layers,
                                                bidirectional,
                                                dropout)

        if turn_type != 'none' or turn_type != 'concat':
            if turn_type == 'c_concat':
                self.c_concat_linear = nn.Linear(hidden_size * turn_num, hidden_size)
                init_linear_wt(self.c_concat_linear)
            elif turn_type in ['sequential', 'weight', 'transformer', 'self_attn']:
                self.session_encoder = SessionEncoder(
                    rnn_type,
                    hidden_size,
                    num_layers,
                    bidirectional,
                    dropout,
                )
            else:
                pass

        # fact encoder
        if model_type == 'kg':
            if pre_embedding_size != hidden_size:
                self.f_embedded_linear = nn.Linear(pre_embedding_size, hidden_size)
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
                init_wt_normal(decoder_embedding.weight, decoder_embedding.embedding_dim)

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
                decoder_inputs_length,
                f_embedded_inputs,
                f_embedded_inputs_length,
                f_ids_inputs,
                f_ids_inputs_length,
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

            f_ids_inputs: [f_max_len, batch_size, topk]
            f_ids_inputs_length: [f_max_len, batch_size, topk]
            f_topks_length: [batch_size]
            f_embedded_inputs: [batch_size, r_max_len, topk]
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
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_embedded_inputs,
                                                  f_embedded_inputs_length,
                                                  f_ids_inputs,
                                                  f_ids_inputs_length,
                                                  f_topks_length,
                                                  decoder_hidden_state,
                                                  batch_size)

        # decoder
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_outputs, decoder_hidden_state, attn_weights = self.decoder(decoder_inputs,
                                                                               decoder_hidden_state,
                                                                               h_encoder_outputs,
                                                                               h_encoder_lengths,
                                                                               decoder_inputs_length)
            return decoder_outputs

        else:
            decoder_outputs = []
            decoder_input = decoder_inputs[0].view(1, -1)
            for i in range(r_max_len):
                decoder_output, decoder_hidden_state, attn_weights = self.decoder(decoder_input,
                                                                                  decoder_hidden_state,
                                                                                  None,
                                                                                  h_encoder_outputs,
                                                                                  h_encoder_lengths)

                decoder_outputs.append(decoder_output)
                decoder_input = torch.argmax(decoder_output, dim=2).detach().view(1, -1)

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
                 f_embedded_inputs,
                 f_embedded_inputs_length,
                 f_ids_inputs,
                 f_ids_inputs_length,
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
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_embedded_inputs,
                                                  f_embedded_inputs_length,
                                                  f_ids_inputs,
                                                  f_ids_inputs_length,
                                                  f_topks_length,
                                                  decoder_hidden_state,
                                                  batch_size)

        # decoder
        decoder_outputs = []
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, _ = self.decoder(decoder_input,
                                                                   decoder_hidden_state,
                                                                   h_encoder_outputs,
                                                                   h_encoder_lengths)

            decoder_input = torch.argmax(
                decoder_output, dim=2).detach()  # [1, batch_size]
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        return decoder_outputs

    '''decode'''

    def decode(self,
               h_inputs,
               h_turns_length,
               h_inputs_length,
               h_inputs_position,
               f_embedded_inputs,
               f_embedded_inputs_length,
               f_ids_inputs,
               f_ids_inputs_length,
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
            decoder_hidden_state = h_encoder_outputs[-1].unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
        else:
            decoder_hidden_state = self.reduce_state(h_encoder_hidden_state)

        # fact encoder
        if self.model_type == 'kg':
            decoder_hidden_state = self.f_forward(f_embedded_inputs,
                                                  f_embedded_inputs_length,
                                                  f_ids_inputs,
                                                  f_ids_inputs_length,
                                                  f_topks_length,
                                                  decoder_hidden_state,
                                                  batch_size)
        # decoder
        greedy_outputs = None
        beam_outputs = None
        #  if decode_type == 'greedy':
        greedy_outputs = []
        input = torch.ones((1, batch_size), dtype=torch.long, device=self.device) * sosid
        for i in range(r_max_len):
            decoder_output, decoder_hidden_state, attn_weights = self.decoder(input,
                                                                              decoder_hidden_state,
                                                                              h_encoder_outputs,
                                                                              h_encoder_lengths)

            input = torch.argmax(decoder_output, dim=2).detach()  # [1, batch_size]
            greedy_outputs.append(input)

            if input[0][0].item() == eosid:
                break

        greedy_outputs = torch.cat(greedy_outputs, dim=0)
        # [len, batch_size]  -> [batch_size, len]
        greedy_outputs.transpose_(0, 1)

        # prediction ([batch_size, topk, r_max_len])
        beam_outputs, _, _ = self.beam_decode(
            decoder_hidden_state,
            h_encoder_outputs,
            h_encoder_lengths,
            r_max_len,
            beam_width,
            batch_size,
            sosid,
            eosid
        )
        beam_outputs = beam_outputs.tolist()

        return greedy_outputs, beam_outputs

    def beam_decode(self,
                    hidden_state=None,
                    h_encoder_outputs=None,
                    h_encoder_lengths=None,
                    r_max_len=35,
                    beam_width=8,
                    batch_size=128,
                    sosid=2,
                    eosid=3):
        """
        Args:
            hidden_state : [num_layers, batch_size, hidden_size] (optional)
            h_encoder_outputs : [max_len, batch_size, hidden_size]
            h_encoder_lengths : [batch_size] (optional)

        Return:
            prediction: [batch_size, beam, max_len]
        """
        #  print('hidden_state: ', hidden_state.shape)
        # [1, batch_size x beam_width]
        input = torch.ones(batch_size * beam_width, dtype=torch.long, device=self.device) * sosid
        #  print("input: ", input.shape)

        # [num_layers, batch_size x beam_width, hidden_size]
        hidden_state = hidden_state.repeat(1, beam_width, 1)
        #  print('hidden_state: ', hidden_state.shape)

        if h_encoder_outputs is not None:
            h_encoder_outputs = h_encoder_outputs.repeat(1, beam_width, 1)
            h_encoder_lengths = h_encoder_lengths.repeat(beam_width)
            #  print('h_encoder_outputs: ', h_encoder_outputs.shape)
            #  print(h_encoder_lengths)

        # batch_position [batch_size]
        #   [0, 1 * beam_width, 2 * 2 * beam_width, .., (batch_size-1) * beam_width]
        #   Points where batch_size starts in [batch_size x beam_width] tensors
        #   Ex. position_idx[5]: when 5-th batch_size starts
        batch_position = torch.arange(0, batch_size, dtype=torch.long, device=self.device) * beam_width

        # Initialize scores of sequence
        # [batch_size x beam_width]
        # Ex. batch_size: 5, beam_width: 3
        # [0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf]
        score = torch.ones(batch_size * beam_width, device=self.device) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size, dtype=torch.long, device=self.device) * beam_width, 0.0)

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
                                                   h_encoder_outputs,
                                                   h_encoder_lengths)

            # output: [1, batch_size * beam_width, vocab_size]
            log_prob = output.squeeze(0) # [batch_size * beam_width, vocab_size]
            score = score.view(-1, 1) + log_prob # [batch_size * beam_width, vocab_size]
            #  print('score: ', score.shape)

            # Select `beam size` transitions out of `vocab size` combinations

            # [batch_size x beam_width, vocab_size]
            # => [batch_size, beam_width x vocab_size]
            # Cutoff and retain candidates with top-k scores
            # score: [batch_size, beam_width]
            # top_k_idx: [batch_size, beam_width]
            #       each element of top_k_idx [0 ~ beam x vocab)

            score, top_k_idx = score.view(batch_size, -1).topk(beam_width, dim=1)

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
            top_k_pointer = (beam_idx + batch_position.view(1, -1)).view(-1)
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
            inputs = h_inputs[:, :, 0] # [max_len, batch_size]
            inputs_length = h_inputs_length[:, 0]

            # [max_len, batch_size, hidden_size]
            outputs, hidden_state = self.simple_encoder(inputs, inputs_length)
            return outputs, hidden_state, inputs_length
        else:
            stack_outputs = []
            #  stack_hidden_states = []
            for ti in range(self.turn_num):
                inputs = h_inputs[:, :, ti] # [max_len, batch_size]

                if self.turn_type == 'transformer':
                    inputs_position = h_inputs_position[:, :, ti]
                    outputs = self.transformer_encoder(inputs.transpose(0, 1), inputs_position.transpose(0, 1))
                    #  print('transformer: ', outputs.shape) # [batch_size, max_len, hidden_size]
                    outputs = outputs.transpose(0, 1)
                elif self.turn_type == 'self_attn':
                    inputs_length = h_inputs_length[:, ti] # [batch_size]
                    outputs, hidden_state = self.self_attn_encoder(inputs, inputs_length)
                else:
                    inputs_length = h_inputs_length[:, ti] # [batch_size]
                    outputs, hidden_state = self.simple_encoder(inputs, inputs_length)

                stack_outputs.append(outputs[-1].unsqueeze(0))

            if self.turn_type == 'sum':
                stack_outputs = torch.cat(stack_outputs, dim=0) # [turn_num, batch_size, hidden_size]
                return stack_outputs.sum(dim=0).unsqueeze(0), None, None # [1, batch_size, hidden_size]
            elif self.turn_type == 'c_concat':
                c_concat_outputs = torch.cat(stack_outputs, dim=2) # [1, hidden_size * turn_num]
                return self.c_concat_linear(c_concat_outputs), None, None # [1, batch_size, hidden_size]
            elif self.turn_type == 'sequential':
                stack_outputs = torch.cat(stack_outputs, dim=0) # [turn_num, batch_size, hidden_size]
                session_outputs, session_hidden_state = self.session_encoder(stack_outputs, h_turns_length) # [1, batch_size, hidden_size]
                return session_outputs[-1].unsqueeze(0), session_hidden_state, h_turns_length
            elif self.turn_type == 'weight':
                stack_outputs = torch.cat(stack_outputs, dim=0) # [turn_num, batch_size, hidden_size]
                session_outputs, session_hidden_state = self.session_encoder(stack_outputs, h_turns_length) # [1, batch_size, hidden_size]
                return session_outputs, session_hidden_state, h_turns_length
            elif self.turn_type == 'transformer':
                stack_outputs = torch.cat(stack_outputs, dim=0) # [turn_num, batch_size, hidden_size]
                #  print('stack_outputs shape: ', stack_outputs.shape)
                #  print(h_turns_length)
                session_outputs, session_hidden_state = self.session_encoder(stack_outputs, h_turns_length) # [1, batch_size, hidden_size]
                #  print(session_outputs.shape)
                #  print(session_hidden_state.shape)
                return session_outputs, session_hidden_state, h_turns_length
            elif self.turn_type == 'self_attn':
                stack_outputs = torch.cat(stack_outputs, dim=0) # [turn_num, batch_size, hidden_size]
                session_outputs, session_hidden_state = self.session_encoder(stack_outputs, h_turns_length) # session_hidden_state: [1, batch_size, hidden_size]
                return session_outputs, session_hidden_state, h_turns_length

    def f_forward(self,
                  f_embedded_inputs,
                  f_embedded_inputs_length,
                  f_ids_inputs,
                  f_ids_inputs_length,
                  f_topks_length,
                  decoder_hidden_state,
                  batch_size):
        """
        Args:
            - f_embedded_inputs: [batch_size, topk, embedding_size]
            - hidden_state: [num_layers, batch_size, hidden_size]
            -f_ids_inputs: [max_len, batch_size, topk]
            -f_ids_inputs_length: [batch_size, topk]
            -hidden_state: [num_layers, batch_size, hidden_size]
        """

        f_outputs = list()
        for bi in range(batch_size):
            f_ids_input = f_ids_inputs[:, bi, :] # [f_max_len, topk]
            f_ids_input_length = f_ids_inputs_length[bi, :] # [topk]

            outputs, hidden_state = self.self_attn_encoder(
                f_ids_input,
                f_ids_input_length
            ) # [1, topk, hidden_size]

            outputs = outputs.transpose(0, 1) # [topk, 1, hidden_size]

            f_topk_length = f_topks_length[bi].view(1)
            session_outputs, session_hidden_state = self.session_encoder(
                outputs,
                f_topk_length
            ) # [topk, 1, hidden_size]

            f_outputs.append(session_outputs.squeeze(1))

        f_outputs = torch.stack(f_outputs, dim=0) # [batch_size, topk, hidden_size]

        # [batch_size, topk, embedding_size] -> [batch_size, topk, hidden_size]
        #  if self.pre_embedding_size != self.hidden_size:
            #  f_embedded_inputs = self.f_embedded_linear(f_embedded_inputs)

        # M [batch_size, topk, hidden_size]
        fM = self.fact_linearA(f_outputs)

        # C [batch_size, topk, hidden_size]
        fC = self.fact_linearC(f_outputs)

        # [batch_size, num_layers, topk]
        tmpP = torch.bmm(decoder_hidden_state.transpose(0, 1), fM.transpose(1, 2))

        mask = sequence_mask(f_topks_length, max_len=tmpP.size(-1))
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        tmpP.masked_fill_(1 - mask, -float('inf'))

        P = F.softmax(tmpP, dim=2)

        o = torch.bmm(P, fC)  # [batch_size, num_layers, hidden_size]

        u_ = torch.add(o, decoder_hidden_state.transpose(0, 1))

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
