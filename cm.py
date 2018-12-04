# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn

from modules.normal_cnn import NormalCNN
from modules.normal_encoder import NormalEncoder
from modules.self_attn import SelfAttentive
from modules.session_encoder import SessionEncoder
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
        super(ConversationModel, self).__init__()

        self.config = config
        self.device = device

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

        """
        # context encoder, self Attention
        self.c_encoder = SelfAttentive(
            config,
            enc_embedding
        )

        # fact encoder, CNN
        self.f_encoder = NormalCNN(
            config,
            enc_embedding
        )
        """

        # query encoder, Transformer
        self.q_encoder = transformer.Encoder(
            config,
            enc_embedding
        )

        # response decoder, Transformer
        self.decoder = transformer.Decoder(
            config,
            dec_embedding
        )

        # encoder embedding sharing
        #  self.c_encoder.embedding.weight = self.q_encoder.embedding.weight
        #  self.f_encoder.embedding.weight = self.q_encoder.embedding.weight

        # decoder, encoder embedding sharing
        if config.share_embedding:
            self.decoder.embedding.weight = self.q_encoder.embedding.weight

        self.output_linear = nn.Linear(
            config.transformer_size,
            config.vocab_size,
            bias=False
        )
        nn.init.xavier_normal_(self.output_linear.weight)

        if config.tied and config.transformer_size == config.embedding_size:
            self.output_linear.weight = self.decoder.embedding.weight
            self.x_logit_scale = (config.transformer_size ** -0.5)
        else:
            self.x_logit_scale = 1

        # [batch_size, max_len, vocab_size]
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self,
                h_inputs,
                h_inputs_length,
                h_turns_length,
                h_inputs_pos,
                f_inputs,
                f_inputs_length,
                f_topks_length,
                dec_inputs,
                dec_inputs_length,
                dec_inputs_pos):
        '''
        input:
            h_inputs: # [turn_num, max_len, batch_size]
            h_inputs_length: [turn_num, batch_size]
            h_turns_length: [batch_size]
            h_inputs_pos: [turn_num, max_len, batch_size]

            dec_inputs: [batch_size, r_max_len]
            dec_inputs_length: [batch_size]
            dec_inputs_pos: [batch_size, r_max_len]

            f_inputs: [topk, batch_size, f_max_len]
            f_inputs_length: [batch_size, f_max_len]
            f_topks_length: [batch_size]
        '''
        # c forward
        """
        c_inputs, c_inputs_length, c_turn_length, c_inputs_pos = (
            h_inputs[:-1], h_inputs_length[:-1], h_turns_length - 1, h_inputs_pos[:-1])

        # [batch_size, turn_num-1, hidden_size]
        c_enc_outputs = self.c_forward(
            c_inputs,
            c_inputs_length,
            c_turn_length,
            c_inputs_pos
        )

        # fact encoder [batch_size, f_topk, hidden_size]
        f_enc_outputs = self.f_forward(f_inputs, f_inputs_length)
        """
        c_enc_outputs = None
        f_enc_outputs = None

        # query forward
        q_input, q_input_length, q_input_pos = h_inputs[-1].transpose(0, 1), \
                                                h_inputs_length[-1], \
                                                h_inputs_pos[-1].transpose(0, 1)

        # [batch_size, max_len, transformer_size]
        #  print('q_input: ', q_input)
        #  print('q_input_pos: ', q_input_pos)
        q_enc_outputs = self.q_encoder(q_input, q_input_pos)
        print('q_enc_outputs: ', q_enc_outputs)

        # decoder [batch_size, max_len, transformer_size]
        dec_outputs = self.decoder(
            dec_inputs,
            dec_inputs_pos,
            q_input,
            q_enc_outputs,
            c_enc_outputs,
            f_enc_outputs
        )

        # output_linear
        dec_outputs = self.output_linear(dec_outputs) * self.x_logit_scale

        dec_outputs = dec_outputs.view(-1, self.config.vocab_size)
        print('dec_outputs: ', dec_outputs)

        return dec_outputs

    '''decode'''

    def decode(self,
               h_inputs,
               h_inputs_length,
               h_turns_length,
               h_inputs_pos,
               f_inputs,
               f_inputs_length,
               f_topks_length):
        """
        # c forward
        c_inputs, c_inputs_length, c_turn_length, c_inputs_pos = (
            h_inputs[:-1], h_inputs_length[:-1], h_turns_length - 1, h_inputs_pos[:-1])

        # [batch_size, turn_num-1, hidden_size]
        c_enc_outputs = self.c_forward(
            c_inputs,
            c_inputs_length,
            c_turn_length,
            c_inputs_pos
        )

        # fact encoder [batch_size, f_topk, hidden_size]
        f_enc_outputs = self.f_forward(f_inputs, f_inputs_length)
        """
        c_enc_outputs = None
        f_enc_outputs = None

        # query forward
        q_input, q_input_length, q_input_pos = h_inputs[-1].transpose(
            0, 1), h_inputs_length[-1], h_inputs_pos[-1].transpose(0, 1)

        # [batch_size, max_len, transformer_size]
        q_enc_outputs = self.q_encoder(q_input, q_input_pos)

        # decoder
        beam_outputs, beam_score, beam_length = self.beam_decode(
            q_input,
            q_enc_outputs,
            c_enc_outputs,
            f_enc_outputs
        )

        return beam_outputs, beam_length

    def beam_decode(self,
                    q_input,
                    q_enc_outputs,
                    c_enc_outputs,
                    f_enc_outputs):

        def get_idx_to_tensor_position_map(idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {idx: tensor_pos for tensor_pos, idx in enumerate(idx_list)}

        def collect_active_part(beamed_tensor, curr_active_idx, n_prev_active, beam_size):
            ''' Collect tensor parts associated to active instances. '''
            _, *d_hs = beamed_tensor.size()
            n_curr_active = len(curr_active_idx)
            new_shape = (n_curr_active * beam_size, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)
            return beamed_tensor

        def collate_active_info(q_input,
                                q_enc_outputs,
                                idx_to_position_map,
                                active_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active = len(idx_to_position_map)
            active_idx = [idx_to_position_map[k] for k in active_idx_list]
            active_idx = torch.LongTensor(active_idx).to(self.device)

            active_q_input = collect_active_part(
                q_input, active_idx, n_prev_active, beam_size)
            active_q_enc_outputs = collect_active_part(
                q_enc_outputs, active_idx, n_prev_active, beam_size)
            active_idx_to_pos_map = get_inst_idx_to_tensor_position_map(
                active_idx_list)

            return active_q_input, active_q_enc_outputs, active_idx_to_pos_map

        def beam_decode_step(
                dec_beams, len_dec_seq, q_input, enc_output, idx_to_pos_map, beam_size):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state()
                                   for b in dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active, beam_size):
                dec_partial_pos = torch.arange(
                    1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(
                    0).repeat(n_active * beam_size, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, q_input, enc_output, n_active, beam_size):
                dec_output, * \
                    _ = self.model.decoder(
                        dec_seq, dec_pos, q_input, enc_output)
                # Pick the last step: (bh * bm) * d_h
                dec_output = dec_output[:, -1, :]
                word_prob = F.log_softmax(
                    self.model.tgt_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active, beam_size, -1)

                return word_prob

            def collect_active_idx_list(inst_beams, word_prob, idx_to_pos_map):
                active_idx_list = []
                for idx, pos in idx_to_pos_map.items():
                    is_inst_complete = inst_beams[idx].advance(word_prob[pos])
                    if not is_inst_complete:
                        active_idx_list += [idx]

                return active_idx_list

            n_active = len(idx_to_pos_map)

            dec_seq = prepare_beam_dec_seq(dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active, beam_size)
            word_prob = predict_word(
                dec_seq, dec_pos, q_input, enc_output, n_active, beam_size)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_idx_list = collect_active_idx_list(
                dec_beams, word_prob, idx_to_pos_map)

            return active_idx_list

        def collect_hypothesis_and_scores(dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(dec_beams)):
                scores, tail_idxs = dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [dec_beams[inst_idx].get_hypothesis(
                    i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        # Repeat data for beam search
        beam_size, batch_size = self.config.beam_size, self.config.batch_size
        c_max_len, r_max_len = self.config.c_max_len, self.config.r_max_len

        q_input = q_input.repeat(1, beam_size).view(
            batch_size * beam_size, c_max_len)
        q_enc_outputs = q_enc_outputs.repeat(1, beam_size, 1).view(
            batch_size * beam_size, c_max_len, -1)

        # Prepare beams
        dec_beams = [transformer.Beam(beam_size, device=self.device)
                     for _ in range(batch_size)]

        # Bookkeeping for active or not
        active_idx_list = list(range(batch_size))

        idx_to_position_map = get_idx_to_tensor_position_map(active_idx_list)

        # Decode
        for len_dec_seq in range(1, r_max_len + 1):

            active_idx_list = beam_decode_step(
                dec_beams,
                len_dec_seq,
                q_input,
                q_enc_outputs,
                idx_to_pos_map,
                beam_size)

            if not active_idx_list:
                break  # all instances have finished their path to <EOS>

            q_input, q_enc_outputs, idx_to_pos_map = collate_active_info(
                q_input,
                q_enc_outputs,
                idx_to_pos_map,
                active_idx_list
            )

        batch_hyp, batch_scores = collect_hypothesis_and_scores(
            dec_beams, self.config.n_best)

        return batch_hyp, batch_scores

    def c_forward(self,
                  c_inputs,
                  c_inputs_length,
                  c_turns_length,
                  c_inputs_pos):
        """history forward
        Args:
            c_inputs: # [turn_num, max_len, batch_size]
            c_inputs_length: [turn_num, batch_size]
            c_turns_length: [batch_size]
        Return: [batch_size, turn_num-1, hidden_size]

        """
        stack_outputs = list()
        for i in range(self.config.turn_num - 1):
            input = c_inputs[i, :, :]  # [batch_size, max_len]
            input_length = c_inputs_length[i, :]  # [batch_size]

            # [batch_size, hidden_size]
            output = self.c_encoder(input, input_length)

            stack_outputs.append(output)

        # [batch_size, turn_num-1, hidden_size]
        stack_outputs = torch.stack(stack_outputs, dim=1)
        return stack_outputs

    def f_forward(self,
                  f_inputs,
                  f_inputs_length):
        """
        Args:
            -f_inputs: [topk, batch_sizes, max_len]
            -f_inputs_length: [topk, batch_size]
        Return: [batch_size, f_topk, hidden_size]
        """
        f_outputs = list()
        for i in range(self.config.f_topk):
            f_input = f_inputs[i, :, :]  # [batch_size, max_len]
            f_input_length = f_inputs_length[i, :]  # [batch_size]

            # [batch_size, 1, max_len]
            output, _ = self.f_encoder(f_input, f_input_length)

            f_outputs.append(output)

        # [batch_size, topk, hidden_size]
        f_outputs = torch.cat(f_outputs, dim=1)
        return f_outputs
