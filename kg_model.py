# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

#  from modules.normal_cnn import NormalCNN
from modules.normal_encoder import NormalEncoder
#  from modules.self_attn import SelfAttentive
from modules.session_encoder import SessionEncoder
from modules.reduce_state import ReduceState
from modules.luong_attn_decoder import LuongAttnDecoder
from modules.beam import Beam
#  import modules.transformer as transformer
#  import modules.tf as tf
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

        # c, q encoder
        self.encoder = NormalEncoder(
            config,
            enc_embedding
        )

        self.f_encoder = None
        if config.model_type == 'kg':
            """
            self.f_encoder = transformer.Encoder(
                config,
                enc_embedding,
                has_position=False
            )
            """
            """
            self.f_encoder = tf.Models.Encoder(
                n_src_vocab=config.vocab_size,
                len_max_seq=config.f_topk,
                d_word_vec=config.embedding_size,
                n_layers=config.t_num_layers,
                n_head=config.num_heads,
                d_k=config.k_size,
                d_v=config.v_size,
                d_model=config.transformer_size,
                d_inner=config.inner_hidden_size,
                dropout=config.dropout
            )
            """
            self.f_encoder = enc_embedding

        # session encoder
        if config.enc_type.count('_h') != 0:
            self.session_encoder = SessionEncoder(config)

        if config.enc_type.count('concat') != 0:
            self.concat_linear = nn.Linear(
                (config.turn_num) * config.hidden_size,
                config.hidden_size
            )
            init_linear_wt(self.concat_linear)

        self.reduce_state = ReduceState(config.rnn_type)

        # decoder
        self.decoder = LuongAttnDecoder(config, dec_embedding)

        if self.f_encoder is not None:
            #  self.f_encoder.embedding.weight = self.encoder.embedding.weight
            #  self.f_encoder.src_word_emb.weight = self.encoder.embedding.weight
            self.f_encoder.weight = self.encoder.embedding.weight

        # encoder, decode embedding share
        if config.share_embedding:
            self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(self,
                enc_inputs,
                enc_inputs_length,
                enc_turn_length,
                dec_inputs,
                f_inputs,
                f_inputs_length,
                f_topk_length):
        '''
        Args:
            enc_inputs: [max_len, batch_size] or [turn_num, max_len, batch_size]
            enc_inputs_length: [batch_size] or [turn_num, batch_size]
            enc_turn_length: [] or [batch_size]

            dec_inputs: [max_len, batch_size], first step: [sos * batch_size]

            f_inputs: [batch_size, f_topk]
            f_inputs_length: [batch_size]
            f_topk_length: None
        '''
        enc_type = self.config.enc_type
        if enc_type == 'q' or enc_type == 'qc':
            # [max_len, batch_size]
            enc_outputs, enc_hidden = self.encoder(
                enc_inputs,
                lengths=enc_inputs_length,
                sort=True
            )

            dec_hidden = self.reduce_state(enc_hidden)
            enc_length = enc_inputs_length
        else:  # []
            # [turn_num, batch_size, hidden_size]
            enc_outputs, enc_hidden = self.utterance_forward(
                enc_inputs,
                enc_inputs_length
            )
            enc_length = enc_turn_length

            if enc_type.count('_h') != 0:  # hierarchical
                # [turn_num, batch_size, hidden_size]
                enc_outputs, enc_hidden = self.inter_utterance_forward(
                    enc_outputs,
                    enc_turn_length
                )

            # [qc_concat, qc_sum, qc_concat_h, qc_sum_h]
            if enc_type.count('sum') != 0:
                # [1, batch_size, hidden_size]
                dec_input = enc_outputs.sum(dim=0).unsqueeze(0)
                dec_hidden = dec_input.repeat(
                    self.config.decoder_num_layers, 1, 1)
            elif enc_type.count('concat') != 0:
                dec_input = enc_outputs.transpose(
                    0, 1).view(self.config.turn_num, -1)
                dec_input = self.concat_linear(dec_input)
                dec_hidden = dec_input.repeat(
                    self.config.decoder_num_layers, 1, 1)
            else:
                # [qc_seq, qc_seq_h]
                dec_hidden = self.reduce_state(enc_hidden)

        # [q_attn, qc_attn, qc_seq_attn, qc_seq_h_attn]
        if enc_type not in ['q', 'qc']:
            if enc_type.count('attn') == 0:
                enc_outputs = None

        # fact encoder
        f_enc_outputs = None
        if self.config.model_type == 'kg':
            f_enc_outputs = self.f_forward(
                f_inputs,
                f_inputs_length,
                f_topk_length
            )

        # decoder
        dec_outputs, _, _ = self.decoder(
            dec_inputs,
            dec_hidden,
            enc_outputs,
            enc_length,
            f_enc_outputs=f_enc_outputs,
            f_enc_length=f_inputs_length
        )

        # [max_len * batch_, vocab_size]
        dec_outputs = dec_outputs.view(-1, dec_outputs.size(-1)).contiguous()
        #  print('dec_outputs: ', dec_outputs)

        return dec_outputs

    '''decode'''

    def decode(self,
               enc_inputs,
               enc_inputs_length,
               enc_turn_length,
               f_inputs,
               f_inputs_length,
               f_topk_length):

        enc_type = self.config.enc_type
        if enc_type == 'q' or \
                enc_type == 'qc':
            # [max_len, batch_size]
            enc_outputs, enc_hidden = self.encoder(
                enc_inputs,
                enc_inputs_length
            )

            dec_hidden = self.reduce_state(enc_hidden)
            enc_length = enc_inputs_length
        else:  # []
            # [turn_num, batch_size, hidden_size]
            enc_outputs, enc_hidden = self.utterance_forward(
                enc_inputs,
                enc_inputs_length
            )
            enc_length = enc_turn_length

            if enc_type.count('_h') != 0:  # hierarchical
                # [turn_num, batch_size, hidden_size]
                enc_outputs, enc_hidden = self.inter_utterance_forward(
                    enc_outputs,
                    enc_turn_length
                )

            # [qc_concat, qc_sum, qc_concat_h, qc_sum_h]
            if enc_type.count('sum') != 0:
                # [1, batch_size, hidden_size]
                dec_input = enc_outputs.sum(dim=0).unsqueeze(0)
                dec_hidden = dec_input.repeat(
                    self.config.decoder_num_layers, 1, 1)
            elif enc_type.count('concat') != 0:
                dec_input = enc_outputs.transpose(
                    0, 1).view(self.config.turn_num, -1)
                dec_input = self.concat_linear(dec_input)
                dec_hidden = dec_input.repeat(
                    self.config.decoder_num_layers, 1, 1)
            else:
                # [qc_seq, qc_seq_h]
                dec_hidden = self.reduce_state(enc_hidden)

        # [q_attn, qc_attn, qc_seq_attn, qc_seq_h_attn]
        if enc_type not in ['q', 'qc']:
            if enc_type.count('attn') == 0:
                enc_outputs = None

        # fact encoder
        f_enc_outputs = None
        if self.config.model_type == 'kg':
            f_enc_outputs = self.f_forward(
                f_inputs,
                f_inputs_length,
                f_topk_length
            )

        # decoder
        beam_outputs, beam_score, beam_length = self.beam_decode(
            dec_hidden,
            enc_outputs,
            enc_length,
            f_enc_outputs,
            f_inputs_length
        )

        greedy_outputs = self.greedy_decode(
            dec_hidden,
            enc_outputs,
            enc_length,
            f_enc_outputs,
            f_inputs_length
        )

        return greedy_outputs, beam_outputs, beam_length

    def greedy_decode(self,
                      dec_hidden,
                      enc_outputs,
                      enc_length,
                      f_enc_outputs,
                      f_enc_length):

        greedy_outputs = []
        dec_input = torch.ones((1, self.config.batch_size),
                               dtype=torch.long, device=self.device) * SOS_ID

        for i in range(self.config.r_max_len):
            output, dec_hidden,  _ = self.decoder(
                dec_input,
                dec_hidden,
                enc_outputs,
                enc_length,
                f_enc_outputs=f_enc_outputs,
                f_enc_length=f_enc_length
            )
            output = F.log_softmax(output, dim=2)
            dec_input = torch.argmax(output, dim=2).detach().view(1, -1)  # [1, batch_size]
            greedy_outputs.append(dec_input)

        # [len, batch_size]  -> [batch_size, len]
        greedy_outputs = torch.cat(greedy_outputs, dim=0).transpose(0, 1)

        return greedy_outputs

    def beam_decode(self,
                    dec_hidden,
                    enc_outputs,
                    enc_length,
                    f_enc_outputs,
                    f_enc_length):
        '''
        Args:
            dec_hidden : [num_layers, batch_size, hidden_size] (optional)
            enc_outputs : [max_len, batch_size, hidden_size]
            enc_length : [batch_size] (optional)

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
        if enc_outputs is not None:
            enc_outputs = enc_outputs.repeat(1, beam_size, 1)
            enc_length = enc_length.repeat(beam_size)

        if f_enc_outputs is not None:
            f_enc_outputs = f_enc_outputs.repeat(1, beam_size, 1)
            f_enc_length = f_enc_length.repeat(beam_size)

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
                enc_outputs,
                enc_length,
                f_enc_outputs=f_enc_outputs,
                f_enc_length=f_enc_length
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

    def utterance_forward(self,
                          enc_inputs,
                          enc_inputs_length):
        """
        Args:
            enc_inputs: [turn_num, max_len, batch_size]
            enc_inputs: [turn_num, batch_size]

        """
        utterance_outputs = list()

        hidden_state = None
        for ti in range(self.config.turn_num):
            inputs = enc_inputs[ti, :, :]  # [max_len, batch_size]
            inputs_length = enc_inputs_length[ti, :]  # [batch_size]

            outputs, hidden_state = self.encoder(
                inputs,
                lengths=inputs_length,
                hidden_state=hidden_state,
                sort=False
            )

            utterance_outputs.append(outputs[-1])

        # [turn_num, batch_size, hidden_size]
        utterance_outputs = torch.stack(utterance_outputs, dim=0)

        return utterance_outputs, hidden_state

    def inter_utterance_forward(self,
                                utterance_outputs,
                                enc_turn_length):
        # [turn_num, batch_size, hidden_size]
        outputs, hidden_state = self.session_encoder(
            utterance_outputs, enc_turn_length)

        return outputs, hidden_state

    def f_forward(self,
                  f_inputs,
                  f_inputs_length,
                  f_topk_length):
        """
        Args:
            -f_inputs: [topk, batch_size, max_len] or [batch_size, f_topk]
            -f_inputs_length: [topk, batch_size] or [batch_size]
            -f_topk_length: [batch_size]
        """
        #  print('f_inputs: ', f_inputs)

        # [batch_size, max_len, hidden_size]
        #  f_enc_outputs = self.f_encoder(f_inputs, f_inputs_length)

        # [batch_size, f_topk, embedding_size]
        f_enc_outputs = self.f_encoder(f_inputs)

        # [max_len, batch_size, hidden_size]
        f_enc_outputs = f_enc_outputs.transpose(0, 1)

        #  print('f_enc_outputs: ', f_enc_outputs.shape)

        return f_enc_outputs
