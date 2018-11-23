#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Fork From:
https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/blob/master/model/layers/beam_search.py
"""

import torch


class Beam(object):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 vocab_size,
                 beam_width,
                 max_len,
                 batch_position,
                 eosid=3):

        """Beam class for beam search"""
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.max_len = max_len
        self.eosid = eosid

        # batch_position [batch_size]
        #   [0, beam_width, beam_width * 2, .., beam_width * (batch_size-1)]
        #   Points where batch_size starts in [batch_size x beam_width] tensors
        #   Ex. position_idx[5]: when 5-th batch_size starts
        self.batch_position = batch_position

        self.log_probs = list()  # [(batch_size*k, vocab_size)] * sequence_length

        self.scores = list()  # [(batch_size*k)] * sequence_length
        self.back_pointers = list()  # [(batch_size*k)] * sequence_length
        self.token_ids = list()  # [(batch_size*k)] * sequence_length

        self.metadata = {
            'inputs': None,
            'output': None,
            'scores': None,
            'length': None,
            'sequence': None,
        }

    def update(self, score, back_pointer, token_id):
        """
        Append intermediate top-k candidates to beam at each step
            score: [batch_size, beam_width]
            back_pointer: [batch_size * beam_width]
            token_id: [batch_size * beam_width]
        """
        self.scores.append(score)
        self.back_pointers.append(back_pointer)
        self.token_ids.append(token_id)

    def backtrack(self):
        """Backtracks over batch_size to generate optimal k-sequences
        Returns:
            prediction ([batch_size, k, max_len])
                A list of Tensors containing predicted sequence
            final_score [batch_size, k]
                A list containing the final scores for all top-k sequences
            length [batch_size, k]
                A list specifying the length of each sequence in the top-k candidates
        """
        prediction = list()

        # Initialize for length of top-k sequences
        length = [[self.max_len] * self.beam_width for _ in range(self.batch_size)]

        # Last step output of the beam are not sorted => sort here!
        # Size not changed [batch_size size, beam_width]
        top_k_score, top_k_idx = self.scores[-1].topk(self.beam_width, dim=1)

        # Initialize sequence scores
        top_k_score = top_k_score.clone()
        #  print("top_k_score: ", top_k_score.shape)

        n_eos_in_batch = [0] * self.batch_size

        # Initialize Back-pointer from the last step
        # Add self.position_idx for indexing variable with batch_size x beam as the first dimension
        # [batch_size x beam]
        back_pointer = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)

        for t in reversed(range(self.max_len)):
            # Reorder variables with the Back-pointer
            # [batch_size x beam]
            token_id = self.token_ids[t].index_select(0, back_pointer)

            # Reorder the Back-pointer
            # [batch_size x beam]
            back_pointer = self.back_pointers[t].index_select(0, back_pointer)

            # Indices of ended sequences
            # [< batch_size x beam]

            eos_indices = self.token_ids[t].eq(self.eosid).nonzero()

            # For each batch_size, every time we see an EOS in the backtracking process,
            # If not all sequences are ended
            #    lowest scored survived sequence <- detected ended sequence
            # if all sequences are ended
            #    lowest scored ended sequence <- detected ended sequence
            if eos_indices.dim() > 0:
                # Loop over all eosid at current step
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    # absolute index of detected ended sequence
                    eos_idx = eos_indices[i, 0].item()

                    # At which batch_size EOS is located
                    batch_idx = eos_idx // self.beam_width
                    batch_start_idx = batch_idx * self.beam_width

                    # if n_eos_in_batch[batch_idx] > self.beam_width:

                    # Index of sequence with lowest score
                    _n_eos_in_batch = n_eos_in_batch[batch_idx] % self.beam_width
                    beam_idx_to_be_replaced = self.beam_width - _n_eos_in_batch - 1
                    idx_to_be_replaced = batch_start_idx + beam_idx_to_be_replaced

                    # Replace old information with new sequence information
                    back_pointer[idx_to_be_replaced] = self.back_pointers[t][eos_idx].item()
                    token_id[idx_to_be_replaced] = self.token_ids[t][eos_idx].item()
                    top_k_score[batch_idx,
                                beam_idx_to_be_replaced] = self.scores[t].view(-1)[eos_idx].item()
                    length[batch_idx][beam_idx_to_be_replaced] = t + 1

                    n_eos_in_batch[batch_idx] += 1

            # max_len * [batch_size x beam]
            prediction.append(token_id)

        # Sort and re-order again as the added ended sequences may change the order
        # [batch_size, beam]
        top_k_score, top_k_idx = top_k_score.topk(self.beam_width, dim=1)
        final_score = top_k_score.data

        for batch_idx in range(self.batch_size):
            length[batch_idx] = [length[batch_idx][beam_idx.item()]
                                 for beam_idx in top_k_idx[batch_idx]]

        # [batch_size x beam]
        top_k_idx = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in the reverse order
        # [batch_size, beam]

        prediction = [step.index_select(0, top_k_idx).view(
            self.batch_size, self.beam_width) for step in reversed(prediction)]

        # [batch_size, beam, max_len]
        prediction = torch.stack(prediction, 2)

        return prediction, final_score, length

