#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Tok Decode, fork
https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py
"""
import torch
import torch.nn.functional as F
from misc.vocab import SOS_ID, EOS_ID


class TopKDecoder(torch.nn.Module):
    r"""
    Top-beam_size decoding with beam search.
    Args:
        decoder_rnn (DecoderRNN): An object of DecoderRNN used for decoding.
        beam_size (int): Size of the beam.
    Inputs: inputs, encoder_hidden, q_enc_outputs, function, teacher_forcing_ratio
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default is `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features
          in the dec_hidden state `h` of encoder. Used as the initial dec_hidden state of the decoder.
        - **q_enc_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from decoder dec_hidden state
          (default is `torch.nn.functional.log_softmax`).
    Outputs: dec_outputs, dec_hidden, ret_dict
        - **dec_outputs** (batch): batch-length list of tensors with size (max_len, hidden_size) containing the
          outputs of the decoder.
        - **dec_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last dec_hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*length* : list of integers
          representing lengths of output sequences, *topk_length*: list of integers representing lengths of beam search
          sequences, *sequence* : list of sequences, where each sequence is a list of predicted token IDs,
          *topk_sequence* : list of beam search sequences, each beam is a list of token IDs, *inputs* : target
          outputs if provided for decoding}.
    """

    def __init__(self,
                 config,
                 decoder,
                 device):
        super(TopKDecoder, self).__init__()
        self.config = config
        self.decoder = decoder
        self.device = device

    def forward(self,
                dec_hidden=None,
                q_enc_outputs=None,
                q_inputs_length=None,
                c_enc_outputs=None,
                c_turn_length=None,
                f_enc_outputs=None,
                f_topk_length=None):

        batch_size, beam_size = self.config.batch_size, self.config.beam_size
        vocab_size = self.config.vocab_size
        max_len = self.config.r_max_len

        # [batch_size * beam_size, 1]
        self.pos_index = (torch.LongTensor(range(batch_size))
                          * beam_size).view(-1, 1).to(self.device)

        # Inflate the initial dec_hidden states to be of size: batch_size*beam_size x h
        # ... same way for q_enc_outputs and dec_outputs
        dec_hidden = dec_hidden.repeat(1, beam_size, 1)
        # [max_len, batch_size * beam_size, hidden_size]
        q_enc_outputs = q_enc_outputs.repeat(1, beam_size, 1)
        q_inputs_length = q_inputs_length.repeat(beam_size)

        if c_enc_outputs is not None:
            c_enc_outputs = c_enc_outputs.repeat(1, beam_size, 1)
            c_turn_length = c_turn_length.repeat(beam_size)

        if f_enc_outputs is not None:
            f_enc_outputs = f_enc_outputs.repeat(1, beam_size, 1)
            f_topk_length = f_topk_length.repeat(beam_size)

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top beam_size
        sequence_scores = torch.Tensor(batch_size * beam_size, 1).to(self.device)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor(
            [i * beam_size for i in range(0, batch_size)]).to(self.device), 0.0)

        # Initialize the dec_input vector
        # [batch_size * beam_size, 1]
        dec_input = torch.LongTensor(
            [[SOS_ID] * batch_size * beam_size]).to(self.device).transpose(0, 1)

        # Store decisions for backtracking
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()

        for _ in range(0, max_len):
            # output: [batch_size * beam_size, vocab_size]
            output, dec_hidden, _ = self.decoder(
                dec_input.transpose(0, 1), # [1, batch_size * beam_size]
                dec_hidden,
                q_enc_outputs=q_enc_outputs,
                q_enc_length=q_inputs_length,
                c_enc_outputs=c_enc_outputs,
                c_enc_length=c_turn_length,
                f_enc_outputs=f_enc_outputs,
                f_enc_length=f_topk_length
            )

            # output: [1, batch_size * beam_size, vocab_size]
            output = output.squeeze(0).contiguous()

            # log_softmax_output: [batch_size * beam_size, vocab_size]
            log_softmax_output = F.log_softmax(output, dim=1)

            # To get the full sequence scores for the new candidates, add the local
            # scores for t_i to the predecessor scores for t_(i-1)
            #  sequence_scores = sequence_scores.repeat(1, vocab_size)
            # [batch_size * beam_size, 1] + [batch_size * beam_sizes] -> [batch_size * beam_size, vocab_size]
            #  print('log_softmax_output: ', log_softmax_output.shape)
            #  print('sequence_scores: ', sequence_scores.shape)
            sequence_scores = sequence_scores + log_softmax_output
            # [batch_size, beam_size]
            scores, candidates = sequence_scores.view(batch_size, -1).topk(beam_size, dim=1)

            dec_input = (candidates % vocab_size).view(batch_size * beam_size, 1)  # [batch_size, beam_size, 1]
            sequence_scores = scores.view(batch_size * beam_size, 1)

            # Update fields for next timestep
            predecessors = (candidates / vocab_size +
                            self.pos_index.expand_as(candidates)).view(batch_size * beam_size, 1)

            dec_hidden = dec_hidden.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = dec_input.data.eq(EOS_ID)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(dec_input)

        # Do backtracking to return the optimal values
        score, topk_length, topk_sequence = self._backtrack(
            stored_predecessors,
            stored_emitted_symbols,
            stored_scores,
            batch_size,
            beam_size,
            max_len
        )

        metadata = {}
        metadata['score'] = score
        metadata['topk_length'] = topk_length
        metadata['topk_sequence'] = topk_sequence

        metadata['length'] = [seq_len[0] for seq_len in topk_length]
        metadata['sequence'] = [seq[0] for seq in topk_sequence]
        return metadata

    def _backtrack(self, predecessors, symbols, scores, batch_size, beam_size, max_len):
        """Backtracks over batch to generate optimal beam_size-sequences.
        Args:
            predecessors [(batch*beam_size)] * sequence_length: A Tensor of predecessors
            symbols [(batch*beam_size)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*beam_size)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            batch_size: Size of the batch
            hidden_size: Size of the dec_hidden state
        Returns:
            output [(batch, beam_size, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the decoder, for every n = [0, ... , seq_len - 1]
            from the last layer of the decoder, for every n = [0, ... , seq_len - 1]
            score [batch, beam_size]: A list containing the final scores for all top-beam_size sequences
            length [batch, beam_size]: A list specifying the length of each sequence in the top-beam_size candidates
            topk_sequence (batch, beam_size, sequence_len): A Tensor containing predicted sequence
        """
        #  batch_size, beam_size = self.config.batch_size, self.config.beam_size

        topk_sequence = list()
        # Placeholder for last dec_hidden state of top-beam_size sequences.
        # the last dec_hidden state of decoding.
        # Placeholder for lengths of top-beam_size sequences
        topk_length = [[max_len] * beam_size for _ in range(batch_size)]

        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(
            batch_size, beam_size).topk(beam_size)
        # initialize the sequence scores with the sorted last step beam scores
        score = sorted_score.clone()

        batch_eos_found = [0] * batch_size   # the number of eos_id found
        # in the backward loop below for each batch

        t = max_len - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with batch_size*beam_size as the first dimension.
        t_predecessors = (
            sorted_idx + self.pos_index.expand_as(sorted_idx)).view(batch_size * beam_size)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see eos_id earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see eos_id early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an eos_id in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(EOS_ID).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the eos_id symbol for both variables
                    # with batch_size*beam_size as the first dimension, and batch_size, beam_size for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / beam_size)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_size - \
                        (batch_eos_found[b_idx] % beam_size) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_size + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]

                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    score[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    topk_length[b_idx][res_k_idx] = t + 1 # record the back tracked results
            topk_sequence.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        score, re_sorted_idx = score.topk(beam_size)
        for b_idx in range(batch_size):
            topk_length[b_idx] = [topk_length[b_idx][k_idx.item()]
                                  for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(batch_size * beam_size)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        topk_sequence = [step.index_select(0, re_sorted_idx).view(batch_size, beam_size, -1) for step in reversed(topk_sequence)]

        score = score.data

        return score, topk_length, topk_sequence
