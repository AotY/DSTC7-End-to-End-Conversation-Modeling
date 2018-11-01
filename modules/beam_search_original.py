#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Beam Search,
Original method
"""
import copy
import numpy as np

import torch


def beam_decode(
    decoder,
    c_encoder_outputs,
    h_encoder_outputs,
    decoder_hidden_state,
    decoder_input,
    batch_size,
    beam_width,
    best_n,
    eosid,
    r_max_len,
    vocab_size,
    device):

    """
    Args:
        c_encoder_outputs: [len, batch, hidden_size]
        decoder_hidden_state: [layers, batch, hidden_size]
        decoder_input: [1, batch_size] * sosid
    """
    batch_utterances = []
    for bi in range(batch_size):
        init_c_encoder_outputs = c_encoder_outputs[:, bi, :].unsqueeze(1).contiguous()  # [max_length, 1, hidden_size]

        if h_encoder_outputs is not None:
            init_h_encoder_outputs = h_encoder_outputs[:, bi, :].unsqueeze(1).contiguous()  # [num, 1, hidden_size]
        else:
            init_h_encoder_outputs = None

        init_decoder_hidden_state = decoder_hidden_state[:, bi, :].unsqueeze(1).contiguous()  # [layers, 1, hidden_size]
        init_decoder_input = decoder_input[:, bi].unsqueeze(1).contiguous()  # [1, 1]

        next_input = None
        next_hidden_state = None
        next_decoder_hidden_state = None
        next_c_encoder_outputs = None
        next_h_encoder_outputs = None

        last_scores = [0] * beam_width
        cur_scores = [0] * beam_width

        last_sentences = [[]] * beam_width
        cur_sentences = [[]] * beam_width

        res = []

        for ri in range(r_max_len):

            if len(cur_scores) == 0 or len(cur_sentences) == 0:
                break

            # init step
            if ri == 0:
                output, hidden_state, _ = decoder(
                    init_decoder_input,
                    init_decoder_hidden_state,
                    init_c_encoder_outputs,
                    init_h_encoder_outputs
                ) # output: [1, 1, vocab_size], decoder_hidden_state: [num_layers, 1, hidden_size]

                log_probs, indices = torch.topk(output, beam_width, dim=2) # [1, 1, beam_width]

                # for next step
                next_decoder_input = indices.squeeze(0)

                # accumulate
                last_scores = log_probs.view(-1).tolist()

                for vi, vocab_index in enumerate(indices.tolist()):
                    last_sentences[vi].append(vocab_index)

                # repeat
                next_decoder_hidden_state = hidden_state.repeat(1, beam_width, 1)
                next_c_encoder_outputs = init_c_encoder_outputs.repeat(1, beam_width, 1)

                if init_h_encoder_outputs is not None:
                    next_h_encoder_outputs = init_h_encoder_outputs.repeat(1, beam_width, 1)
            else:
                cur_beam_width = next_decoder_input.size(1)
                print('cur_beam_width: %d' % cur_beam_width)

                output, hidden_state, _ = decoder(
                    next_decoder_input,
                    next_decoder_hidden_state,
                    next_c_encoder_outputs,
                    next_h_encoder_outputs
                ) # output: [1, k, vocab_size], hidden_state: [num_layers, k, hidden_size]

                # squeeze
                output = output.view(-1).contiguous()
                log_probs, indices = output.topk(cur_beam_width)

                next_decoder_input = []
                next_indices = []
                for i, (index, prob) in enumerate(zip(indices.tolist(), log_probs.tolist())):
                    last_index = index // vocab_size
                    vocab_index = index % vocab_size
                    print('last_index: %d, vocab_index: %d' % (last_index, vocab_index))

                    if vocab_index == eosid:
                        res.append((cur_scores[i] / len(cur_sentences[i]), cur_sentences[i]))
                        cur_scores.remove(cur_scores[i])
                        cur_sentences.remove(cur_sentences[i])
                    else:
                        next_decoder_input.append(vocab_index)
                        next_indices.append(i)

                        cur_scores[i] = last_scores[last_index] + prob
                        cur_sentences[i].extend(last_sentences[i])
                        cur_sentences[i].append(vocab_index)

                next_decoder_input = torch.tensor(next_decoder_input, dtype=torch.long, device=device)
                next_decoder_input = next_decoder_input.view(1, -1)

                next_indices = torch.tensor(next_indices, dtype=torch.long, device=device)
                next_decoder_hidden_state = hidden_state.index_select(1, next_indices)
                next_c_encoder_outputs = next_c_encoder_outputs.index_select(1, next_indices)

                if next_h_encoder_outputs is not None:
                    next_h_encoder_outputs = next_h_encoder_outputs.index_select(1, next_indices)

                del last_scores
                del last_sentences
                last_scores = cur_scores.copy()
                last_sentences = copy.deepcopy(cur_sentences)

        for score, sentence in zip(cur_scores, cur_sentences):
            res.append((score / len(sentence), sentence))

        sorted_sentences = sorted(res, key=lambda item: item[0], reverse=True)

        batch_utterances.append(sorted_sentences[:best_n])

    return batch_utterances
