#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Beam Search,
Original method
"""
import torch

from queue import Queue

"""
Beam Node
"""


class BeamNode:
    def __init__(self):
        self.sentence = ''
        self.log_prob = 0.0

    def push(self, word_idx, log_prob):
        self.sentence += str(word_idx) + ','
        self.log_prob += log_prob

    def get_ids(self):
        return [int(item) for item in self.sentence[:-1].split(',')]

    def get_score(self):
        reward = 0.00
        ids = self.get_ids()
        return self.log_prob / len(ids) + reward * len(ids)


def beam_decode(
    decoder,
    c_encoder_outputs,
    h_encoder_outputs,
    decoder_hidden_state,
    decoder_input,
    batch_size,
    _beam_width,
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
        beam_width = _beam_width
        init_c_encoder_outputs = c_encoder_outputs[:, bi, :].unsqueeze(1).contiguous()  # [max_length, 1, hidden_size]

        init_h_encoder_outputs = None
        if h_encoder_outputs is not None:
            init_h_encoder_outputs = h_encoder_outputs[:, bi, :].unsqueeze(1).contiguous()  # [num, 1, hidden_size]

        init_decoder_hidden_state = decoder_hidden_state[:, bi, :].unsqueeze(1).contiguous()  # [layers, 1, hidden_size]
        init_decoder_input = decoder_input[:, bi].view(1, -1).contiguous()  # [1, 1]

        node_queue = Queue()
        output, hidden_state, _ = decoder(
            init_decoder_input,
            init_decoder_hidden_state,
            init_c_encoder_outputs,
            init_h_encoder_outputs
        ) # output: [1, 1, vocab_size], hidden_sate: [num_layers, 1, hidden_size]

        print('output: ', output.shape)
        print('beam_width:', beam_width)
        log_probs, indices = output.topk(beam_width, dim=2) # [1, 1, beam_width]

        init_node_list = []
        for word_idx, log_prob in zip(indices.view(-1).tolist(), log_probs.view(-1).tolist()):
            node = BeamNode()
            node.push(word_idx, log_prob)
            init_node_list.append(node)

        node_queue.put(init_node_list)

        # for next step
        next_decoder_input = indices.squeeze(0) #[1, beam_width]
        next_decoder_hidden_state = init_decoder_hidden_state.repeat(1, beam_width, 1)
        next_c_encoder_outputs = init_c_encoder_outputs.repeat(1, beam_width, 1)

        next_h_encoder_outputs = None
        if init_h_encoder_outputs is not None:
            next_h_encoder_outputs = init_h_encoder_outputs.repeat(1, beam_width, 1)

        res = []

        for ri in range(r_max_len):
            outputs, hidden_states, _ = decoder(
                next_decoder_input,
                next_decoder_hidden_state,
                next_c_encoder_outputs,
                next_h_encoder_outputs
            )

            # squeeze
            log_probs, indices = outputs.view(-1).topk(beam_width)

            last_node_list = node_queue.get()
            cur_node_list = []

            next_decoder_input = []
            indices_select = []

            for j, (log_prob, index) in enumerate(zip(log_probs.tolist(), indices.tolist())):
                last_j = index // outputs.size(2)
                word_idx = index % outputs.size(2)

                if word_idx == eosid:
                    tmp_ids = last_node_list[last_j].get_ids()
                    tmp_score = last_node_list[last_j].get_score()
                    res.append((tmp_score, tmp_ids))
                    beam_width -= 1
                    continue

                tmp_node = BeamNode()
                tmp_node.push(last_node_list[last_j].sentence + str(word_idx), log_prob)
                cur_node_list.append(tmp_node)

                next_decoder_input.append(word_idx)
                indices_select.append(j)

            node_queue.put(cur_node_list)

            if len(next_decoder_input) == 0:
                break

            next_decoder_input = torch.tensor(next_decoder_input, dtype=torch.long, device=device).view(1, -1)
            if len(indices_select) != outputs.size(1):
                indices_select = torch.tensor(indices_select, dtype=torch.long, device=device).view(-1)
                next_decoder_hidden_state = hidden_states.index_select(dim=1, index=indices_select)

                next_c_encoder_outputs = next_c_encoder_outputs.index_select(1, indices_select)
                if next_h_encoder_outputs is not None:
                    next_h_encoder_outputs = next_h_encoder_outputs.index_select(1, indices_select)
            else:
                next_decoder_hidden_state = hidden_states

        final_node_list = node_queue.get()

        for node in final_node_list:
            ids = node.get_ids()
            score = node.get_score()
            res.append((score, ids))

        best_n_sentences = [sentence for _, sentence in sorted(res, key=lambda item: item[0], reverse=True)][:best_n]
        batch_utterances.append(best_n_sentences)

    return batch_utterances
