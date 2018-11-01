#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Beam Search
"""
import torch

from queue import PriorityQueue


class BeamNode:
    def __init__(self,
                 previous_node,
                 hidden_state,
                 decoder_input,
                 log_prob,
                 length):
        """
        hidden_sate: dialogue_decoder_state
        previous_node: previous BeamNode
        decoder_input:
        length: cur decoded length
        """

        self.previous_node = previous_node
        self.hidden_state = hidden_state
        self.decoder_input = decoder_input
        self.log_prob = log_prob
        self.length = length

        self.score = 0

    def set_score(self, score):
        self.score = score

    def __lt__(self, other):
        if self.length == other.length:
            return self.score < other.score
        else:
            return (self.score / self.length) < (other.score / other.length)


    def __gt__(self, other):
        if self.length == other.length:
            return self.score > other.score
        else:
            return (self.score / self.length) > (other.score / other.length)


    def evaluate_score(self, alpha=1.0):
        score = 0.0
        # Add here a function for shaping a reward
        reward = 0.5

        self.score = - (self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward)

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
    max_queue_size=3000):

    max_queue_size = max(2048, beam_width * r_max_len + 512)
    """
    Args:
        c_encoder_outputs: [len, batch, hidden_size]
        decoder_hidden_state: [layers, batch, hidden_size]
        decoder_input: [1, batch_size] * sosid
    """
    batch_utterances = []
    for bi in range(batch_size):
        c_encoder_outputs_bi = c_encoder_outputs[:, bi, :].unsqueeze(1)  # [max_length, 1, hidden_size]

        if h_encoder_outputs is not None:
            h_encoder_outputs_bi = h_encoder_outputs[:, bi, :].unsqueeze(1)  # [num, 1, hidden_size]
        else:
            h_encoder_outputs_bi = None

        init_decoder_hidden_state = decoder_hidden_state[:, bi, :].unsqueeze(1)  # [layers, 1, hidden_size]
        init_decoder_input = decoder_input[:, bi].unsqueeze(1)  # [1, 1]

        # Number of sentence to generate
        res_nodes = []

        node_queue = PriorityQueue()

        # starting node
        init_node = BeamNode(None,
                             init_decoder_hidden_state,
                             init_decoder_input, 0, 1)

        # start the queue
        init_node.evaluate_score()
        node_queue.put(init_node)
        #  node_queue.put(tuple([float(init_score), init_node]))

        # start beam search
        while True:
            # give up, when decoding takes too long
            if node_queue.qsize() >= max_queue_size or node_queue.qsize() == 0:
                break

            # fetch the best node
            cur_node = node_queue.get()
            cur_score = cur_node.score

            cur_decoder_input = cur_node.decoder_input
            cur_decoder_hidden_state = cur_node.hidden_state

            # break
            if (cur_decoder_input[0][0].item() == eosid or
                    cur_node.length >= r_max_len) and \
                    cur_node.previous_node != None:

                res_nodes.append((cur_score, cur_node))
                # if we reached maximum
                if len(res_nodes) >= best_n:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, next_decoder_hidden_state, _ = decoder(
                cur_decoder_input.contiguous(),
                cur_decoder_hidden_state.contiguous(),
                c_encoder_outputs_bi.contiguous(),
                h_encoder_outputs_bi
            )

            # put here real beam search of top
            log_probs, indices = torch.topk(decoder_output, beam_width, dim=2)

            #  next_nodes = []
            #  count = 0
            for i in range(beam_width):
                next_decoder_input = indices[0, 0, i].view(-1, 1)  # [1, 1]
                log_prob = log_probs[0, 0, i].item()

                next_node = BeamNode(
                    cur_node,
                    next_decoder_hidden_state,
                    next_decoder_input,
                    cur_node.log_prob + log_prob,
                    cur_node.length + 1
                )

                next_node.evaluate_score()
                node_queue.put(next_node)

        # choose n_best paths, back trace them
        if len(res_nodes) == 0:
            res_nodes = [node_queue.get() for _ in range(min(best_n, len(res_nodes)))]

        utterances = []
        for score, node in sorted(res_nodes, key=lambda item: item[0], reverse=True):
            utterance = []
            utterance.append(node.decoder_input.item())

            # back trace
            while node.previous_node is not None:
                node = node.previous_node
                utterance.append(node.decoder_input.item())

            # reverse
            utterance.reverse()
            utterances.append(utterance)

        batch_utterances.append(utterances)
        node_queue = None
        res_nodes = None

    return batch_utterances
