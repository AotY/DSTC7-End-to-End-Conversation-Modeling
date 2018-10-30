#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Beam Search
"""
import operator
from queue import PriorityQueue

import torch
import torch.nn as nn

class Node(object):
    def __init__(self,
                 hidden_state,
                 previous_node,
                 decoder_input,
                 log_prob,
                 length):
        """
        hidden_sate: dialogue_decoder_state
        previous_node: previous Node
        decoder_input:
        length: cur decoded length
        """

        self.hidden_state = hidden_state
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.log_prob = log_prob
        self.length = length

    def evaluate(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        score = self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward
        return score

    """
    def __lt__(self, other):
        if self.length == self.length:
            return self.log_prob < other.log_prob
        else:
            return self.length < self.length
    """

class BeamSearch(nn.Module):
    def __init__(self, max_queue_size=2000):
        super(BeamSearch, self).__init__()
        self.max_queue_size = max_queue_size

    def forward(self,
                decoder,
                c_encoder_outputs,
                h_encoder_outputs,
                decoder_hidden_state,
                decoder_input,
                batch_size,
                beam_width,
                best_n,
                eosid,
                r_max_len):
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
                h_encoder_outputs_bi = h_encoder_outputs[:, bi, :].unsqueeze(1) # [num, 1, hidden_size]
            else:
                h_encoder_outputs_bi = None

            init_decoder_hidden_state = decoder_hidden_state[:, bi, :].unsqueeze(1) #[layers, 1, hidden_size]
            init_decoder_input = decoder_input[:, bi].unsqueeze(1) #[1, 1]

            # Number of sentence to generate
            res_nodes = []

            # starting node
            init_node = Node(init_decoder_hidden_state, None, init_decoder_input, 0, 1)
            node_queue = PriorityQueue()

            # start the queue
            node_queue.put((-init_node.evaluate(), init_node))

            q_size = 1

            # start beam search
            while True:
                # give up, when decoding takes too long
                if q_size > self.max_queue_size:
                    break

                # fetch the best node
                cur_score, cur_node = node_queue.get()

                cur_decoder_input = cur_node.decoder_input
                cur_decoder_hidden_state = cur_node.hidden_state

                # break
                if (cur_decoder_input[0][0].item() == eosid or \
                    cur_node.length == r_max_len) and \
                    cur_node.previous_node != None:

                    res_nodes.append((cur_score, cur_node))
                    # if we reached maximum
                    if len(res_nodes) >= best_n:
                        break
                    else:
                        continue

                # decode for one step using decoder
                #  print('cur_decoder_input shape: {}'.format(cur_decoder_input.shape))
                #  print('cur_decoder_hidden_state shape: {}'.format(cur_decoder_hidden_state.shape))
                #  print('c_encoder_outputs_bi shape: {}'.format(c_encoder_outputs_bi.shape))

                decoder_output, next_decoder_hidden_state, _ = decoder(
                    cur_decoder_input.contiguous(),
                    cur_decoder_hidden_state.contiguous(),
                    c_encoder_outputs_bi.contiguous(),
                    h_encoder_outputs_bi
                )

                # put here real beam search of top
                log_probs, indices = torch.topk(decoder_output, beam_width, dim=2)
                print('decoder_output shape: {}'.format(decoder_output.shape))

                for i in range(beam_width):
                    next_decoder_input = indices[0, 0, i].view(-1, 1)  # [1, 1]
                    log_prob = log_probs[0, 0, i].item()

                    next_node = Node(
                        next_decoder_hidden_state,
                        cur_node,
                        next_decoder_input,
                        cur_node.log_prob + log_prob,
                        cur_node.length + 1
                    )

                    next_score = - next_node.evaluate()
                    #  print(next_node)
                    #  print('next_score: ', next_score)

                    # put them into queue
                    node_queue.put((next_score, next_node))

                # increase q_size
                q_size += beam_width

            # choose n_best paths, back trace them
            if len(res_nodes) == 0:
                res_nodes = [node_queue.get() for _ in range(best_n)]

            utterances = []
            for score, node in sorted(res_nodes, key=operator.itemgetter(0)):
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
        return batch_utterances


