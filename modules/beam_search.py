#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Beam Search
"""
import math
import random
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
        return self.log_prob / float(self.length - 1 + 1e-6) + alpha * reward



class BeamSearch(nn.Module):
    def __init__(self,
                 max_queue_size=2000
                 ):
        super(BeamSearch, self).__init__()
        slef.max_queue_size = max_queue_size

    def forward(self,
                c_encoder_outputs,
                c_encoder_inputs_length,
                h_encoder_outputs,
                decoder_hidden_state,
                decoder_input,
                batch_size,
                beam_width,
                best_n,
                eosid,
                max_len):
        """
        Args:
            c_encoder_outputs: [len, batch, hidden_size]
            c_encoder_inputs_length: [max_len, batch_size]
            decoder_hidden_state: [layers, batch, hidden_size]
            decoder_input: [1, batch_size] * sosid
        """

        batch_utterances = []
        for bi in range(batch_size):

            c_encoder_outputs_bi = c_encoder_outputs[:, bi, :].unsqueeze(1)  # [max_length, 1, hidden_size]
            c_encoder_inputs_length_bi = c_encoder_inputs_length[:, bi] # [batch]
            
            if h_encoder_outputs is not None:
                h_encoder_outputs_bi = h_encoder_outputs[:, bi, :].unsqueeze(1) # [num, 1, hidden_size]
            else:
                h_encoder_outputs_bi = None

            decoder_hidden_bi = decoder_hidden_state[:, bi, :].unsqueeze(1) #[layers, 1, hidden_size]
            decoder_input_bi = decoder_input[:, bi].unsqueeze(1) #[1, 1]

            # Number of sentence to generate
            res_nodes = []
            number_required = min((self.best_n + 1), self.best_n - len(res_nodes))

            # starting node
            init_node = Node(decoder_hidden_bi, None, decoder_input_bi, 0, 1)
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
                if (cur_decoder_input[0][0].item() == self.eosid or \
                    cur_node.length == self.max_len) and \
                    cur_node.previous_node != None:

                    res_nodes.append((cur_score, cur_node))
                    # if we reached maximum
                    if len(res_nodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden_state, _ = decoder(
                    cur_decoder_input,
                    cur_decoder_hidden_state,
                    c_encoder_outputs_bi,
                    h_encoder_outputs_bi
                )
                    
                # put here real beam search of top
                log_probs, indices = torch.topk(decoder_output, self.beam_width, dim=2)

                for i in range(self.beam_width):
                    next_decoder_input = indices[0, 0, i].view(-1, 1)  # [1, 1]
                    print('next_decoder_input shape: {}'.format(next_decoder_input.shape))
                    log_prob = log_probs[0, 0, i].item()
                    print('log_prob: ', log_prob)

                    next_node = Node(decoder_hidden_state, 
                                    cur_node, 
                                    next_decoder_input,
                                    cur_node.log_prob + new_log_prob,
                                    cur_node.length + 1)

                    new_score = - next_node.evaluate()
                    # put them into queue
                    node_queue.put((next_score, next_node))

                # increase q_size
                q_size += len(next_nodes) - 1

            # choose n_best paths, back trace them
            if len(res_nodes) == 0:
                res_nodes = [node_queue.get() for _ in range(self.best_n)]

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

        



