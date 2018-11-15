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

def _inflate(tensor, times, dim):
        """
        Given a tensor, 'inflates' it along the given dimension by replicating each slice specified number of times (in-place)
        Args:
            tensor: A :class:`Tensor` to inflate
            times: number of repetitions
            dim: axis for inflation (default=0)
        Returns:
            A :class:`Tensor`
        Examples::
            >> a = torch.LongTensor([[1, 2], [3, 4]])
            >> a
            1   2
            3   4
            [torch.LongTensor of size 2x2]
            >> b = ._inflate(a, 2, dim=1)
            >> b
            1   2   1   2
            3   4   3   4
            [torch.LongTensor of size 2x4]
            >> c = _inflate(a, 2, dim=0)
            >> c
            1   2
            3   4
            1   2
            3   4
            [torch.LongTensor of size 4x2]
        """
        repeat_dims = [1] * tensor.dim()
        repeat_dims[dim] = times

        return tensor.repeat(*repeat_dims)

class BeamNode:
    def __init__(self, word_idx, log_prob):
        self.sentence = '' + str(word_idx)
        self.log_prob = 0.0 + log_prob

    def push(self, word_idx, log_prob):
        self.sentence += ',' + str(word_idx)
        self.log_prob += log_prob

    def get_ids(self):
        return [int(item) for item in self.sentence.split(',')]

    def get_score(self, p=0.0001):
        reward = 0.0
        ids = self.get_ids()
        for id in ids:
            if id not in [0, 1, 2, 3, 4]:
                reward += p

        return self.log_prob / len(ids) + reward


def beam_decode(
    decoder,
    h_encoder_outputs,
    h_decoder_lengths,
    decoder_hidden_state,
    decoder_input,
    batch_size,
    _beam_width,
    best_n,
    eosid,
    r_max_len,
    vocab_size,
    device,
    z=None):

    """
    Args:
        decoder_hidden_state: [layers, batch, hidden_size]
        decoder_input: [1, batch_size] * sosid
    """
    rnn_type = decoder.rnn_type
    batch_utterances = []
    for bi in range(batch_size):

        beam_width = _beam_width

        init_h_encoder_outputs = None
        init_h_decoder_length = None
        init_z = None
        if h_decoder_lengths is not None:
            init_h_encoder_outputs = h_encoder_outputs[:, bi, :].unsqueeze(1).contiguous()  # [num, 1, hidden_size]
            init_h_decoder_length = h_decoder_lengths[bi].view(1)
        if z is not None:
            init_z = z[:, bi, :].unsqueeze(1).contiguous() #[num_layers, 1, latent_size]

        if rnn_type == 'GRU':
            init_decoder_hidden_state = decoder_hidden_state[:, bi, :].unsqueeze(1).contiguous()  # [layers, 1, hidden_size]
        else:
            init_decoder_hidden_state = tuple([item[:, bi, :].unsqueeze(1).contiguous() for item in decoder_hidden_state])  # [layers, 1, hidden_size]

        init_decoder_input = decoder_input[:, bi].view(1, -1).contiguous()  # [1, 1]

        node_queue = Queue()

        output, hidden_state, _ = decoder(
            init_decoder_input,
            init_decoder_hidden_state,
            init_h_encoder_outputs,
            init_h_decoder_length,
            init_z
        ) # output: [1, 1, vocab_size], hidden_sate: [num_layers, 1, hidden_size]

        log_probs, indices = output.topk(beam_width, dim=2) # [1, 1, beam_width]

        init_node_list = []
        for word_idx, log_prob in zip(indices.view(-1).tolist(), log_probs.view(-1).tolist()):
            node = BeamNode(word_idx, log_prob)
            init_node_list.append(node)

        node_queue.put(init_node_list)

        # for next step
        next_decoder_input = indices.view(1, -1).contiguous() #[1, beam_width]
        if rnn_type == 'GRU':
            #  next_decoder_hidden_state = hidden_state.repeat(1, beam_width, 1) #[num_layers, beam_width, hidden_size]
            next_decoder_hidden_state = _inflate(hidden_state, beam_width, dim=1)
        else:
            #  next_decoder_hidden_state = tuple([item.repeat(1, beam_width, 1) for item in hidden_state])  # [layers, 1, hidden_size]
            next_decoder_hidden_state = tuple([_inflate(item, beam_width, dim=1) for item in hidden_state])  # [layers, 1, hidden_size]

        next_h_encoder_outputs = None
        next_h_decoder_length = None
        if h_decoder_lengths is not None:
            next_h_encoder_outputs = _inflate(init_h_encoder_outputs, beam_width, dim=1)
            next_h_decoder_length = _inflate(init_h_decoder_length, beam_width, dim=0)

        next_z = None
        if z is not None:
            next_z = _inflate(init_z, beam_width, dim=1)

        res = []

        for ri in range(r_max_len):
            outputs, hidden_states, _ = decoder(
                next_decoder_input,
                next_decoder_hidden_state,
                next_h_encoder_outputs,
                next_h_decoder_length,
                next_z
            )

            # squeeze
            log_probs, indices = outputs.view(-1).topk(beam_width)

            last_node_list = node_queue.get()
            cur_node_list = []

            next_decoder_input = []
            indices_select = []

            for j, (index, log_prob) in enumerate(zip(indices.tolist(), log_probs.tolist())):
                last_j = index // outputs.size(2)
                word_idx = index % outputs.size(2)

                last_node_j = last_node_list[last_j]

                #  if word_idx == eosid:
                    #  tmp_ids = last_node_j.get_ids()
                    #  tmp_score = last_node_j.get_score()
                    #  res.append((tmp_score, tmp_ids))
                    #  beam_width -= 1
                    #  continue
                #  elif word_idx == 4:
                    #  pass

                tmp_node = BeamNode(last_node_j.sentence, last_node_j.log_prob)
                tmp_node.push(word_idx, log_prob)
                cur_node_list.append(tmp_node)

                next_decoder_input.append(word_idx)
                indices_select.append(j)

            node_queue.put(cur_node_list)
            del last_node_list

            if len(next_decoder_input) == 0:
                break

            next_decoder_input = torch.tensor(next_decoder_input, dtype=torch.long, device=device).view(1, -1)
            if len(indices_select) != outputs.size(1):
                indices_select = torch.tensor(indices_select, dtype=torch.long, device=device).view(-1)

                if rnn_type == 'GRU':
                    next_decoder_hidden_state = hidden_states.index_select(dim=1, index=indices_select)
                else:
                    next_decoder_hidden_state = tuple([item.index_select(dim=1, index=indices_select) for item in hidden_states])  # [layers, 1, hidden_size]

                if h_decoder_lengths is not None:
                    next_h_encoder_outputs = next_h_encoder_outputs.index_select(1, indices_select)
                    next_h_decoder_length = next_h_decoder_length.index_select(0, indices_select)

                if z is not None:
                    next_z  = next_z.index_select(1, indices_select)
            else:
                next_decoder_hidden_state = hidden_states

        final_node_list = node_queue.get()

        for node in final_node_list:
            ids = node.get_ids()
            score = node.get_score()
            res.append((score, ids))

        best_n_ids = [ids for _, ids in sorted(res, key=lambda item: item[0], reverse=True)][:]
        #  print('best_n_ids: {}'.format(best_n_ids))
        batch_utterances.append(best_n_ids)

    return batch_utterances
