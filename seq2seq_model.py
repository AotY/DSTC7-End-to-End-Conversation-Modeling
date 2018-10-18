# -*- coding: utf-8 -*-

import math
import random
import operator
from queue import PriorityQueue

import torch
import torch.nn as nn

from modules.encoder import RNNEncoder
from modules.decoder import StdRNNDecoder
from modules.utils import init_lstm_orth, init_gru_orth


class Seq2SeqModel(nn.Module):
    '''
    Knowledge-ground model
    Conditoning responses on both conversation history and external "facts", allowing the model
    to be versatile and applicable in an open-domain setting.
    '''

    def __init__(self,
                 dialogue_encoder_embedding_size=300,
                 dialogue_encoder_vocab_size=None,
                 dialogue_encoder_hidden_size=300,
                 dialogue_encoder_num_layers=2,
                 dialogue_encoder_rnn_type='LSTM',
                 dialogue_encoder_dropout_probability=0.5,
                 dialogue_encoder_max_length=32,
                 dialogue_encoder_bidirectional=True,
                 dialogue_encoder_embedding=None,

                 dialogue_decoder_embedding_size=300,
                 dialogue_decoder_vocab_size=None,
                 dialogue_decoder_hidden_size=300,
                 dialogue_decoder_num_layers=2,
                 dialogue_decoder_rnn_type='LSTM',
                 dialogue_decoder_dropout_probability=0.5,
                 dialogue_decoder_max_length=32,
                 dialogue_decoder_embedding=None,
                 dialogue_decoder_pad_id=0,
                 dialogue_decoder_sos_id=2,
                 dialogue_decoder_eos_id=3,
                 dialogue_decoder_attention_type='general',
                 dialogue_decode_type='greedy',
                 dialogue_decoder_tied=False,
                 device=None):
        # super init
        super(Seq2SeqModel, self).__init__()

        '''Dialog encoder parameters'''
        self.dialogue_encoder_vocab_size = dialogue_encoder_vocab_size
        self.dialogue_encoder_embedding_size = dialogue_encoder_embedding_size
        self.dialogue_encoder_hidden_size = dialogue_encoder_hidden_size
        self.dialogue_encoder_num_layers = dialogue_encoder_num_layers
        self.dialogue_encoder_rnn_type = dialogue_encoder_rnn_type
        self.dialogue_encoder_dropout_probability = dialogue_encoder_dropout_probability
        self.dialogue_encoder_max_length = dialogue_encoder_max_length
        self.dialogue_encoder_bidirectional = dialogue_encoder_bidirectional

        '''Dialog decoder parameters'''
        self.dialogue_decoder_vocab_size = dialogue_decoder_vocab_size
        self.dialogue_decoder_embedding_size = dialogue_decoder_embedding_size
        self.dialogue_decoder_hidden_size = dialogue_decoder_hidden_size
        self.dialogue_decoder_num_layers = dialogue_decoder_num_layers
        self.dialogue_decoder_rnn_type = dialogue_decoder_rnn_type
        self.dialogue_decoder_dropout_probability = dialogue_decoder_dropout_probability
        self.dialogue_decoder_max_length = dialogue_decoder_max_length
        self.dialogue_decoder_pad_id = dialogue_decoder_pad_id
        self.dialogue_decoder_sos_id = dialogue_decoder_sos_id
        self.dialogue_decoder_eos_id = dialogue_decoder_eos_id
        self.dialogue_decoder_attention_type = dialogue_decoder_attention_type
        self.dialogue_decode_type = dialogue_decode_type
        self.dialogue_decoder_tied = dialogue_decoder_tied

        self.device = device
        # num_embedding, embedding_dim
        ''''Ps: dialogue_encoder, facts_encoder, and dialogue_decoder may have different
            word embedding.
        '''

        # Dialog Encoder
        self.dialogue_encoder = RNNEncoder(
            rnn_type=self.dialogue_encoder_rnn_type,
            bidirectional=self.dialogue_encoder_bidirectional,
            num_layers=self.dialogue_encoder_num_layers,
            hidden_size=self.dialogue_encoder_hidden_size,
            dropout=self.dialogue_encoder_dropout_probability,
            embedding=dialogue_encoder_embedding,
            device=device)

        # get the recommended gain value for the given nonlinearity function.
        gain = nn.init.calculate_gain('sigmoid')

        if self.dialogue_encoder_rnn_type == 'LSTM':
            init_lstm_orth(self.dialogue_encoder.rnn, gain)
        elif self.dialogue_encoder_rnn_type == 'GRU':
            init_gru_orth(self.dialogue_encoder.rnn, gain)

        # Dialog Decoder with Attention
        self.dialogue_decoder = StdRNNDecoder(
            rnn_type=self.dialogue_decoder_rnn_type,
            bidirectional_encoder=self.dialogue_encoder_bidirectional,
            num_layers=self.dialogue_decoder_num_layers,
            hidden_size=self.dialogue_decoder_hidden_size,
            dropout=self.dialogue_decoder_dropout_probability,
            embedding=dialogue_decoder_embedding,
            attn_type=self.dialogue_decoder_attention_type)

        if self.dialogue_decoder == 'LSTM':
            init_lstm_orth(self.dialogue_decoder.rnn, gain)
        elif self.dialogue_encoder_rnn_type == 'GRU':
            init_gru_orth(self.dialogue_decoder.rnn, gain)

        self.dialogue_decoder_linear = nn.Linear(self.dialogue_decoder_hidden_size, self.dialogue_decoder_vocab_size)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if self.dialogue_decoder_tied:
            if self.dialogue_decoder_embedding_size != self.dialogue_decoder_hidden_size:
                raise ValueError(
                    'When using the tied flag, hidden_size must be equal to embedding_size.')
            print('using tied..................')
            self.dialogue_decoder_linear.weight = dialogue_decoder_embedding.get_embedding_weight()

        self.dialogue_decoder_softmax = nn.LogSoftmax(dim=1)

    '''
    Seq2SeqModel forward
    '''

    def forward(self,
                dialogue_encoder_inputs,  # LongTensor
                dialogue_encoder_inputs_length,
                dialogue_decoder_inputs,
                teacher_forcing_ratio,
                batch_size=128):

        # init, [-sqrt(3/hidden_size), sqrt(3/hidden_size)]
        dialogue_encoder_state = self.dialogue_encoder.init_hidden(batch_size)

        '''dialogue_encoder forward'''
        dialogue_encoder_state, dialogue_encoder_outputs = self.dialogue_encoder(
            inputs=dialogue_encoder_inputs,
            lengths=dialogue_encoder_inputs_length,
            encoder_state=dialogue_encoder_state)

        '''dialogue_decoder forward'''
        # inputs, encoder_outputs, state, encoder_inputs_length=None
        dialogue_decoder_state = self.dialogue_decoder.init_decoder_state(encoder_final=dialogue_encoder_state)

        dialogue_decoder_outputs = torch.ones((self.dialogue_decoder_max_length,
                                               batch_size, self.dialogue_decoder_hidden_size),
                                              device=self.device) * self.dialogue_decoder_pad_id

        dialogue_decoder_attns_std = torch.zeros((self.dialogue_decoder_max_length, batch_size, self.dialogue_decoder_max_length-1),
                                                 device=self.device,
                                                 dtype=torch.float)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.dialogue_decoder_max_length):
                dialogue_decoder_state, dialogue_decoder_output, \
                    dialogue_decoder_attn = self.dialogue_decoder(
                        inputs=dialogue_decoder_inputs[di].view(1, -1),
                        encoder_outputs=dialogue_encoder_outputs,
                        decoder_state=dialogue_decoder_state,
                        encoder_inputs_length=dialogue_encoder_inputs_length)
                dialogue_decoder_outputs[di] = dialogue_decoder_output
                dialogue_decoder_attns_std[di] = dialogue_decoder_attn['std']
        else:
            # Without teacher forcing: use its own predictions as the next input
            if self.dialogue_decode_type == 'greedy':
                dialogue_decoder_input = torch.ones(
                    (1, batch_size), dtype=torch.long, device=self.device) * self.dialogue_decoder_sos_id
                dialogue_decoder_outputs, dialogue_decoder_attns_std = self.greedy_decode(dialogue_decoder_input, dialogue_encoder_outputs,
                                                                                          dialogue_decoder_state, dialogue_encoder_inputs_length,
                                                                                          dialogue_decoder_outputs, dialogue_decoder_attns_std)
            elif self.dialogue_decode_type == 'beam_search':
                pass
            else:
                raise ValueError(
                    'invalid decoder type: %s, greedy or beam_search' % self.dialogue_decode_type)

        #  dialogue_decoder_outputs -> [tgt_len x batch x hidden] ->
        dialogue_decoder_outputs = self.dialogue_decoder_linear(dialogue_decoder_outputs)

        # log softmax
        dialogue_decoder_outputs = self.dialogue_decoder_softmax(dialogue_decoder_outputs)

        return ((dialogue_encoder_state, dialogue_encoder_outputs),
                (dialogue_decoder_state, dialogue_decoder_outputs, dialogue_decoder_attns_std))

    def evaluate(self,
                 dialogue_encoder_inputs,  # LongTensor
                 dialogue_encoder_inputs_length,
                 batch_size=128):

        dialogue_encoder_state = self.dialogue_encoder.init_hidden(batch_size)

        '''dialogue_encoder forward'''
        dialogue_encoder_state, dialogue_encoder_outputs = self.dialogue_encoder(
            inputs=dialogue_encoder_inputs,
            lengths=dialogue_encoder_inputs_length,
            encoder_state=dialogue_encoder_state)

        '''dialogue_decoder forward'''

        dialogue_decoder_outputs = torch.ones((self.dialogue_decoder_max_length,
                                               batch_size, self.dialogue_decoder_hidden_size),
                                              device=self.device) * self.dialogue_decoder_pad_id

        dialogue_decoder_attns_std = torch.zeros((self.dialogue_decoder_max_length, batch_size, self.dialogue_decoder_max_length-1),
                                                 device=self.device)

        if self.dialogue_decode_type == 'greedy':
            dialogue_decoder_state = self.dialogue_decoder.init_decoder_state(encoder_final=dialogue_encoder_state)
            dialogue_decoder_input = torch.ones((1, batch_size), dtype=torch.long, device=self.device) * self.dialogue_decoder_sos_id
            dialogue_decoder_outputs, dialogue_decoder_attns_std = self.greedy_decode(dialogue_decoder_input, dialogue_encoder_outputs,
                                                                                      dialogue_decoder_state, dialogue_encoder_inputs_length,
                                                                                      dialogue_decoder_outputs, dialogue_decoder_attns_std)
        elif self.dialogue_decode_type == 'beam_search':
            pass
        else:
            raise ValueError(
                'invalid decoder type: %s, greedy or beam_search' % self.dialogue_decode_type)

        # beam search  dialogue_decoder_outputs -> [tgt_len x batch x hidden]
        dialogue_decoder_outputs = self.dialogue_decoder_linear(
            dialogue_decoder_outputs)

        # log softmax
        dialogue_decoder_outputs = self.dialogue_decoder_softmax(
            dialogue_decoder_outputs)

        return ((dialogue_encoder_state, dialogue_encoder_outputs),
                (dialogue_decoder_state, dialogue_decoder_outputs, dialogue_decoder_attns_std))

    """return sentence"""

    def generate(self,
                 dialogue_encoder_inputs,  # LongTensor
                 dialogue_encoder_inputs_length,
                 batch_size=128,
                 beam_width=10,
                 topk=1):

        dialogue_encoder_state = self.dialogue_encoder.init_hidden(batch_size)

        '''dialogue_encoder forward'''
        dialogue_encoder_state, dialogue_encoder_outputs = self.dialogue_encoder(
            inputs=dialogue_encoder_inputs,
            lengths=dialogue_encoder_inputs_length,
            encoder_state=dialogue_encoder_state)

        '''dialogue_decoder forward'''

        dialogue_decoder_outputs = torch.ones((self.dialogue_decoder_max_length,
                                               batch_size, self.dialogue_decoder_hidden_size),
                                              device=self.device) * self.dialogue_decoder_pad_id

        if self.dialogue_decode_type == 'greedy':
            dialogue_decoder_input = torch.ones((1, batch_size), dtype=torch.long, device=self.device) * self.dialogue_decoder_sos_id
            dialogue_decoder_state = self.dialogue_decoder.init_decoder_state(encoder_final=dialogue_encoder_state)
            dialogue_decoder_outputs, _ = self.greedy_decode(dialogue_decoder_input, dialogue_encoder_outputs,
                                                             dialogue_decoder_state, dialogue_encoder_inputs_length,
                                                             dialogue_decoder_outputs, None)
            # beam search  dialogue_decoder_outputs -> [tgt_len x batch x hidden]
            dialogue_decoder_outputs = self.dialogue_decoder_linear(
                dialogue_decoder_outputs)

            # log softmax
            dialogue_decoder_outputs = self.dialogue_decoder_softmax(
                dialogue_decoder_outputs)

            # dialogue_decoder_outputs -> [max_length, batch_size, vocab_sizes]
            dialogue_decoder_outputs_argmax = torch.argmax(dialogue_decoder_outputs, dim=2)

            # [max_length, batch_size] -> [batch_size, max_length]
            return dialogue_decoder_outputs_argmax.transpose(0, 1).detach().cpu().numpy()
        elif self.dialogue_decode_type == 'beam_search':
            batch_utterances = self.beam_search_decode(dialogue_encoder_outputs,
                                                       dialogue_encoder_state,
                                                       dialogue_encoder_inputs_length,
                                                       batch_size,
                                                       beam_width,
                                                       topk)
            return batch_utterances
        else:
            raise ValueError(
                'invalid decoder type: %s, greedy or beam_search' % self.dialogue_decode_type)

    def greedy_decode(self, dialogue_decoder_input, dialogue_encoder_outputs,
                      dialogue_decoder_state, dialogue_encoder_inputs_length,
                      dialogue_decoder_outputs, dialogue_decoder_attns_std=None):
        """
        dialogue_encoder_outputs: [max_length, batch_size, hidden_size]
        dialogue_decoder_state:
        dialogue_encoder_inputs_length: [batch_size]
        dialogue_decoder_outputs: [max_length, batch_size, hidden]
        """
        for di in range(self.dialogue_decoder_max_length):
            dialogue_decoder_state, dialogue_decoder_output, \
                dialogue_decoder_attn = self.dialogue_decoder(inputs=dialogue_decoder_input,
                                                              encoder_outputs=dialogue_encoder_outputs,
                                                              decoder_state=dialogue_decoder_state,
                                                              encoder_inputs_length=dialogue_encoder_inputs_length)

            dialogue_decoder_outputs[di] = dialogue_decoder_output
            if dialogue_decoder_attns_std is not None:
                dialogue_decoder_attns_std[di] = dialogue_decoder_attn['std']
            dialogue_decoder_input = torch.argmax(dialogue_decoder_output, dim=2).detach()

            if dialogue_decoder_input[0][0].item() == self.dialogue_decoder_eos_id:
                break

        return dialogue_decoder_outputs, dialogue_decoder_attns_std

    def beam_search_decode(self, dialogue_encoder_outputs,
                           dialogue_encoder_state, dialogue_encoder_inputs_length,
                           batch_size, beam_width=10, topk=1):
        """
        dialogue_encoder_outputs: [max_length, batch_size, hidden_size]
        dialogue_encoder_state: [num_layers* \
            num_directions, batch_size, hidden_size // 2]
        dialogue_encoder_inputs_length: [batch_size]
        dialogue_decoder_outputs: [max_length, batch_size, hidden_size]

        batch_size:
        beam_width:
        """

        batch_utterances = []
        for bi in range(batch_size):
            if isinstance(dialogue_encoder_state, tuple):  # LSTM
                dialogue_decoder_hidden_bi = tuple([item[:, bi, :].unsqueeze(1) for item in dialogue_encoder_state[bi]])
            else:
                dialogue_decoder_hidden_bi = dialogue_encoder_state[:, bi, :].unsqueeze(1)

            # dialogue_decoder_hidden_bi: [num_layers*num_directions, 1,
            # hidden_size // 2]  ->  [num_layers, 1, hidden_size]
            dialogue_decoder_state_bi = self.dialogue_decoder.init_decoder_state(
                dialogue_decoder_hidden_bi)

            dialogue_encoder_outputs_bi = dialogue_encoder_outputs[:, bi, :].unsqueeze(1)  # [max_length, 1, hidden_size]
            dialogue_encoder_inputs_length_bi = dialogue_encoder_inputs_length[bi]

            dialogue_decoder_input = torch.LongTensor(
                [[self.dialogue_decoder_sos_id]], device=self.device)  # [1, 1]

            # Number of sentence to generate
            res_nodes = []
            number_required = min((topk + 1), topk - len(res_nodes))

            # starting node
            init_node = BeamsearchNode(
                dialogue_decoder_state_bi, None, dialogue_decoder_input, 0, 1)
            node_queue = PriorityQueue()

            # start the queue
            node_queue.put((-init_node.evaluate(), init_node))
            q_size = 1

            # start beam search
            while True:
                # give up, when decoding takes too long
                if q_size > 2000:
                    break

                # fetch the best node
                cur_score, cur_node = node_queue.get()

                cur_dialogue_decoder_input = cur_node.decoder_input
                print('cur_dialogue_decoder_input shape: {}'.format(
                    cur_dialogue_decoder_input.shape))
                print(cur_dialogue_decoder_input)
                cur_dialogue_decoder_state_bi = cur_node.hidden_state

                # break
                if (cur_node.decoder_input.item() == self.dialogue_decoder_eos_id or
                        cur_node.length == self.dialogue_decoder_max_length) \
                        and cur_node.previous_node != None:

                    res_nodes.append((cur_score, cur_node))
                    # if we reached maximum
                    if len(res_nodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                dialogue_decoder_state_bi, decoder_output_bi, _ = self.dialogue_decoder(
                    inputs=cur_dialogue_decoder_input,
                    encoder_outputs=dialogue_encoder_outputs_bi,
                    state=cur_dialogue_decoder_state_bi,
                    encoder_inputs_length=dialogue_encoder_inputs_length_bi)

                # decoder_output_bi: [1, 1, hidden_size]
                # put here real beam search of top
                print('decoder_output_bi shape: {}'.format(
                    decoder_output_bi.shape))
                log_probs, indices = torch.topk(decoder_output_bi, beam_width)

                next_nodes = []
                for new_i in range(beam_width):
                    new_decoder_input = indices[0][0][new_i].view(
                        1, -1)  # [1, 1]
                    new_log_prob = log_probs[0][0][new_i].item()

                    new_node = BeamsearchNode(dialogue_decoder_state_bi, cur_node, new_decoder_input,
                                              cur_node.log_prob + new_log_prob, cur_node.length + 1)

                    new_score = - new_node.evaluate()
                    next_nodes.append((new_score, new_node))

                # put them into queue
                for i in range(len(next_nodes)):
                    node_queue.put(next_nodes[i])

                # increase q_size
                q_size += len(next_nodes) - 1

            # choose n_best paths, back trace them
            if len(res_nodes) == 0:
                res_nodes = [node_queue.get() for _ in range(topk)]

            utterances = []
            for i, score, node in enumerate(sorted(res_nodes, key=operator.itemgetter(0))):
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


class BeamsearchNode(object):
    def __init__(self, hidden_state, previous_node,
                 decoder_input, log_prob, length):
        """
        hidden_sate: dialogue_decoder_state
        previous_node: previous BeamsearchNode
        decoder_input:
        length:
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


# beam search  tensor.numpy()
def beam_search_decoder(encoder_outputs, beam_size):

    if isinstance(encoder_outputs, torch.Tensor):
        # encoder_outputs = encoder_outputs.numpy()
        if encoder_outputs.is_cuda:
            encoder_outputs = encoder_outputs.cpu()
        encoder_outputs = encoder_outputs.detach().numpy()

    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in encoder_outputs:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * (- math.log(row[j]))]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(
            all_candidates, key=lambda tup: tup[1], reverse=False)
        # ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        # select k best
        sequences = ordered[:beam_size]

    outputs = [sequence[0] for sequence in sequences]
    return outputs
