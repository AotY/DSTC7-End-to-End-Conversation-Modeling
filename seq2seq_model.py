# -*- coding: utf-8 -*-

import math
import random
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
                 dialog_encoder_embedding_size=300,
                 dialog_encoder_vocab_size=None,
                 dialog_encoder_hidden_size=300,
                 dialog_encoder_num_layers=2,
                 dialog_encoder_rnn_type='LSTM',
                 dialog_encoder_dropout_probability=0.5,
                 dialog_encoder_max_length=32,
                 dialog_encoder_clipnorm=50.0,
                 dialog_encoder_bidirectional=True,
                 dialog_encoder_embedding=None,

                 dialog_decoder_embedding_size=300,
                 dialog_decoder_vocab_size=None,
                 dialog_decoder_hidden_size=300,
                 dialog_decoder_num_layers=2,
                 dialog_decoder_rnn_type='LSTM',
                 dialog_decoder_dropout_probability=0.5,
                 dialog_decoder_clipnorm=50.0,
                 dialog_decoder_max_length=32,
                 dialog_decoder_embedding=None,
                 dialog_decoder_pad_id=0,
                 dialog_decoder_sos_id=2,
                 dialog_decoder_eos_id=3,
                 dialog_decoder_attention_type='general',
				 dialog_decoder_type='greedy',
                 dialog_decoder_tied=True,
                 device=None):
        # super init
        super(Seq2SeqModel, self).__init__()

        '''Dialog encoder parameters'''
        self.dialog_encoder_vocab_size = dialog_encoder_vocab_size
        self.dialog_encoder_embedding_size = dialog_encoder_embedding_size
        self.dialog_encoder_hidden_size = dialog_encoder_hidden_size
        self.dialog_encoder_num_layers = dialog_encoder_num_layers
        self.dialog_encoder_rnn_type = dialog_encoder_rnn_type
        self.dialog_encoder_dropout_probability = dialog_encoder_dropout_probability
        self.dialog_encoder_max_length = dialog_encoder_max_length
        self.dialog_encoder_clipnorm = dialog_encoder_clipnorm
        self.dialog_encoder_bidirectional = dialog_encoder_bidirectional

        '''Dialog decoder parameters'''
        self.dialog_decoder_vocab_size = dialog_decoder_vocab_size
        self.dialog_decoder_embedding_size = dialog_decoder_embedding_size
        self.dialog_decoder_hidden_size = dialog_decoder_hidden_size
        self.dialog_decoder_num_layers = dialog_decoder_num_layers
        self.dialog_decoder_rnn_type = dialog_decoder_rnn_type
        self.dialog_decoder_dropout_probability = dialog_decoder_dropout_probability
        self.dialog_decoder_max_length = dialog_decoder_max_length
        self.dialog_decoder_clipnorm = dialog_decoder_clipnorm
        self.dialog_decoder_pad_id = dialog_decoder_pad_id
        self.dialog_decoder_sos_id = dialog_decoder_sos_id
        self.dialog_decoder_eos_id = dialog_decoder_eos_id
        self.dialog_decoder_attention_type = dialog_decoder_attention_type
        self.dialog_decoder_type = dialog_decoder_type
        self.dialog_decoder_tied = dialog_decoder_tied

        self.device = device
        # num_embedding, embedding_dim
        ''''Ps: dialog_encoder, facts_encoder, and dialog_decoder may have different
            word embedding.
        '''

        # Dialog Encoder
        self.dialog_encoder = RNNEncoder(
            rnn_type=self.dialog_encoder_rnn_type,
            bidirectional=self.dialog_encoder_bidirectional,
            num_layers=self.dialog_encoder_num_layers,
            hidden_size=self.dialog_encoder_hidden_size,
            dropout=self.dialog_encoder_dropout_probability,
            embedding=dialog_encoder_embedding)

        if self.dialog_encoder_rnn_type == 'LSTM':
            init_lstm_orth(self.dialog_encoder.rnn)
        elif self.dialog_encoder_rnn_type == 'GRU':
            init_gru_orth(self.dialog_encoder.rnn)

        # Dialog Decoder with Attention
        self.dialog_decoder = StdRNNDecoder(
            rnn_type=self.dialog_decoder_rnn_type,
            bidirectional_encoder=self.dialog_encoder_bidirectional,
            num_layers=self.dialog_decoder_num_layers,
            hidden_size=self.dialog_decoder_hidden_size,
            dropout=self.dialog_decoder_dropout_probability,
            embedding=dialog_decoder_embedding,  # maybe replace by dialog_decoder_embedding
            attn_type=self.dialog_decoder_attention_type)

        if self.dialog_decoder == 'LSTM':
            init_lstm_orth(self.dialog_decoder.rnn)
        elif self.dialog_encoder_rnn_type == 'GRU':
            init_gru_orth(self.dialog_decoder.rnn)

        self.dialog_decoder_linear = nn.Linear(
            self.dialog_decoder_hidden_size, self.dialog_decoder_vocab_size)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if self.dialog_decoder_tied:
            if self.dialog_decoder_embedding_size != self.dialog_decoder_hidden_size:
                raise ValueError(
                    'When using the tied flag, hidden_size must be equal to embedding_size.')
            print('using tied..................')
            self.dialog_decoder_linear.weight = dialog_decoder_embedding.get_embedding_weight()

        self.dialog_decoder_softmax = nn.LogSoftmax(dim=1)

    '''
    Seq2SeqModel forward
    '''

    def forward(self,
                dialog_encoder_inputs,  # LongTensor
                dialog_encoder_inputs_length,
                dialog_decoder_inputs,
                teacher_forcing_ratio=0.5,
                batch_size=128):

        # init, [-sqrt(3/hidden_size), sqrt(3/hidden_size)]
        dialog_encoder_state = self.dialog_encoder.init_hidden(
            batch_size, self.device)

        '''dialog_encoder forward'''
        dialog_encoder_state, dialog_encoder_memory_bank = self.dialog_encoder(
            src=dialog_encoder_inputs,
            lengths=dialog_encoder_inputs_length,
            encoder_state=dialog_encoder_state)

        '''dialog_decoder forward'''
        # tgt, memory_bank, state, memory_lengths=None
        dialog_decoder_state = self.dialog_decoder.init_decoder_state(
            encoder_final=dialog_encoder_state)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.dialog_decoder_max_length):
                dialog_decoder_state, dialog_decoder_output, \
                    dialog_decoder_attn = self.dialog_decoder(
                        tgt=dialog_decoder_inputs[di].view(1, -1),
                        memory_bank=dialog_encoder_memory_bank,
                        state=dialog_decoder_state,
                        memory_lengths=dialog_encoder_inputs_length)
                dialog_decoder_outputs[di] = dialog_decoder_output.squeeze(0)
                dialog_decoder_attns_std[di] = dialog_decoder_attn['std'].squeeze(
                    0)
        else:
            # Without teacher forcing: use its own predictions as the next input
            if self.dialog_decoder_type == 'greedy':
                dialog_decoder_input = torch.ones((1, batch_size), dtype=torch.long, device=device) * self.dialog_decoder_sos_id
                dialog_decoder_outputs, dialog_decoder_attns_std = self.greedy_decode(dialog_decoder_input, dialog_encoder_memory_bank, \
                        dialog_decoder_state, dialog_encoder_inputs_length)
            elif self.dialog_decoder_type == 'beam_search':
                pass
                #  dialog_decoder_outputs = self.beam_search_decoder()
            else:
                raise ValueError('invalid decoder type: %s, greedy or beam_search' % self.dialog_decoder_type)

        #  dialog_decoder_outputs -> [tgt_len x batch x hidden]
        dialog_decoder_outputs=self.dialog_decoder_linear(
            dialog_decoder_outputs)

        # log softmax
        dialog_decoder_outputs=self.dialog_decoder_softmax(
            dialog_decoder_outputs)

        return ((dialog_encoder_state, dialog_encoder_memory_bank),
                (dialog_decoder_state, dialog_decoder_outputs, dialog_decoder_attns_std))

    def evaluate(self,
                 dialog_encoder_inputs,  # LongTensor
                 dialog_encoder_inputs_length,
                 batch_size=128):

        dialog_encoder_state=self.dialog_encoder.init_hidden(
            batch_size, self.device)

        '''dialog_encoder forward'''
        dialog_encoder_state, dialog_encoder_memory_bank=self.dialog_encoder(
            src=dialog_encoder_inputs,
            lengths=dialog_encoder_inputs_length,
            encoder_state=dialog_encoder_state)

        '''dialog_decoder forward'''

        dialog_decoder_outputs=torch.ones((self.dialog_decoder_max_length,
                                             batch_size, self.dialog_decoder_hidden_size),
                                            device=self.device) * self.dialog_decoder_pad_id

        dialog_decoder_attns_std=torch.zeros((self.dialog_decoder_max_length,
                                                batch_size, self.dialog_decoder_max_length-2))

        if self.dialog_decoder_type == 'greedy':
            dialog_decoder_state=self.dialog_decoder.init_decoder_state(
                encoder_final=dialog_encoder_state)
            dialog_decoder_outputs, dialog_decoder_attns_std = self.greedy_decode(dialog_decoder_input, dialog_encoder_memory_bank,
                                dialog_decoder_state, dialog_encoder_inputs_length)
        elif self.dialog_decoder_type == 'beam_search':
            #  dialog_decoder_outputs = self.beam_search_decoder()
            pass
        else:
            raise ValueError('invalid decoder type: %s, greedy or beam_search' % self.dialog_decoder_type)

        # beam search  dialog_decoder_outputs -> [tgt_len x batch x hidden]
        dialog_decoder_outputs=self.dialog_decoder_linear(
            dialog_decoder_outputs)

        # log softmax
        dialog_decoder_outputs=self.dialog_decoder_softmax(
            dialog_decoder_outputs)

        return ((dialog_encoder_state, dialog_encoder_memory_bank),
                (dialog_decoder_state, dialog_decoder_outputs, dialog_decoder_attns_std))

    """return sentence"""
    def generate(self,
                 dialog_encoder_inputs,  # LongTensor
                 dialog_encoder_inputs_length,
                 batch_size=128):

        dialog_encoder_state=self.dialog_encoder.init_hidden(
            batch_size, self.device)

        '''dialog_encoder forward'''
        dialog_encoder_state, dialog_encoder_memory_bank=self.dialog_encoder(
            src=dialog_encoder_inputs,
            lengths=dialog_encoder_inputs_length,
            encoder_state=dialog_encoder_state)

        '''dialog_decoder forward'''

        dialog_decoder_outputs=torch.ones((self.dialog_decoder_max_length,
                                             batch_size, self.dialog_decoder_hidden_size),
                                            device=self.device) * self.dialog_decoder_pad_id

        if self.dialog_decoder_type == 'greedy':
            dialog_decoder_input = torch.ones((1, batch_size), dtype=torch.long, device=device) * self.dialog_decoder_sos_id
            dialog_decoder_state=self.dialog_decoder.init_decoder_state(
                encoder_final=dialog_encoder_state)
            dialog_decoder_outputs, dialog_decoder_attns_std = self.greedy_decode(dialog_decoder_input, dialog_encoder_memory_bank,
                                dialog_decoder_state, dialog_encoder_inputs_length)
        elif self.dialog_decoder_type == 'beam_search':
            pass
        else:
            raise ValueError('invalid decoder type: %s, greedy or beam_search' % self.dialog_decoder_type)

        # beam search  dialog_decoder_outputs -> [tgt_len x batch x hidden]
        dialog_decoder_outputs=self.dialog_decoder_linear(
            dialog_decoder_outputs)

        # log softmax
        dialog_decoder_outputs=self.dialog_decoder_softmax(
            dialog_decoder_outputs)

        # dialog_decoder_outputs -> [max_length, batch_size, vocab_sizes]
        dialog_decoder_outputs_argmax = torch.argmax(dialog_decoder_outputs, dim=2)

        return dialog_decoder_outputs_argmax.detach().numpy()

    def greedy_decode(self, dialog_decoder_input, dialog_encoder_memory_bank, 
                      dialog_decoder_state, dialog_encoder_inputs_length):
        """
        dialog_encoder_memory_bank: [max_length, batch_size, hidden_size]
        dialog_decoder_state:
        dialog_encoder_inputs_length: [batch_size]
        dialog_decoder_outputs: [max_length, batch_size, hidden]
        """
        dialog_decoder_outputs = torch.ones((self.dialog_decoder_max_length,
                                             batch_size, self.dialog_decoder_hidden_size),
                                            device=self.device) * self.dialog_decoder_pad_id

        dialog_decoder_attns_std = torch.zeros((self.dialog_decoder_max_length,
                                                batch_size, self.dialog_decoder_max_length-2),
                                               device=self.device,
                                               dtype=torch.float)
        for di in range(self.dialog_decoder_max_length):
            dialog_decoder_state, dialog_decoder_output, \
            dialog_decoder_attn = self.dialog_decoder(tgt=dialog_decoder_input.view(1, -1), 
                    memory_bank=dialog_encoder_memory_bank,
                    state=dialog_decoder_state,
                    memory_lengths=dialog_encoder_inputs_length)

            dialog_decoder_output=dialog_decoder_output.detach().squeeze(0)
            dialog_decoder_outputs[di]=dialog_decoder_output
            dialog_decoder_attns_std[di]=dialog_decoder_attn['std'].squeeze(0)
            dialog_decoder_input=torch.argmax(
                    dialog_decoder_output, dim=1)

            if dialog_decoder_input[0].item() == self.dialog_decoder_eos_id:
                    break

        return dialog_decoder_outputs, dialog_decoder_attns_std

    def beam_search_decode(self, dialog_encoder_memory_bank,
                           dialog_encoder_state, dialog_encoder_inputs_length,
                           batch_size, beam_width=10, topk=1):
        """
        dialog_encoder_memory_bank: [max_length, batch_size, hidden_size]
        dialog_encoder_state: [num_layers*num_directions, batch_size, hidden_size // 2]
        dialog_encoder_inputs_length: [batch_size]
        dialog_decoder_outputs: [max_length, batch_size, hidden_size]

        batch_size:
        beam_width:
        """

        for bi in range(batch_size):
            if isinstance(dialog_encoder_state, tuple): # LSTM
                dialog_decoder_hidden_bi = tuple([item[:, bi, :].unsqueeze(1) for item in dialog_encoder_state[bi]])
            else:
                dialog_decoder_hidden_bi = dialog_encoder_state[:, bi, :].unsqueeze(1)

            # dialog_decoder_hidden_bi: [num_layers*num_directions, 1,
            # hidden_size // 2]  ->  [num_layers, 1, hidden_size]
            dialog_decoder_state_bi = self.dialog_decoder.init_decoder_state(dialog_decoder_hidden_bi)

            dialog_encoder_memory_bank_bi = dialog_encoder_memory_bank[:, bi, :].unsqueeze(1) # [max_length, 1, hidden_size]
            dialog_encoder_inputs_length_bi = dialog_encoder_inputs_length[bi]


            dialog_decoder_input = torch.LongTensor([[self.dialog_decoder_sos_id]], device=self.device) #[1, 1]

            # Number of sentence to generate
            end_nodes = []
            number_required = min((topk + 1), topk - len(end_nodes))

            # starting node
            node = BeamsearchNode(dialog_decoder_state_bi, None, dialog_decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.evaluate(), node))
            q_size = 1

            # start beam search
            while True:
                # give up, when decoding takes too long
                if q_size > 2000:
                    break

                # fetch the best node
                cur_score, cur_node = nodes.get()
                cur_dialog_decoder_input = cur_node.decoder_input
                cur_dialog_decoder_state_bi = cur_node.hidden_state

                if n.decoder_input.item() == self.dialog_decoder_eos_id and cur_node.previous_node != None:
                    end_nodes.append((cur_score, cur_node))
                    # if we reached maximum
                    if len(end_nodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                cur_dialog_decoder_state_bi, decoder_output_bi, attns_bi = self.dialog_decoder(
                    tgt=cur_dialog_decoder_input,
                    memory_bank=dialog_encoder_memory_bank_bi,
                    state=cur_dialog_decoder_state_bi,
                    memory_lengths=dialog_encoder_inputs_length_bi)

                # decoder_output_bi: [1, 1, hidden_size]
                # put here real beam search of top
                log_probs, indices = torch.topk(decoder_output_bi, beam_width)

                next_nodes = []
                for new_i in range(beam_width):
                    new_decoder_input = indices[0][new_i].view(1, -1) # [1, 1]
                    new_log_prob = log_probs[0][new_i].item()

                    new_node = BeamsearchNode(cur_dialog_decoder_state_bi, cur_node, new_decoder_input,
                                              cur_node.log_prob + new_log_prob, node.length + 1)

                    new_score = - new_node.evaluate()
                    next_nodes.append((new_score, new_node))

                # put them into queue
                for i in range(len(next_nodes)):
                    score, node = next_node[i]
                    nodes.put((score, node))

                # increase q_size
                q_size += len(next_nodes) - 1

            # choose n_best paths, back trace them
            if len(end_nodes) == 0:
                end_nodes = [nodes.get() for _ in range(topk)]
            
            """
            dialog_decoder_outputs_bi = torch.ones((self.dialog_decoder_max_length,
                                                    self.), dtype=torch.long,
                                                    device=self.device) * self.dialog_decoder_pad_id

            for i, score, node in enumerate(sorted(end_nodes, key=operator.itemgetter(0))):
                dialog_decoder_output = torch.ones((1, ))
            """













class BeamsearchNode(object):
    def __init__(self, hidden_state, previous_node,
                 decoder_input, log_prob, length):
        """
        hidden_sate: dialog_decoder_state
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
def beam_search_decoder(memory_bank, beam_size):

    if isinstance(memory_bank, torch.Tensor):
        # memory_bank = memory_bank.numpy()
        if memory_bank.is_cuda:
            memory_bank=memory_bank.cpu()
        memory_bank=memory_bank.detach().numpy()

    sequences=[[list(), 1.0]]
    # walk over each step in sequence
    for row in memory_bank:
        all_candidates=list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score=sequences[i]
            for j in range(len(row)):
                candidate=[seq + [j], score * (- math.log(row[j]))]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered=sorted(
            all_candidates, key=lambda tup: tup[1], reverse=False)
        # ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        # select k best
        sequences=ordered[:beam_size]

    outputs=[sequence[0] for sequence in sequences]
    return outputs
