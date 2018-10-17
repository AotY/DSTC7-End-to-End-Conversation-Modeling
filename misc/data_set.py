# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import pickle

import torch
import numpy as np

from misc import es_helper
from embedding.embedding_score import get_top_k_fact_average

from misc.url_tags_weight import tags_weight_dict


class Seq2seqDataSet:
    """
        assumptions of the data files
        * SOS and EOS are top 2 tokens
        * dictionary ordered by frequency
        """

    def __init__(self,
                 path_conversations_responses_pair,
                 dialogue_encoder_max_length=50,
                 dialogue_encoder_vocab=None,
                 dialogue_decoder_max_length=50,
                 dialogue_decoder_vocab=None,
                 save_path=None,
                 dialogue_turn_num=1,
                 eval_split=0.2,  # how many hold out as eval data
                 device=None,
                 logger=None):

        self.dialogue_encoder_vocab_size = dialogue_encoder_vocab.get_vocab_size()
        self.dialogue_encoder_max_length = dialogue_encoder_max_length
        self.dialogue_encoder_vocab = dialogue_encoder_vocab

        self.dialogue_decoder_vocab_size = dialogue_decoder_vocab.get_vocab_size()
        self.dialogue_decoder_max_length = dialogue_decoder_max_length
        self.dialogue_decoder_vocab = dialogue_decoder_vocab

        self.device = device
        self.logger = logger
        # es
        self.es = es_helper.get_connection()

        self._data_dict = {}
        self._indicator_dict = {}

        self.read_txt(save_path, path_conversations_responses_pair,
                      dialogue_turn_num, eval_split)

    def read_txt(self, save_path, path_conversations_responses_pair, dialogue_turn_num, eval_split):
        self.logger.info('loading data from txt files: {}'.format(
            path_conversations_responses_pair))

        if not os.path.exists(os.path.join(save_path, 'seq2seq_data_dict.pkl')):
            # load source-target pairs, tokenized
            datas = []
            with open(path_conversations_responses_pair, 'r', encoding='utf-8') as f:
                for line in f:
                    conversation, response, hash_value = line.rstrip().split('SPLITTOKEN')
                    if not bool(conversation) or not bool(response):
                        continue
                    response_ids = self.dialogue_decoder_vocab.words_to_id(response.split(' '))
                    if len(response_ids) <= 3:
                            continue

                    # conversation split by EOS, START
                    if conversation.startswith('start eos'):
                        # START: special symbol indicating the start of the
                        # conversation
                        conversation = conversation[10:]
                        history_dialogues = self.assembel_conversation_context(conversation, dialogue_turn_num, 'conversation')
                    elif conversation.startswith('eos'):
                        # EOS: special symbol indicating a turn transition
                        conversation = conversation[4:]
                        history_dialogues = self.assembel_conversation_context(conversation, dialogue_turn_num, 'turn')
                    else:
                        history_dialogues = self.assembel_conversation_context(conversation, dialogue_turn_num, 'other')

                    if history_dialogues is None:
                        continue

                    conversation_context = ' '.join(history_dialogues)
                    conversation_ids = self.dialogue_encoder_vocab.words_to_id(conversation_context.split(' '))
                    conversation_ids = conversation_ids[-min(self.dialogue_encoder_max_length, len(conversation_ids)):]
                    response_ids = response_ids[-min(self.dialogue_decoder_max_length - 1, len(response_ids)):]

                    datas.append((conversation_ids, response_ids))

            np.random.shuffle(datas)
            # train-eval split
            self.n_train = int(len(datas) * (1. - eval_split))
            self.n_eval = len(datas) - self.n_train

            self._data_dict = {
                'train': datas[0: self.n_train],
                'eval': datas[self.n_train:]
            }

            pickle.dump(self._data_dict, open(os.path.join(save_path, 'seq2seq_data_dict.pkl'), 'wb'))
        else:
            self._data_dict = pickle.load(open(os.path.join(save_path, 'seq2seq_data_dict.pkl'), 'rb'))
            self.n_train = len(self._data_dict['train'])
            self.n_eval = len(self._data_dict['eval'])

        self._indicator_dict = {
            'train': 0,
            'eval': 0
        }

    def assembel_conversation_context(self, conversation, dialogue_turn_num=1, symbol='conversation'):
        """
        assemble conversation context by dialogue turn (default 1)
        """
        history_dialogues = conversation.split('eos')
        history_dialogues = [history_dialogue for history_dialogue in history_dialogues if len(history_dialogue.split()) > 3]

        if len(history_dialogues) == 0:
            return None

        history_dialogues = history_dialogues[-min(dialogue_turn_num, len(history_dialogues)): ]

        return history_dialogues

    def reset_data(self, task):
        np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def all_loaded(self, task):
        return self._data_dict[task]

    def load_data(self, task, batch_size):
        # if batch > len(self._data_dict[task] raise error
        task_len = len(self._data_dict[task])
        if batch_size > task_len:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > task_len:
            self.reset_data(task)
            cur_indicator = batch_size

        encoder_inputs = torch.zeros((self.dialogue_encoder_max_length, batch_size),
                                    dtype=torch.long,
                                     device=self.device)
        encoder_inputs_length = []

        decoder_inputs = torch.zeros((self.dialogue_decoder_max_length, batch_size),
                                     dtype=torch.long,
                                     device=self.device)
        decoder_targets = torch.zeros((self.dialogue_decoder_max_length, batch_size),
                                      dtype=torch.long,
                                      device=self.device)

        conversation_texts = []
        response_texts = []

        batch_data = self._data_dict[task][self._indicator_dict[task]: cur_indicator]
        for i, (conversation_ids, response_ids) in enumerate(batch_data):

            # append length
            encoder_inputs_length.append(len(conversation_ids))

            # ids to word
            conversation_text = ' '.join(self.dialogue_encoder_vocab.ids_to_word(conversation_ids))
            response_text = ' '.join(self.dialogue_decoder_vocab.ids_to_word(response_ids))
            conversation_texts.append(conversation_text)
            response_texts.append(response_text)

            # encoder_inputs
            for c, token_id in enumerate(conversation_ids):
                encoder_inputs[c, i] = token_id

            # decoder_inputs
            decoder_inputs[0, i] = self.dialogue_decoder_vocab.sosid
            for r, token_id in enumerate(response_ids):
                decoder_inputs[r + 1, i] = token_id
                decoder_targets[r, i] = token_id

            decoder_targets[len(response_ids), i] = self.dialogue_decoder_vocab.eosid

        # To long tensor
        encoder_inputs_length = torch.tensor(
            encoder_inputs_length, dtype=torch.long)

        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator
        print(encoder_inputs)
        print(decoder_inputs)

        return encoder_inputs, encoder_inputs_length, \
            decoder_inputs, decoder_targets, \
            conversation_texts, response_texts

    def generating_texts(self, batch_utterances, batch_size, decode_type='greedy'):
        """
        decode_type == greedy:
            batch_utterances: [batch_size, max_length]
            return: [batch_size]
        decode_type == 'beam_search':
            batch_utterances: [batch_size, topk, len]
            return: [batch_size, topk]
        """

        batch_texts = []
        if decode_type == 'greedy':
            for bi in range(batch_size):
                text = self.ids_to_text(batch_utterances[bi].tolist())
                batch_texts.append(text)
        elif decode_type == 'beam_search':
            for bi in range(batch_size):
                topk_text_ids = batch_utterances[bi]
                topk_texts = []
                for ids in topk_text_ids:
                    text = self.ids_to_text(ids)
                    topk_texts.append(text)
                batch_texts.append(topk_texts)

        return batch_texts

    def ids_to_text(self, ids):
        words = self.dialogue_decoder_vocab.ids_to_word(ids)
        # remove pad, sos, eos, unk
        words = [word for word in words if word not in [self.dialogue_decoder_vocab.get_pad_unk_sos_eos()]]
        text = ' '.join(words)
        return text

    def save_generated_texts(self, conversation_texts, response_texts, batch_texts, filename, decode_type='greed'):
        with open(filename, 'a', encoding='utf-8') as f:
            for conversation, response, generated_text in zip(conversation_texts, response_texts, batch_texts):
                # conversation, true response, generated_text
                f.write('Conversation: %s\n' % conversation)
                f.write('Response: %s\n' % response)
                if decode_type == 'greedy':
                    f.write('Generated: %s\n' % generated_text)
                elif decode_type == 'beam_search':
                    for i, topk_text in enumerate(batch_texts):
                        f.write('Generated %d: %s\n' % (i, topk_text))

                f.write('---------------------------------\n')




class KnowledgeGroundedDataSet:
    """
    KnowledgeGroundedDataSet
        conversations_responses_pair (Conversation, response, hash_value)
        dialogue_encoder_vocab
        dialogue_encoder_max_length
        fact_vocab
        fact_max_length
        dialogue_decoder_vocab
        dialogue_decoder_max_length
        device
        logger
    """

    def __init__(self,
                 path_conversations_responses_pair=None,
                 dialogue_encoder_max_length=50,
                 dialogue_encoder_vocab=None,
                 fact_vocab=None,
                 fact_max_length=50,
                 dialogue_decoder_max_length=50,
                 dialogue_decoder_vocab=None,
                 save_path=None,
                 eval_split=0.2,  # how many hold out as eval data
                 device=None,
                 logger=None):

        self.dialogue_encoder_vocab_size = dialogue_encoder_vocab.get_vocab_size()
        self.dialogue_encoder_max_length = dialogue_encoder_max_length
        self.dialogue_encoder_vocab = dialogue_encoder_vocab

        self.fact_max_length = fact_max_length
        self.fact_vocab = fact_vocab

        self.dialogue_decoder_vocab_size = dialogue_decoder_vocab.get_vocab_size()
        self.dialogue_decoder_max_length = dialogue_decoder_max_length
        self.dialogue_decoder_vocab = dialogue_decoder_vocab

        self.device = device
        self.logger = logger
        self._data_dict = {}
        self._indicator_dict = {}

        # es
        self.es = es_helper.get_connection()

        # facts dict
        self.top_k_facts_embedded_mean_dict = None

        # read text, prepare data
        self.read_txt(save_path, path_conversations_responses_pair, eval_split)

    def read_txt(self, save_path, path_conversations_responses_pair, eval_split):
        self.logger.info('loading data from txt files: {}'.format(
            path_conversations_responses_pair))

        if not os.path.exists(os.path.join(save_path, 'Knowledge_grounded_data_dict.pkl')):
            # load source-target pairs, tokenized
            datas = []
            with open(path_conversations_responses_pair, 'r', encoding='utf-8') as f:
                for line in f:
                    conversation, response, hash_value = line.rstrip().split('SPLITTOKEN')
                    if not bool(conversation) or not bool(response):
                        continue
                    response_ids = self.dialogue_decoder_vocab.words_to_id(response.split(' '))
                    if len(response_ids) <= 3:
                            continue

                    # conversation split by EOS, START
                    if conversation.startswith('start eos'):
                        # START: special symbol indicating the start of the
                        # conversation
                        conversation = conversation[10:]
                        history_dialogues = self.assembel_conversation_context(conversation, dialogue_turn_num, 'conversation')
                    elif conversation.startswith('eos'):
                        # EOS: special symbol indicating a turn transition
                        conversation = conversation[4:]
                        history_dialogues = self.assembel_conversation_context(conversation, dialogue_turn_num, 'turn')
                    else:
                        history_dialogues = self.assembel_conversation_context(conversation, dialogue_turn_num, 'other')

                    if history_dialogues is None:
                        continue

                    conversation_context = ' '.join(history_dialogues)
                    conversation_ids = self.dialogue_encoder_vocab.words_to_id(conversation_context.split(' '))
                    conversation_ids = conversation_ids[-min(self.dialogue_encoder_max_length, len(conversation_ids)):]
                    response_ids = response_ids[-min(self.dialogue_decoder_max_length - 1, len(response_ids)):]

                    datas.append((conversation_ids, response_ids, hash_value))

            np.random.shuffle(datas)
            # train-eval split
            n_train = int(len(datas) * (1. - eval_split))
            n_eval = len(datas) - n_train

            self._data_dict = {
                'train': datas[0: n_train],
                'eval': datas[n_train:]
            }

            pickle.dump(self._data_dict, open(os.path.join(save_path, 'Knowledge_grounded_data_dict.pkl'), 'wb'))
        else:
            self._data_dict = pickle.load(open(os.path.join(save_path, 'Knowledge_grounded_data_dict.pkl'), 'rb'))

        self._indicator_dict = {
            'train': 0,
            'eval': 0
        }

    def reset_data(self, task):
        np.random.shuffle(self._data_dict[task])
        self._indicator_dict[task] = 0

    def all_loaded(self, task):
        return self._data_dict[task]

    def load_data(self, task, batch_size, top_k, fact_embedding_size):
        # if batch > len(self._data_dict[task] raise error
        task_len = len(self._data_dict[task])
        if batch_size > task_len:
            raise ValueError('batch_size: %d is too large.' % batch_size)

        cur_indicator = self._indicator_dict[task] + batch_size
        if cur_indicator > task_len:
            self.reset_data(task)
            cur_indicator = batch_size

        encoder_inputs = torch.zeros((self.dialogue_encoder_max_length, batch_size),
                                     dtype=torch.long,
                                     device=self.device)
        encoder_inputs_length = []

        decoder_inputs = torch.zeros((self.dialogue_decoder_max_length, batch_size),
                                     dtype=torch.long,
                                     device=self.device)
        decoder_targets = torch.zeros((self.dialogue_decoder_max_length, batch_size),
                                      dtype=torch.long,
                                      device=self.device)

        conversation_texts = []
        response_texts = []

        # facts
        facts_inputs = torch.zeros((batch_size, top_k, fact_embedding_size), device=self.device)
        facts_texts = []

        default_facts_embedded_mean = torch.zeros((top_k, fact_embedding_size), device=self.device)
        batch_data = self._data_dict[task][self._indicator_dict[task]: cur_indicator]
        for i, (conversation_ids, response_ids, hash_value) in enumerate(batch_data):

            # load top_k facts
            top_k_facts_embedded_mean, top_k_fact_texts, \
                top_k_indices_list = self.top_k_facts_embedded_mean_dict.get(hash_value, (default_facts_embedded_mean, None, None))

            # append length
            encoder_inputs_length.append(len(conversation_ids))

            # encoder_inputs
            for t, token_id in enumerate(conversation_ids):
                encoder_inputs[t, i] = token_id

            # decoder_inputs
            decoder_inputs[0, i] = self.dialogue_decoder_vocab.sosid
            for t, token_id in enumerate(response_ids):
                decoder_inputs[t + 1, i] = token_id
                decoder_targets[t, i] = token_id

            decoder_targets[len(response_ids), i] = self.dialogue_decoder_vocab.eosid

            # ids to word
            conversation_texts.append(
                ' '.join(self.dialogue_encoder_vocab.ids_to_word(conversation_ids)))
            response_texts.append(
                ' '.join(self.dialogue_decoder_vocab.ids_to_word(response_ids)))

            facts_inputs[i] = top_k_facts_embedded_mean
            facts_texts.append(top_k_fact_texts)

        # To long tensor
        encoder_inputs_length = torch.tensor(encoder_inputs_length,
                                             dtype=torch.long,
                                             device=self.device)
        # update _indicator_dict[task]
        self._indicator_dict[task] = cur_indicator

        return encoder_inputs, encoder_inputs_length, \
            facts_inputs, decoder_inputs, decoder_targets, \
            conversation_texts, response_texts, facts_texts

    def computing_similarity_offline(self, encoder_embedding, fact_embedding,
                                     encoder_embedding_size, fact_embedding_size,
                                     top_k, device, filename, logger):
        if os.path.exists(filename):
            logger.info('Loading top_k_facts_embedded_mean_dict')
            self.top_k_facts_embedded_mean_dict = pickle.load(
                open(filename, 'rb'))
        else:
            logger.info('Computing top_k_facts_embedded_mean_dict..........')

            top_k_facts_embedded_mean_dict = {}
            with torch.no_grad():
                """ computing similarity between conversation and facts, then saving to dict"""
                for task, datas in self._data_dict.items():
                    logger.info('computing similarity: %s ' % task)

                    for conversation_ids, _, hash_value in datas:
                        if not bool(conversation_ids) or not bool(hash_value):
                            continue

                        # search facts ?
                        hit_count, facts = es_helper.search_facts(self.es, hash_value)
                    
                        # parser html tags, <h1-6> <title> <p> etc.
                        # facts to id
                        facts, facts_weight = get_facts_weight(facts) 
                        facts_ids = [self.fact_vocab.words_to_id(fact.split(' ')) for fact in facts]
                        if len(facts_ids) == 0:
                            continue
                        facts_weight = torch.tensor(facts_weight, dtype=torch.float, device=self.device)

                        facts_ids = [fact_ids[-min(self.fact_max_length, len(facts_ids)): ] for fact_ids in facts_ids]

                        fact_texts = [' '.join(fact) for fact in facts]

                        # score top_k_facts_embedded -> [top_k, embedding_size]
                        top_k_facts_embedded_mean, top_k_indices = get_top_k_fact_average(encoder_embedding, fact_embedding,
                                                                                          encoder_embedding_size, fact_embedding_size,
                                                                                          conversation_ids, facts_ids, facts_weight,
                                                                                          top_k, device)
                        top_k_fact_texts = []
                        top_k_indices_list = top_k_indices.tolist()
                        for topi in top_k_indices_list:
                            top_k_fact_texts.append(fact_texts[topi])

                        top_k_facts_embedded_mean_dict[hash_value] = (
                            top_k_facts_embedded_mean, top_k_fact_texts, top_k_indices_list)

            # save top_k_facts_embedded_mean_dict
            pickle.dump(top_k_facts_embedded_mean_dict, open(filename, 'wb'))
            self.top_k_facts_embedded_mean_dict = top_k_facts_embedded_mean_dict

    def get_facts_weight(facts):
        """ facts: [[w_n] * size]"""
        facts_wight = []
        new_facts = []
        for fact in facts:
            new_fact = ''
            fact_weight = tags_weight_dict['default']
            fact = ' '.join(fact)
            for tag, weight in tags_weight_dict.items():
                if not bool(fact):
                    break
                soup =  BeautifulSoup(fact, 'lxml')
                tag = soup.find(tag)
                if tag is not None:
                    new_fact += tag.text
                    fact_weight = max(fact_weight, weight)
                    fact = fact.replace(str(tag), '')
            new_facts.append(new_fact)
            facts_wight.append(fact_weight)

        return new_facts,  facts_weight


    def generating_texts(self, decoder_outputs_argmax, batch_size):
        """
        decoder_outputs_argmax: [max_length, batch_size]
        return: [text * batch_size]
        """
        batch_texts = []
        decoder_outputs_argmax.transpose_(0, 1)
        for bi in range(batch_size):
            word_ids = decoder_outputs_argmax[bi].tolist()
            words = self.dialogue_encoder_vocab.ids_to_word(word_ids)
            words = [word for word in words if word not in [self.dialogue_decoder_vocab.get_pad_unk_sos_eos()]]
            text = ' '.join(words)
            batch_texts.append(text)
        return batch_texts

    def save_generated_texts(self, conversation_texts, response_texts, batch_texts, top_k_fact_texts, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for conversation, response, generated_text in zip(conversation_texts, response_texts, batch_texts):
                # conversation, true response, generated_text
                f.write('Conversation: %s\n' % conversation)
                f.write('Response: %s\n' % response)
                f.write('Generated: %s\n' % generated_text)
                for fi, fact_text in enumerate(top_k_fact_texts):
                    f.write('Facts %d: %s\n' % (fi, fact_text))
                f.write('---------------------------------\n')



if __name__ == '__main__':
    pass
