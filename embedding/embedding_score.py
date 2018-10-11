# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn.functional as F

'''
Average:
An utterance representation can be obtained by averaging the embeddings of all the words in that utterance, of which the cosine similarity gives the Average metric
'''


def get_top_k_fact_average_batch(encoder_embedding, fact_embedding, encoder_embedding_size,
                                fact_embedding_size, encoder_inputs, fact_inputs,
                                batch_size=1, top_k=20, device=None):

    encoder_inputs_embedded = encoder_embedding(encoder_inputs)
    encoder_inputs_embedded_mean = encoder_inputs_embedded.transpose(0, 1).mean(dim=1)  # [batch_size, embedding_size]

    new_fact_inputs = torch.zeros((batch_size, top_k, fact_embedding_size), device=device)
    for bi in range(batch_size):
        cur_fact_input = fact_inputs[bi]
        # cur_fact_input: [total_n, len()]
        cur_fact_input_embedded_mean = torch.zeros(
            (len(cur_fact_input), fact_embedding_size), device=device)
        for fi, cur_part_fact in enumerate(cur_fact_input):
            cur_part_fact = torch.LongTensor(cur_part_fact, device=device)
            cur_part_fact_embedded = fact_embedding(cur_part_fact.view(1, -1))
            cur_part_fact_embedded_mean = cur_part_fact_embedded.mean(dim=1)
            cur_fact_input_embedded_mean[fi] = cur_part_fact_embedded_mean.view(-1)

        # get top_k
        cur_encoder_input_embedded_mean = encoder_inputs_embedded_mean[bi]
        cosine_scores = F.cosine_similarity(
            cur_encoder_input_embedded_mean, cur_fact_input_embedded_mean)

        # sort
        sorted_scores, sorted_indices = cosine_scores.sort(
            dim=0, descending=True)
        top_k_indices = sorted_indices[:top_k]
        # [top_k, embedding_size]
        top_k_fact_embedded = cur_fact_input_embedded_mean[top_k_indices]
        new_fact_inputs[bi] = top_k_fact_embedded

    return new_fact_inputs


'''without batch'''


def get_top_k_fact_average(encoder_embedding, fact_embedding, encoder_embedding_size,
                           fact_embedding_size, conversation_ids, facts_ids,
                           top_k=20, device=None):
    """
    Args:
        - conversation_ids: [len]
        - facts_ids: total_num, len
    """
    with torch.no_grad():
        encoder_input = torch.tensor(conversation_ids, dtype=torch.long, device=device)
        encoder_input_embedded = encoder_embedding(encoder_input, is_dropout=False)
        encoder_input_embedded_mean = encoder_input_embedded.mean(dim=0).unsqueeze(0)  # [1, embedding_size]

        facts_embedded_mean = torch.zeros((len(facts_ids), fact_embedding_size), device=device)
        for fi, fact_ids in enumerate(facts_ids):
            fact_input = torch.tensor(fact_ids, dtype=torch.long, device=device)
            fact_input_embedded = fact_embedding(fact_input, is_dropout=False)
            fact_input_embedded_mean = fact_input_embedded.mean(dim=0)  # [embedding_size]
            facts_embedded_mean[fi] = fact_input_embedded_mean

        # get top_k
        print(facts_embedded_mean.shape)
        print(encoder_input_embedded_mean.shape)
        cosine_scores = F.cosine_similarity(facts_embedded_mean, encoder_input_embedded_mean)  # [len(facts_ids)]
        # sort
        _, sorted_indices = cosine_scores.sort(dim=0, descending=True)

        top_k_indices = sorted_indices[:top_k]
        print(top_k_indices)

        #  top_k_facts_embedded_mean = torch.ones((top_k, fact_embedding_size), device=device)
        #  for i in top_k_indices:
            #  top_k_facts_embedded_mean[i] = facts_embedded_mean[i]

        # [top_k, embedding_size]
        top_k_facts_embedded_mean = facts_embedded_mean[top_k_indices]
        del facts_embedded_mean
        del encoder_input_embedded_mean
        del encoder_input_embedded
        del encoder_input
        del cosine_scores

    return top_k_facts_embedded_mean.detach().cpu(), top_k_indices.detach().cpu()


'''
Extreme:
Achieve an utterance representation by taking the largest extreme values among the embedding vectors of all the words it contains
'''


def get_extreme_embedding_score(vocab, gensim_model, input_str, candidate_replies,
                                stop_word_obj, lower=None, normal=False):
    query_words = stop_word_obj.remove_words(input_str.strip().replace('\t', ' '))  # to lower if lower:
    query_words = [word.lower() for word in query_words]

    query_words_embedding = []
    for word in query_words:
        try:
            word_embedding = gensim_model.wv[word]
        except KeyError:
            word_embedding = gensim_model.wv[vocab.unk]
        query_words_embedding.append(word_embedding)

    extreme_vector_query = np.array(query_words_embedding).max(axis=0)

    extreme_matrix_candidate = []
    for candidate_reply in candidate_replies:
        candidate_words = stop_word_obj.remove_words(
            candidate_reply.strip().replace('\t', ' '))

        # to lower
        if lower:
            candidate_words = [word.lower() for word in candidate_words]

        candidate_words_embedding = []
        for word in candidate_words:
            try:
                word_embedding = gensim_model.wv[word]
            except KeyError:
                word_embedding = gensim_model.wv[vocab.unk]

            candidate_words_embedding.append(word_embedding)

        extreme_vector_candidate = np.array(
            candidate_words_embedding).max(axis=0)
        extreme_matrix_candidate.append(extreme_vector_candidate)

    extreme_matrix_candidate = np.array(extreme_matrix_candidate)

    score_vector = gensim_model.cosine_similarities(
        extreme_vector_query, extreme_matrix_candidate)

    # normalization
    if normal:
        score_vector = score_normalization(score_vector)

    return score_vector


'''
Greedy:
Greedily match words in two given utterances based on the cosine similarities of their embeddings, and to average the obtained scores
'''


def get_greedy_embedding_score(vocab, gensim_model, input_str, candidate_replies, stop_word_obj, lower=None,
                               normal=False):
    query_words = stop_word_obj.remove_words(
        input_str.strip().replace('\t', ' '))

    # to lower
    if lower:
        query_words = [word.lower() for word in query_words]

    score_vector = []
    for candidate_reply in candidate_replies:
        candidate_words = stop_word_obj.remove_words(
            candidate_reply.strip().replace('\t', ' '))

        # to lower
        if lower:
            candidate_words = [word.lower() for word in candidate_words]

        max_scores = []
        for query_word in query_words:
            max_score = 0.0
            for candidate_word in candidate_words:
                try:
                    score = gensim_model.wv.similarity(
                        query_word, candidate_word)
                except KeyError:
                    score = -1.0

                max_score = max(score, max_score)

            max_scores.append(max_score)

        score_vector.append(np.mean(max_scores))

    # normalization
    if normal:
        score_vector = score_normalization(score_vector)

    return score_vector


'''
Optimal Matching
TODO
'''

'''
Word Mover's Distance
'''


def get_wmd_embedding_score(gensim_model, input_str, candidate_replies, stop_word_obj, lower=None, normal=False):
    distance_vector = np.random.rand(len(candidate_replies))

    query_words = stop_word_obj.remove_words(
        input_str.strip().replace('\t', ' '))
    # to lower
    if lower:
        query_words = [word.lower() for word in query_words]

    for idx, candidate_reply in enumerate(candidate_replies):
        candidate_words = stop_word_obj.remove_words(
            candidate_reply.strip().replace('\t', ' '))
        # to lower
        if lower:
            candidate_words = [word.lower() for word in candidate_words]

        distance_vector[idx] = gensim_model.wmdistance(
            query_words, candidate_words)

    # distance to score
    score_vector = np.max(distance_vector) - distance_vector

    # normalization
    if normal:
        score_vector = score_normalization(score_vector)

    return score_vector


'''
Max-Min normalization.
'''


def score_normalization(score_vector):
    max_score = np.max(score_vector)
    min_score = np.min(score_vector)
    score_vector = np.divide((score_vector - min_score),
                             (max_score - min_score) * 1.0)
    return score_vector
