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

def get_topk_facts(embedding_size,
                   embedding,
                   conversation_ids,
                   facts_ids,
                   topk,
                   facts_weight,
                   device):
    """
    Args:
        - conversation_ids: [len]
        - facts_ids: total_num, len
    """
    conversation_ids = torch.tensor(conversation_ids, dtype=torch.long, device=device)
    conversation_embedded = embedding(conversation_ids)
    conversation_embedded_mean = conversation_embedded.mean(dim=0).unsqueeze(0)  # [1, embedding_size]

    #  facts_embedded_mean = torch.zeros((len(facts_ids), embedding_size), device=device)
    facts_embedded_mean = []
    for fi, fact_ids in enumerate(facts_ids):
        fact_ids = torch.tensor(fact_ids, dtype=torch.long, device=device)
        fact_embedded = fact_embedding(fact_ids)
        fact_embedded_mean = fact_embedded.mean(dim=0).unsqueeze(0)  # [1, embedding_size]
        #  facts_embedded_mean[fi] = fact_embedded_mean
        facts_embedded_mean.append(fact_embedded_mean)

    facts_embedded_mean = torch.cat(facts_embedded_mean, dim=0) # [len(facts_ids), embedding_size]

    # get topk
    cosine_scores = F.cosine_similarity(facts_embedded_mean, conversation_embedded_mean)  # [len(facts_ids)]

    # * tag_weight
    cosine_scores = cosine_scores * facts_weight

    # sort
    _, sorted_indices = cosine_scores.sort(dim=0, descending=True)

    topk_indexes = sorted_indices[:topk]
    # [topk, embedding_size]

    topk_facts_embedded = torch.index_select(facts_embedded_mean, 0, topk_indexes)

    del facts_embedded_mean
    del conversation_embedded_mean
    del conversation_embedded
    del conversation_ids
    del cosine_scores

    return topk_facts_embedded.detach(), topk_indexes.detach()


'''
Extreme:
Achieve an utterance representation by taking the 
largest extreme values among the embedding vectors of all the words it contains
'''


def get_extreme_embedding_score(vocab, gensim_model, input_str, candidate_replies,
                                stop_word_obj, lower=None, normal=False):
    # to lower if lower:
    query_words = stop_word_obj.remove_words(
        input_str.strip().replace('\t', ' '))
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
Greedily match words in two given utterances based on the cosine similarities of their embeddings, 
and to average the obtained scores
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
