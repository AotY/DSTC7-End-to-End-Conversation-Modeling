# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import logging
import numpy as np


class DSTCDataSet:
    def __init__(self, post_file, response_file, label_file, vocab, num_timesteps):
        self._vocab = vocab
        self._num_timesteps = num_timesteps

        # matrix
        self._posts = []
        self._responses = []

        # vector
        self._labels = []

        self._indicator = 0

        self._parse_file(post_file, response_file, label_file)

    def _parse_file(self, post_file, response_file, label_file):
        print('Loading data from {}, {}, {}'.format(post_file, response_file, label_file))

        with open(post_file, 'r') as f:
            posts = f.readlines()

        with open(response_file, 'r') as f:
            responses = f.readlines()

        with open(label_file, 'r') as f:
            labels = f.readlines()

        for post, response, label in zip(posts, responses, labels):
            id_post_words = self._vocab.sentence_to_id(post.strip('\n'))
            id_post_words = id_post_words[0: self._num_timesteps]

            id_response_words = self._vocab.sentence_to_id(response.strip('\n'))
            id_response_words = id_response_words[0: self._num_timesteps]

            padding_num_post = self._num_timesteps - len(id_post_words)
            padding_num_response = self._num_timesteps - len(id_response_words)

            id_post_words = id_post_words + [
                self._vocab.unk for i in range(padding_num_post)]

            id_response_words = id_response_words + [
                self._vocab.unk for i in range(padding_num_response)]

            self._posts.append(id_post_words)
            self._responses.append(id_response_words)
            self._labels.append(label.strip('\n'))

        self._posts = np.asarray(self._posts, dtype=np.int32)
        self._responses = np.asarray(self._responses, dtype=np.int32)
        self._labels = np.asanyarray(self._labels, dtype=np.int32)

        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(len(self._posts))
        self._posts = self._posts[p]
        self._responses = self._responses[p]
        self._labels = self._labels[p]

    # next batch method for classification problem
    def next_batch_4classification(self, batch_size):
        end_indicator = self._indicator + batch_size

        if end_indicator > len(self._posts):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size

        if end_indicator > len(self._posts):
            raise Exception("batch_size: %d is too large" % batch_size)

        batch_posts = self._posts[self._indicator: end_indicator]
        batch_responses = self._responses[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]

        self._indicator = end_indicator

        return batch_posts, batch_responses, batch_labels

    # next batch method for generation problem
    def next_batch_4generation(self, batch_size):
        end_indicator = self._indicator + batch_size

        if end_indicator > len(self._posts):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size

        if end_indicator > len(self._posts):
            raise Exception("batch_size: %d is too large" % batch_size)

        batch_posts = self._posts[self._indicator: end_indicator]
        batch_responses = self._responses[self._indicator: end_indicator]

        self._indicator = end_indicator

        return batch_posts, batch_responses


if __name__ == '__main__':
    # train_dataset = TextDataSet(
    #     train_file, vocab, category_vocab, hps.num_timesteps)
    # val_dataset = TextDataSet(
    #     val_file, vocab, category_vocab, hps.num_timesteps)
    # test_dataset = TextDataSet(
    #     test_file, vocab, category_vocab, hps.num_timesteps)

    pass
