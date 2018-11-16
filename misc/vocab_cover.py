#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

from tqdm import tqdm
from collections import Counter
from vocab import Vocab

"""
unique word: 233147
cover num: 79997
cover ratio: 0.3431

80004:
    unique word: 210203
    total num: 166507978
    cover num: 160769499
    cover ratio: 0.9655

60004:
    unique word: 210203
    total num: 166507978
    cover num: 160549829
    cover ratio: 0.9642

50004:
    unique word: 210203
    total num: 166507978
    cover num: 160348172
    cover ratio: 0.9630

kg 60004:
    unique word: 514445
    total num: 256787112
    cover num: 248868534
    cover ratio: 0.9692
"""

def vocab_cover_stats(pair_path, facts_path=None, vocab=None):
    unique_words = Counter()
    if facts_path is not None:
        with open(facts_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                _, _, _, fact = line.rstrip().split('\t')
                unique_words.update(fact.split())


    with open(pair_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            _, conversation, response, hash_value = line.split('SPLITTOKEN')

            # skip if source has nothing
            if conversation == 'START' or len(conversation.rstrip()) == 0:
                continue

            if conversation.startswith('start eos'):
                # START: special symbol indicating the start of the
                # conversation
                conversation = conversation[10:]
            elif conversation.startswith('eos'):
                # EOS: special symbol indicating a turn transition
                conversation = conversation[4:]
            conversation_turns = conversation.split('eos')
            for conversation_turn in conversation_turns:
                unique_words.update(conversation_turn.split(' '))
            unique_words.update(response.split(' '))

    print('unique word: %d ' % len(unique_words))
    total_num = sum(unique_words.values())
    cover_num = 0
    for word in unique_words.keys():
        if vocab.word2idx.get(word, None) is not None:
            cover_num += unique_words.get(word)

    print('total num: %d ' % total_num)
    print('cover num: %d ' % cover_num)
    print('cover ratio: %.4f ' % (cover_num / total_num))

if __name__ == '__main__':
    pair_path = './../data/conversations_responses.pair.txt'
    facts_path = './../data/facts.txt'
    vocab = Vocab()
    vocab.load('./../data/vocab_word2idx_kg.60004.dict')

    vocab_cover_stats(pair_path, facts_path, vocab)

