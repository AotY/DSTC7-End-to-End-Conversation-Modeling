#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""

def turn_stats(pair_path, logger=None):
    turn_dict = {}
    with open(pair_path, 'r', encoding='utf-8') as f:
        for line in f:
            conversation, response, hash_value = line.split('SPLITTOKEN')

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
            turn_num = len(conversation_turns)
            turn_dict[turn_num] = turn_dict.get(turn_num, 0) + 1


def save_distribution(distribution, name, key=None):
    if key is None:
        key = lambda item: item[0]
    distribution_list = sorted(distribution.items(), key=key)
    with open(name + '.distribution.txt', 'w', encoding="utf-8") as f:
        f.write('length\tcount\n')
        for length, count in distribution_list:
            f.write('%d\t%d\n' % (length, count))

if __name__ == '__main__':
    pair_path = './../data/conversations_responses.pair.txt'
    turn_dict = turn_stats(pair_path)
    save_distribution(turn_dict, 'turn_freq')

