# -*- coding: utf-8 -*-

import sys
import os

import argparse
import logging

from misc_opts import merge_convos_facts_opt

'''
merge data-official-2011,  data-official-2012-13 2014 2015-2017
to
raw.convos.txt, raw.facts.txt
'''


def merge(opt, logger):
    convos_file = open(opt.save_convos_path, 'w', encoding='utf-8')
    facts_file = open(opt.save_facts_path, 'w', encoding='utf-8')
    convos_hash_set = set()
    facts_hash_set = set()

    for convos_facts_folder in opt.convos_facts_folder_list:
        # Return a list containing the names of the files in the directory.
        for file_name in os.listdir(convos_facts_folder):
            if file_name.endswith('convos.txt'):
                logger.info("merge concos: %s" % (file_name))
                file_path = os.path.join(convos_facts_folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.rstrip()
                        hash_value = line.split('\t')[0]
                        if hash_value in convos_hash_set:
                            continue
                        convos_hash_set.add(hash_value)
                        convos_file.write('%s\n' % line)

            elif file_name.endswith('facts.txt'):
                logger.info("merge facts: %s" % (file_name))
                file_path = os.path.join(convos_facts_folder, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.rstrip()
                        hash_value = line.split('\t')[0]
                        if hash_value in facts_hash_set:
                            continue
                        facts_hash_set.add(hash_value)
                        facts_file.write('%s\n' % line)
            else:
                continue

    convos_file.close()
    facts_file.close()


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s", ' '.join(sys.argv))

    # get optional parameters
    parser = argparse.ArgumentParser(description=program,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    merge_convos_facts_opt(parser)
    opt = parser.parse_args()

    merge(opt, logger)

    logger.info('Merge finished.')
