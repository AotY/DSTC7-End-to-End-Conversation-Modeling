# -*- coding: utf-8 -*-

import sys
import os

import argparse
import logging

from misc_opts import merge_convos_facts_opt
from data_targets import targets_dict

'''
merge train, dev, valid, test
to
raw.convos.txt, raw.facts.txt
'''

def merge(args, logger):
    convos_file = open(args.save_convos_path, 'w', encoding='utf-8')
    facts_file = open(args.save_facts_path, 'w', encoding='utf-8')

    missings = []
    for target, names in targets_dict.items():
        names = names.split()
        for name in names:
            path = os.path.join(args.data_dir, name)
            if not os.path.exists(path):
                missings.append(name)
                continue

            logger.info("merge: %s" % (target))

            data_type = target.split('_')[1]
            if target.endswith('REFS'):
                data_type = 'REFS'

            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()

                    if name.endswith('convos.txt') or name.endswith('refs.txt'):
                        convos_file.write('%s\t%s\n' % (data_type, line))
                    elif name.endswith('facts.txt'):
                        facts_file.write('%s\t%s\n' % (data_type, line))

    convos_file.close()
    facts_file.close()
    logger.info('missing: {}'.format(missings))


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
    args = parser.parse_args()

    merge(args, logger)

    logger.info('Merge finished.')
