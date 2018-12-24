# -*- coding: utf-8 -*-

import os
import sys

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
    tmp_convos_path = 'tmp.convos.txt'
    tmp_convos_file = open(tmp_convos_path, 'w', encoding='utf-8')
    facts_file = open(args.facts_save_path, 'w', encoding='utf-8')

    missings = []
    for target_name, names_str in targets_dict.items():
        filenames = names_str.split()
        for filename in filenames:
            filepath = os.path.join(args.data_dir, filename)
            if not os.path.exists(filepath):
                missings.append(filename)
                continue

            parts = target_name.split('_')
            data_type = parts[1]

            if parts[-1] == 'REFS':
                data_type = 'REFS'

            logger.info("filename: %s" % (filename))
            #  logger.info("data_type: %s" % (data_type))

            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip()

                    if filename.endswith('convos.txt') or filename.endswith('refs.txt'):
                        tmp_convos_file.write('%s\t%s\n' % (data_type, line))
                    elif filename.endswith('facts.txt'):
                        facts_file.write('%s\t%s\n' % (data_type, line))

    tmp_convos_file.close()
    facts_file.close()
    os.system('cat %s | sort -R | uniq > %s' % (tmp_convos_path, args.convos_save_path))

    os.system('rm -f %s' % tmp_convos_path)
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
