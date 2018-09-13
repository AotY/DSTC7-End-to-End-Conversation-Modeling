# -*- coding: utf-8 -*-

import sys
import os

import argparse
import logging

from opts import merge_convos_facts_opt

'''
merge data-official-2011,  data-official-2012-13
to
train.convos.txt, train.facts.txt
'''

def merge(opt, logger):
    convos_file = open(opt.save_convos_path, 'w', encoding='utf-8')
    facts_file = open(opt.save_facts_path, 'w', encoding='utf-8')

    for convos_facts_folder in opt.convos_facts_folder_list:
        # Return a list containing the names of the files in the directory.
        for file_name in os.listdir(convos_facts_folder):
            if file_name.endswith('convos.txt'):
                logger.info("merge %s" % (file_name))
                file_path = os.path.join(convos_facts_folder, file_name)
                with open(file_path) as f:
                    convos_file.writelines(f.readlines())

            elif file_name.endswith('facts.txt'):
                logger.info("merge %s" % (file_name))
                file_path = os.path.join(convos_facts_folder, file_name)
                with open(file_path) as f:
                    facts_file.writelines(f.readlines())
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
    merge_convos_facts_opt(parser, logger)
    opt = parser.parse_args()

    merge(opt, logger)





