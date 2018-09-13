# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function


# from opennmt


def merge_convos_facts_opt(parser):

    # Data options
    group = parser.add_argument_group('Merge Convos Facts')

    group.add_argument('-convos_facts_folder_list',
                       nargs='+',
                       required=True,
                       help="Path to the convos and facts folders.")

    group.add_argument('-save_convos_path', required=True,
                       help="Output file for the convos.")

    group.add_argument('-save_facts_path', required=True,
                       help="Output file for the facts.")




def preprocess_opt(parser):

    # Data options
    group = parser.add_argument_group('Preprocess')


    group.add_argument('-convos_file_path', required=True,
                       help="train, test or valid convos.txt file path.")

    group.add_argument('-facts_file_path', required=True,
                       help="train, test or valid facts.txt file path.")

    group.add_argument('-conversations_num_save_path', required=True,
                       help="Path for saving conversations num.")

    group.add_argument('-responses_num_save_path', required=True,
                       help="Path for saving responses num.")

    group.add_argument('-min_count',
                       type=int,
                       default=3,
                       help="Ignores all words with total frequency lower than this.")

    group.add_argument('-max_vocab_size',
                       default=2e5,
                       type=float,
                       help="Limits the RAM during vocabulary building. Every 10 million word types need about 1GB of RAM.")

