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

