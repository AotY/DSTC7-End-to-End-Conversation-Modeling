# -*- coding:utf--8 -*-
from __future__ import division
from __future__ import print_function


# from opennmt


def merge_convos_facts_opt(parser):
    # Data options
    group = parser.add_argument_group('Merge Convos Facts')

    group.add_argument('--convos_facts_folder_list',
                       nargs='+',
                       required=True,
                       help="Path to the convos and facts folders.")

    group.add_argument('--save_convos_path', required=True,
                       help="Output file for the convos.")

    group.add_argument('--save_facts_path', required=True,
                       help="Output file for the facts.")


def preprocess_opt(parser):
    # Data options
    group = parser.add_argument_group('Preprocess')

    group.add_argument('--convos_file_path', required=True,
                       help="train, test or valid convos.txt file path.")

    group.add_argument('--facts_file_path', required=True,
                       help="train, test or valid facts.txt file path.")

    group.add_argument('--min_count',
                       type=int,
                       default=3,
                       help="Ignores all words with total frequency lower than this.")

    group.add_argument('--c_max_len',
                       type=int,
                       help="max len of conversation(including multi-turn dialogue)")

    group.add_argument('--c_min_len',
                       type=int,
                       help="min len of conversation(including multi-turn dialogue)")

    group.add_argument('--r_max_len',
                       type=int,
                       help="max len of response.")

    group.add_argument('--r_min_len',
                       type=int,
                       help="min len of response.")

    group.add_argument('--f_max_len',
                       type=int,
                       help="max len of response.")

    group.add_argument('--f_min_len',
                       type=int,
                       help="min len of response.")

    group.add_argument('--max_vocab_size',
                       default=5e4,
                       type=float,
                       help="Limits the RAM during vocabulary building. Every 10 million word types need about 1GB of RAM.")

    group.add_argument('--word_embedding_model_name',
                       type=str,
                       help="Model name for own trained word embedding model.")

    group.add_argument('--google_vec_file',
                       type=str,
                       help="Google word2vec pretrained word embedding file.")

    group.add_argument('--google_vec_dim',
                       type=int,
                       help="Google word2vec pretrained word embedding dim.")

    group.add_argument('--fasttext_vec_file',
                       type=str,
                       help="fasttext pretrained word embedding file.")

    group.add_argument('--fasttext_vec_dim',
                       type=int,
                       help="fasttext pretrained word embedding dim.")

    group.add_argument('--binary', action='store_true', help='is binary format.')

    group.add_argument('--vocab_path',
                       type=str,
                       help='location of save vocab dict file. ')

    group.add_argument('--model_name',
                       type=str,
                       help='model name, seq2seq or Knowledge_grounded')
