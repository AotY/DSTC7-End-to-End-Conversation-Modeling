# -*- coding:utf--8 -*-
from __future__ import division
from __future__ import print_function


# from opennmt


def train_embedding_opt(parser):
    # Data options
    group = parser.add_argument_group('Train word embedding.')

    group.add_argument('--save_path', required=True,
                       help="Output file for the trained word embedding.")

    group.add_argument('--max_words',
                       type=int,
                       default=80,
                       help="""Max words size.
                       """)

    group.add_argument('--size',
                       type=int,
                       default=300,
                       help="""Dimensionality of the feature vectors.
                       """)

    group.add_argument('--window',
                       type=int,
                       default=5,
                       help="""The maximum distance between the current and predicted word within a sentence.
                       """)

    group.add_argument('--alpha',
                       type=float,
                       default=0.025,
                       help="""The initial learning rate.
                       """)

    group.add_argument('--seed',
                       type=int,
                       default=7,
                       help="""Seed for the random number generator.
                       """)

    # group.add_argument('--min_count',
    #                    type=int,
    #                    default=3,
    #                    help="""Ignores all words with total frequency lower than this.
    #                    """)

    # group.add_argument('--max_vocab_size',
    #                    default=None,
    #                    help="""Limits the RAM during vocabulary building; if there are more unique
    #         words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
    #                    """)

    group.add_argument('--sample',
                       type=float,
                       default=1e-3,
                       help="""The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
            """)

    group.add_argument('--hs',
                       type=int,
                       default=0,
                       help="""If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
            """)

    group.add_argument('--negative',
                       type=int,
                       default=5,
                       help="""If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            """)

    group.add_argument('--epochs',
                       type=int,
                       default=5,
                       help="""Number of iterations (epochs) over the corpus.
            """)

    group.add_argument('--lower', action='store_true', help='lowercase data')