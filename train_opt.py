# -*- coding:utf--8 -*-


def data_set_opt(parser):
    # Data options
    group = parser.add_argument_group('Data Set Opt.')

    group.add_argument('--pair_path',
                       type=str,
                       help='path of the conversations and responses pair. ')

    group.add_argument('--save_path',
                       type=str,
                       help='location of save files. ')

    group.add_argument('--vocab_path',
                       type=str,
                       help='location of save vocab dict file. ')

    group.add_argument('--vocab_size',
                       default=6e4,
                       type=float,
                       help="Limits the RAM during vocabulary building. Every 10 million word types need about 1GB of RAM.")

    group.add_argument('--min_len',
                       type=int,
                       help="Ignores all words with total frequency lower than this.")

    group.add_argument('--f_max_len',
                       type=int,
                       help='clip fact by max_len.')

    group.add_argument('--q_max_len',
                       type=int,
                       help="tokens after the first max_seq_len tokens will be discarded.")

    group.add_argument('--c_max_len',
                       type=int,
                       help="cur conversation max len.")

    group.add_argument('--r_max_len',
                       type=int,
                       help='response max len.')

    group.add_argument('--turn_num', type=int,
                       default=1,
                       help='input of the model including how many turn dialogue.')

    group.add_argument('--turn_min', type=int,
                       default=1,
                       help='minimal turn num.')

    group.add_argument('--turn_type',
                       type=str,
                       default='concat',
                       help='''how to process conversation historys.
                       sum | concat | sequential | weight | HRAN | transformer

                       How to Make Context More Useful? An Empirical Study on Context-Aware Neural Conversational Models,
                       Hierarchical Recurrent Attention Network for Response Generation,
                       Context-Sensitive Generation of Open-Domain Conversational Responses
                       ''')

    group.add_argument('--eval_split',
                       type=float,
                       help="Ratio for splitting data set.")

    group.add_argument('--test_split',
                       type=float,
                       help="Ratio for splitting data set (test).")

def model_opt(parser):

    group = parser.add_argument_group('model opt.')

    ''' encoder, decoder '''
    group.add_argument('--embedding_size',
                       type=int,
                       default=300,
                       help='embedding size for dialog encoder.')

    group.add_argument('--pre_embedding_size',
                       type=int,
                       default=300,
                       help='pre embedding size for dialog encoder.')

    group.add_argument('--pre_trained_embedding',
                       action='store_true',
                       help='using pre_trained word embedded.')

    group.add_argument('--share_embedding',
                       action='store_true',
                       help='is sharing embedding between encoder and decoder.')

    group.add_argument('--fasttext_vec',
                       type=str,
                       help='1 million word vectors trained with subword infomation on Wikipedia 2017, UMBC webbase corpus and statmt.org news dataset (16B tokens).')

    group.add_argument('--rnn_type',
                       type=str,
                       help='LSTM | GRU')

    group.add_argument('--hidden_size',
                       type=int,
                       default=300,
                       help='number of hidden units per layer')

    group.add_argument('--latent_size',
                       type=int,
                       default=0,
                       help='number of latent units per layer')

    group.add_argument("--kl_anneal",
                       type=str,
                       default='',
                       help="KL Annealing function, select 'logistic' or 'step'.")

    group.add_argument("--kla_denom",
                       type=int,
                       default=10,
                       help="For 'step' KL Annealing: Epoch denominator.")

    group.add_argument("--kla_k",
                       type=float,
                       default=0.00025,
                       help="For 'logistic' KL Annealing: Steepness of Annealing function")

    group.add_argument("--kla_x0",
                       type=int,
                       default=15000,
                       help="For 'logistic' KL Annealing: Midpoint of Annealing function (i.e. weight=0.5)")



    group.add_argument('--num_layers',
                       type=int,
                       help='number of layers')

    group.add_argument('--encoder_num_layers',
                       type=int,
                       help='number of layers')

    group.add_argument('--decoder_num_layers',
                       type=int,
                       help='number of layers')

    group.add_argument('--cnn_kernel_width',
                       type=int,
                       help='Size of the convolving kernel.')

    group.add_argument('--dropout',
                       type=float,
                       default=0.5,
                       help='probability of an element to be zeroed.')

    group.add_argument('--bidirectional',
                       action='store_true',
                       help='is bidirectional.')

    group.add_argument('--tied',
                       action='store_true',
                       help='tie the word embedding and linear weights')

    group.add_argument('--attn_type',
                       type=str,
                       help='dot | general | mlp')

    group.add_argument('--teacher_forcing_ratio',
                       type=float,
                       default=0.5,
                       help='''
                           “Teacher forcing” is the concept of using the real target outputs as each next input, instead of using the decoder’s guess as the next input.
                           Using teacher forcing causes it to converge faster but when the trained network is exploited, it may exhibit instability.
                           Because of the freedom PyTorch’s autograd gives us, we can randomly choose to use teacher forcing or not with a simple if statement.
                           see https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training .
                           ''')

    '''fact parameters'''

    group.add_argument('--f_topk',
                       default=20,
                       type=int,
                       help='select top k by cosine similarity.')

    ''' decode '''
    group.add_argument('--decode_type',
                       type=str,
                       help='greedy | beam_search')

    group.add_argument('--decoder_type',
                       type=str,
                       help='normal | bahdanau | luong')

    group.add_argument('--beam_size',
                       type=int,
                       help='The greater the beam width, the fewer states are pruned. ')

    group.add_argument('--best_n',
                       type=int,
                       help='best partial solutions are kept as candidates.')


def train_opt(parser):
    ''' train parameters '''
    group = parser.add_argument_group('train opt.')

    group.add_argument('--lr', type=float, default=0.001,
                       help='initial learning rate')

    group.add_argument('--n_warmup_steps', type=int, default=3000,
                       help='warm up step.')

    group.add_argument('--max_grad_norm',
                       type=float,
                       default=0.8,
                       help='max norm of the gradients.')

    group.add_argument('--epochs', type=int, default=5,
                       help='upper epoch limit')

    group.add_argument('--start_epoch',
                       type=int,
                       default=0,
                       help='start from a previous epoch.')

    group.add_argument('--batch_size',
                       type=int,
                       default=128,
                       help='batch size')

    group.add_argument('--seed',
                       type=int,
                       default=7,
                       help='random seed')

    group.add_argument('--device',
                       type=str,
                       default='cuda',
                       help='use cuda or cpu.')

    group.add_argument('--eval_interval',
                       type=int,
                       help='eavl interval.')

    group.add_argument('--log_interval',
                       type=int,
                       help='report interval')

    group.add_argument('--model_path',
                       type=str,
                       default='./models',
                       help='path to save models')

    group.add_argument('--log_path',
                       type=str,
                       help='path to save logger.')

    group.add_argument('--checkpoint',
                       type=str,
                       help='for loading checkpoint.')

    group.add_argument('--task',
                       type=str,
                       help='run for training, eval, or generation.')

    group.add_argument('--model_type',
                        type=str,
                        help='seq2seq or kg model.')

    group.add_argument('--label_smoothing',
                       action='store_true',
                       help='loss label_smoothing.')

    group.add_argument('--offline_type',
                        type=str,
                        help='how to retrieval relative facts. elastic | fasttext | elmo.')
