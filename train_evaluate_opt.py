# -*- coding:utf--8 -*-
from __future__ import division
from __future__ import print_function


# from opennmt

def data_set_opt(parser):
    # Data options
    group = parser.add_argument_group('Data Set Opt.')

    group.add_argument('--conversations_num_path',
            type=str,
            help='location of the conversations num. ')

    group.add_argument('--responses_num_path',
            type=str,
            help='location of the responses num. ')

    group.add_argument('--facts_num_path',
            type=str,
            help='location of the facts num. ')

    group.add_argument('--save_path',
            type=str,
            help='location of save files. ')

    group.add_argument('--vocab_save_path',
            type=str,
            help='location of save vocab dict file. ')

    group.add_argument('--min_count',
            type=int,
            default=3,
            help="Ignores all words with total frequency lower than this.")

    group.add_argument('--test_split',
            default=0.2,
            type=float,
            help="Ratio for splitting data set.")


def train_seq2seq_opt(parser):

    group = parser.add_argument_group('Train Seq2seq Model.')

    '''dialog encoder parameters'''
    # group.add_argument('--dialog_encoder_vocab_size',
    #                    default=8e5 + 4,
    #                    type=float,
    #                    help="Dialog encoder vocab size. Because encoder and decoder can have different vocab")

    group.add_argument('--dialog_encoder_embedding_size',
            type=int,
            default=300,
            help='embedding size for dialog encoder.')

    group.add_argument('--dialog_encoder_hidden_size',
            type=int,
            default=300,
            help='number of hidden units per layer')

    group.add_argument('--dialog_encoder_num_layers',
            type=int,
            default=2,
            help='number of layers')

    group.add_argument('--dialog_encoder_rnn_type',
            type=str,
            default='LSTM',
            help='type of recurrent net (RNN, LSTM, GRU)')

    group.add_argument('--dialog_encoder_dropout_rate',
            type=float,
            default=0.8,
            help='size of word embeddings')

    group.add_argument('--dialog_encoder_max_length',
            default=50,
            type=int,
            help="tokens after the first max_seq_len tokens will be discarded.")

    group.add_argument('--dialog_encoder_clipnorm',
            type=float,
            default=1.0,
            help='All parameter gradients will be clipped to a maximum norm of clipnorm.')

    group.add_argument('--dialog_encoder_clipvalue',
            type=float,
            default=0.5,
            help='All parameter gradients will be clipped to a maximum value of clipvalue and a minimum value of -clipvalue')

    group.add_argument('--dialog_encoder_bidirectional', action='store_true',
            help='is bidirection.')

    group.add_argument('--dialog_encoder_pretrained_embedding_path',
            type=str,
            help='pre-trained embedding for dialog encoder.')


    group.add_argument('--dialog_encoder_tied',
            action='store_ture',
            help='tie the word embedding and softmax weights'
            )

    '''dialog decoder parameters'''

    group.add_argument('--dialog_decoder_embedding_size',
                type=int,
            default=300,
            help='embedding size for dialog decoder.')

    group.add_argument('--dialog_decoder_vocab_size',
            default=8e5 + 4,
            type=float,
            help="Dialog decoder vocab size. Because encoder and decoder can have different vocab")

    group.add_argument('--dialog_decoder_hidden_size', type=int, default=300,
            help='number of hidden units per layer')

    group.add_argument('--dialog_decoder_num_layers',
            type=int,
            default=2,
            help='number of layers')

    group.add_argument('--dialog_decoder_rnn_type', type=str, default='LSTM',
            help='type of recurrent net (RNN, LSTM, GRU)')

    group.add_argument('--dialog_decoder_dropout_rate', type=float, default=0.8,
            help='size of word embeddings')

    group.add_argument('--dialog_decoder_max_length',
            default=50,
            type=int,
            help="tokens after the first max_seq_len tokens will be discarded.")

    group.add_argument('--dialog_decoder_clipnorm',
            type=float,
            default=1.0,
            help='All parameter gradients will be clipped to a maximum norm of clipnorm.')

    group.add_argument('--dialog_decoder_clipvalue',
            type=float,
            default=0.5,
            help='All parameter gradients will be clipped to a maximum value of clipvalue and a minimum value of -clipvalue')

    group.add_argument('--dialog_decoder_bidirectional', action='store_true',
            help='is bidirection.')

    group.add_argument('--dialog_decoder_pretrained_embedding_path',
            type=str,
            help='pre-trained embedding for dialog decoder.')

    group.add_argument('--dialog_decoder_attention_type',
            type=str,
            default='dot',
            help='dialog decoder attention type. "dot", "general", or "mlp" ')

    group.add_argument('--dialog_decoder_tied',
            action='store_ture',
            help='tie the word embedding and softmax weights'
            )

    ''' train parameters '''
    group.add_argument('--lr', type=float, default=0.001,
            help='initial learning rate')

    group.add_argument('--epochs', type=int, default=5,
            help='upper epoch limit')

    group.add_argument('--start_epoch',
            type=int,
            default=0,
            help='start from a previous epoch.')

    group.add_argument('--batch_size', type=int, default=128, metavar='N',
            help='batch size')

    group.add_argument('--teacher_forcing_ratio',
            type=float,
            default=0.5,
            help='''
                           “Teacher forcing” is the concept of using the real target outputs as each next input, instead of using the decoder’s guess as the next input. 
                           Using teacher forcing causes it to converge faster but when the trained network is exploited, it may exhibit instability.
                           Because of the freedom PyTorch’s autograd gives us, we can randomly choose to use teacher forcing or not with a simple if statement.
                           see https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training . 
                           ''')

    group.add_argument('--use_teacher_forcing', action='store_true',
            help='is use teacher forcing.')

    group.add_argument('--seed',
            type=int,
            default=7,
            help='random seed')

    group.add_argument('--device',
            type=str,
            default='cuda',
            help='use cuda or cpu.')

    group.add_argument('--log_interval', type=int, default=200, metavar='N',
            help='report interval')

    group.add_argument('--model_save_path',
            type=str,
            default='./models',
            help='path to save models')

    group.add_argument('--optim_method',
            type=str,
            default='adam',
            help='''
                           method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
                           ''')

    group.add_argument('--batch_per_load',
            type=int,
            default=1,
            help='each batch load: batch_size * batch_per_load')

    group.add_argument('--train_from',
            type=str,
            help='loading checkpoint if we resume from a previous training.')


def evaluate_seq2seq_model_opt(parser):
    pass
