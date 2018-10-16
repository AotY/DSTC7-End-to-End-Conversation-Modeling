# -*- coding:utf--8 -*-
from __future__ import division
from __future__ import print_function


# from opennmt

def data_set_opt(parser):
    # Data options
    group = parser.add_argument_group('Data Set Opt.')

    group.add_argument('--path_conversations_responses_pair',
                       type=str,
                       help='location of the conversations and responses pair. ')

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
    
    group.add_argument('--dialogue_turn_num',
                       type=int,
                       default=1,
                       help='input of the model including how many turn dialogue.')

    group.add_argument('--eval_split',
                       default=0.2,
                       type=float,
                       help="Ratio for splitting data set.")


def train_seq2seq_opt(parser):

    group = parser.add_argument_group('Train Seq2seq Model.')

    '''dialog encoder parameters'''
    # group.add_argument('--dialogue_encoder_vocab_size',
    #                    default=8e5 + 4,
    #                    type=float,
    #                    help="Dialog encoder vocab size. Because encoder and decoder can have different vocab")

    group.add_argument('--dialogue_encoder_embedding_size',
                       type=int,
                       default=300,
                       help='embedding size for dialog encoder.')

    group.add_argument('--dialogue_encoder_hidden_size',
                       type=int,
                       default=300,
                       help='number of hidden units per layer')

    group.add_argument('--dialogue_encoder_num_layers',
                       type=int,
                       default=2,
                       help='number of layers')

    group.add_argument('--dialogue_encoder_rnn_type',
                       type=str,
                       default='LSTM',
                       help='type of recurrent net (RNN, LSTM, GRU)')

    group.add_argument('--dialogue_encoder_dropout_probability',
                       type=float,
                       default=0.8,
                       help='size of word embeddings')

    group.add_argument('--dialogue_encoder_max_length',
                       default=50,
                       type=int,
                       help="tokens after the first max_seq_len tokens will be discarded.")

    group.add_argument('--dialogue_encoder_bidirectional',
                       action='store_true',
                       help='is bidirectional.')

    group.add_argument('--dialogue_encoder_pretrained_embedding_path',
                       type=str,
                       help='pre-trained embedding for dialog encoder.')

    '''dialog decoder parameters'''

    group.add_argument('--dialogue_decoder_embedding_size',
                       type=int,
                       default=300,
                       help='embedding size for dialog decoder.')

    group.add_argument('--dialogue_decoder_vocab_size',
                       default=8e5 + 4,
                       type=float,
                       help="Dialog decoder vocab size. Because encoder and decoder can have different vocab")

    group.add_argument('--dialogue_decoder_hidden_size', type=int, default=300,
                       help='number of hidden units per layer')

    group.add_argument('--dialogue_decoder_num_layers',
                       type=int,
                       default=2,
                       help='number of layers')

    group.add_argument('--dialogue_decoder_rnn_type', type=str, default='LSTM',
                       help='type of recurrent net (RNN, LSTM, GRU)')

    group.add_argument('--dialogue_decoder_dropout_probability', type=float, default=0.8,
                       help='size of word embeddings')

    group.add_argument('--dialogue_decoder_max_length',
                       default=50,
                       type=int,
                       help="tokens after the first max_seq_len tokens will be discarded.")

    group.add_argument('--dialogue_decoder_pretrained_embedding_path',
                       type=str,
                       help='pre-trained embedding for dialog decoder.')

    group.add_argument('--dialogue_decoder_attention_type',
                       type=str,
                       default='dot',
                       help='dialog decoder attention type. "dot", "general", or "mlp" ')

    group.add_argument('--dialogue_decode_type',
                       type=str,
                       default='greedy',
                       help='beam search or greedy search.')

    group.add_argument('--beam_width',
                       type=int,
                       default=10,
                       help='beam width for beam search')

    group.add_argument('--topk',
                       type=int,
                       default=2,
                       help='topk sentence for beam search.')

    group.add_argument('--dialogue_decoder_tied',
                       action='store_true',
                       help='tie the word embedding and softmax weights')

    ''' train parameters '''
    group.add_argument('--lr', type=float, default=0.001,
                       help='initial learning rate')

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

    group.add_argument('--teacher_forcing_ratio',
                       type=float,
                       default=0.5,
                       help='''
                           “Teacher forcing” is the concept of using the real target outputs as each next input, instead of using the decoder’s guess as the next input.
                           Using teacher forcing causes it to converge faster but when the trained network is exploited, it may exhibit instability.
                           Because of the freedom PyTorch’s autograd gives us, we can randomly choose to use teacher forcing or not with a simple if statement.
                           see https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training .
                           ''')

    #  group.add_argument('--use_teacher_forcing', action='store_true',
    #  help='is use teacher forcing.')

    group.add_argument('--seed',
                       type=int,
                       default=7,
                       help='random seed')

    group.add_argument('--device',
                       type=str,
                       default='cuda',
                       help='use cuda or cpu.')

    group.add_argument('--log_interval',
                       type=int,
                       default=200,
                       help='report interval')

    group.add_argument('--model_save_path',
                       type=str,
                       default='./models',
                       help='path to save models')

    group.add_argument('--log_file',
                       type=str,
                       help='path to save logger.')

    group.add_argument('--max_norm',
                       type=float,
                       default=100.0,
                       help='max norm of the gradients.')

    group.add_argument('--checkpoint',
                       type=str,
                       help='for loading checkpoint.')

    group.add_argument('--train_or_eval',
                       type=str,
                       help='select train model or eval model')


def train_knowledge_gournded_opt(parser):

    group = parser.add_argument_group('Train KnowledgeGroundedModel Model.')

    '''dialog encoder parameters'''
    group.add_argument('--dialogue_encoder_embedding_size',
                       type=int,
                       default=300,
                       help='embedding size for dialog encoder.')

    group.add_argument('--dialogue_encoder_hidden_size',
                       type=int,
                       default=300,
                       help='number of hidden units per layer')

    group.add_argument('--dialogue_encoder_num_layers',
                       type=int,
                       default=2,
                       help='number of layers')

    group.add_argument('--dialogue_encoder_rnn_type',
                       type=str,
                       default='LSTM',
                       help='type of recurrent net (RNN, LSTM, GRU)')

    group.add_argument('--dialogue_encoder_dropout_probability',
                       type=float,
                       default=0.8,
                       help='size of word embeddings')

    group.add_argument('--dialogue_encoder_max_length',
                       default=50,
                       type=int,
                       help="tokens after the first max_seq_len tokens will be discarded.")

    group.add_argument('--dialogue_encoder_bidirectional',
                       action='store_true',
                       help='is bidirectional.')

    group.add_argument('--dialogue_encoder_pretrained_embedding_path',
                       type=str,
                       help='pre-trained embedding for dialog encoder.')

    '''fact encoder parameters'''

    group.add_argument('--fact_embedding_size',
                       type=int,
                       default=300,
                       help='embedding size for fact embedding.')

    group.add_argument('--fact_vocab_size',
                       default=8e5 + 4,
                       type=float,
                       help="fact vocab size.")

    group.add_argument('--fact_dropout_probability',
                       type=float,
                       default=0.8,
                       help='dropout probability.')

    group.add_argument('--fact_max_length',
                       default=50,
                       type=int,
                       help="tokens after the first max_seq_len tokens will be discarded.")

    group.add_argument('--fact_top_k',
                       default=20,
                       type=int,
                       help='select top k by cosine similarity.')

    '''dialog decoder parameters'''

    group.add_argument('--dialogue_decoder_embedding_size',
                       type=int,
                       default=300,
                       help='embedding size for dialog decoder.')

    group.add_argument('--dialogue_decoder_vocab_size',
                       default=8e5 + 4,
                       type=float,
                       help="Dialog decoder vocab size. Because encoder and decoder can have different vocab")

    group.add_argument('--dialogue_decoder_hidden_size', type=int, default=300,
                       help='number of hidden units per layer')

    group.add_argument('--dialogue_decoder_num_layers',
                       type=int,
                       default=2,
                       help='number of layers')

    group.add_argument('--dialogue_decoder_rnn_type', type=str, default='LSTM',
                       help='type of recurrent net (RNN, LSTM, GRU)')

    group.add_argument('--dialogue_decoder_dropout_probability', type=float, default=0.8,
                       help='size of word embeddings')

    group.add_argument('--dialogue_decoder_max_length',
                       default=50,
                       type=int,
                       help="tokens after the first max_seq_len tokens will be discarded.")

    group.add_argument('--dialogue_decoder_pretrained_embedding_path',
                       type=str,
                       help='pre-trained embedding for dialog decoder.')

    group.add_argument('--dialogue_decoder_attention_type',
                       type=str,
                       default='dot',
                       help='dialog decoder attention type. "dot", "general", or "mlp" ')

    group.add_argument('--dialogue_decoder_tied',
                       action='store_true',
                       help='tie the word embedding and softmax weights'
                       )

    ''' train parameters '''
    group.add_argument('--lr', type=float, default=0.001,
                       help='initial learning rate')

    group.add_argument('--max_norm',
                       type=float,
                       default=100.0,
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

    group.add_argument('--teacher_forcing_ratio',
                       type=float,
                       default=0.5,
                       help='''
                           “Teacher forcing” is the concept of using the real target outputs as each next input, instead of using the decoder’s guess as the next input.
                           Using teacher forcing causes it to converge faster but when the trained network is exploited, it may exhibit instability.
                           Because of the freedom PyTorch’s autograd gives us, we can randomly choose to use teacher forcing or not with a simple if statement.
                           see https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training .
                           ''')

    group.add_argument('--seed',
                       type=int,
                       default=7,
                       help='random seed')

    group.add_argument('--device',
                       type=str,
                       default='cuda',
                       help='use cuda or cpu.')

    group.add_argument('--log_interval',
                       type=int,
                       default=200,
                       help='report interval')

    group.add_argument('--model_save_path',
                       type=str,
                       default='./models',
                       help='path to save models')

    group.add_argument('--log_file',
                       type=str,
                       help='path to save logger.')

    group.add_argument('--checkpoint',
                       type=str,
                       help='for loading checkpoint.')

    group.add_argument('--train_or_eval',
                       type=str,
                       help='select train model or eval model')


