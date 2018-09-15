# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import random
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from modules.Optim import Optim

'''
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import seaborn

seaborn.set()
'''

from misc.vocab import Vocab
from train_evaluate_opt import data_set_opt, train_seq2seq_opt
from seq2seq_model import Seq2SeqModel
from misc.data_set import Seq2seqDataSet

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("Running %s", ' '.join(sys.argv))

# get optional parameters
parser = argparse.ArgumentParser(description=program,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
data_set_opt(parser)
train_seq2seq_opt(parser)
opt = parser.parse_args()

device = opt.device
logging.info("device: %s" % device)

use_gpu = False
if device == 'cuda':
    use_gpu = True

if opt.use_teacher_forcing:
    teacher_forcing_ratio = opt.teacher_forcing_ratio
    logging.info("teacher_forcing_ratio: %f" % teacher_forcing_ratio)

torch.manual_seed(opt.seed)


def train_epochs(seq2seq_model=None,
                 seq2seq_dataset=None,
                 batch_size=128,
                 epochs=5,
                 batch_per_load=10,
                 log_interval=200,
                 optimizer=None,
                 criterion=None,
                 vocab=None,
                 opt=None):
    start = time.time()

    log_loss_total = 0  # Reset every logger.info_every

    max_load = np.ceil(seq2seq_dataset.n_train / batch_size / batch_per_load)

    for epoch in range(epochs):
        load = 0
        seq2seq_dataset.reset()
        while not seq2seq_dataset.all_loaded('train'):
            load += 1
            logger.info('\n***** Epoch %i/%i - load %.2f perc *****' % (epoch + 1, epochs, 100 * load / max_load))

            # load data
            num_samples, \
            encoder_input_data, \
            decoder_input_data, \
            decoder_target_data, \
            conversation_texts, response_texts = seq2seq_dataset.load_data('train', batch_size * batch_per_load)

            # train and get cur loss
            loss = train(seq2seq_model,
                         encoder_input_data,
                         decoder_input_data,
                         decoder_target_data,
                         optimizer,
                         criterion,
                         num_samples,
                         vocab,
                         opt)

            log_loss_total += loss

            if load % log_interval == 0:
                log_loss_avg = log_loss_total / log_interval
                log_loss_total = 0
                logger.info('%s (%d %d%%) %.4f' % (timeSince(start, iter / max_load),
                                                   load, load / max_load * 100, log_loss_avg))


def train(seq2seq_model,
          encoder_input_data,
          decoder_input_data,
          decoder_target_data,
          optimizer,
          criterion,
          batch_size,
          vocab,
          opt):
    encoder_input_lengths = torch.Tensor((batch_size,)) * opt.dialog_encoder_max_length
    decoder_input_lengths = torch.Tensor((batch_size,)) * opt.dialog_decoder_max_length
    # decoder_input_lengths = decoder_input_data[0]

    # output, encoder_input_lengths = pack_padded_sequence(sequence=encoder_input_data,
    #                                                      batch_first=False,
    #                                                      padding_value=vocab.padid,
    #                                                      total_length=None)


    (dialog_encoder_final_state, dialog_encoder_memory_bank), \
    (dialog_decoder_memory_bank, dialog_decoder_final_stae, dialog_decoder_attns, dialog_decoder_outputs) \
        = seq2seq_model.forward(
        dialog_encoder_src=encoder_input_data,  # LongTensor
        dialog_encoder_src_lengths=encoder_input_lengths,
        dialog_decoder_tgt=decoder_input_data,
        dialog_decoder_tgt_lengths=decoder_input_lengths,
        use_teacher_forcing=opt.use_teacher_forcing,
        teacher_forcing_ratio=opt.teacher_forcing_ratio,
        batch_size=batch_size)

    optimizer.zero_grad()

    loss = 0

    #  Compute loss
    loss = criterion(dialog_decoder_outputs, decoder_target_data)

    loss.backward()

    optimizer.step()

    return loss.item() / decoder_input_lengths

    # if opt.use_teacher_forcing:
    #     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # else:
    #     use_teacher_forcing = False


def dialog(self, input_text):
    source_seq_int = []
    for token in input_text.strip().strip('\n').split(' '):
        source_seq_int.append(self.dataset.token2index.get(token, self.dataset.UNK))
    return self._infer(np.atleast_2d(source_seq_int))


def build_optim(model, opt, checkpoint=None):
    # if opt.train_from:
    #     logger.info('Loading optimizer from checkpoint.')
    #     optim = checkpoint['optim']
    #     optim.optimizer.load_state_dict(
    #         checkpoint['optim'].optimizer.state_dict())
    # else:
    logger.info('Make optimizer for training.')
    optim = Optim(
        opt.optim_method,
        opt.lr,
        opt.dialog_encoder_clip_grads,
        # lr_decay=opt.learning_rate_decay,
        # start_decay_at=opt.start_decay_at,
        # beta1=opt.adam_beta1,
        # beta2=opt.adam_beta2,
        # adagrad_accum=opt.adagrad_accumulator_init,
        # decay_method=opt.decay_method,
        # warmup_steps=opt.warmup_steps,
        # model_size=opt.rnn_size
    )

    optim.set_parameters(model.parameters())

    return optim


def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    logger.info('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    logger.info('encoder: ', enc)
    logger.info('project: ', dec)


def build_model(opt, dialog_encoder_vocab, dialog_decoder_vocab, checkpoint=None):
    logger.info('Building model...')

    # load pre-trained embedding
    logger.info("Load pre-trained word embeddig: %s ." % opt.dialog_decoder_pretrained_embedding_path)
    dialog_encoder_pretrained_embedding_weight = np.load(opt.dialog_decoder_pretrained_embedding_path)
    dialog_decoder_pretrained_embedding_weight = dialog_encoder_pretrained_embedding_weight

    seq2seq_model = Seq2SeqModel(
        dialog_encoder_vocab_size=dialog_encoder_vocab.get_vocab_size(),
        dialog_encoder_hidden_size=opt.dialog_encoder_hidden_size,
        dialog_encoder_num_layers=opt.dialog_encoder_num_layers,
        dialog_encoder_rnn_type=opt.dialog_encoder_rnn_type,
        dialog_encoder_dropout_rate=opt.dialog_encoder_dropout_rate,
        dialog_encoder_max_length=opt.dialog_encoder_max_length,
        dialog_encoder_clip_grads=opt.dialog_encoder_clip_grads,
        dialog_encoder_bidirectional=opt.dialog_encoder_bidirectional,
        dialog_encoder_pretrained_embedding_weight=dialog_encoder_pretrained_embedding_weight,
        dialog_encoder_pad_id=dialog_encoder_vocab.padid,

        dialog_decoder_vocab_size=dialog_decoder_vocab.get_vocab_size(),
        dialog_decoder_hidden_size=opt.dialog_decoder_hidden_size,
        dialog_decoder_num_layers=opt.dialog_decoder_num_layers,
        dialog_decoder_rnn_type=opt.dialog_decoder_rnn_type,
        dialog_decoder_dropout_rate=opt.dialog_decoder_dropout_rate,
        dialog_decoder_max_length=opt.dialog_decoder_max_length,
        dialog_decoder_clip_grads=opt.dialog_decoder_clip_grads,
        dialog_decoder_bidirectional=opt.dialog_decoder_bidirectional,
        dialog_decoder_pretrained_embedding_weight=dialog_decoder_pretrained_embedding_weight,
        dialog_decoder_pad_id=dialog_decoder_vocab.padid,
        dialog_decoder_attention_type=opt.dialog_decoder_attention_type)

    if use_gpu:
        seq2seq_model.set_cuda()
        # seq2seq_model.cuda()

    return seq2seq_model


def interact(self):
    while True:
        logger.info('----- please input -----')
        input_text = input()
        if not bool(input_text):
            break
        logger.info(self.dialog(input_text))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == '__main__':
    # Load checkpoint if we resume from a previous training.
    # if opt.train_from:
    #     logger.info('Loading checkpoint from %s' % opt.train_from)
    #     checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
    #     model_opt = checkpoint['opt']
    #     # I don't like reassigning attributes of opt: it's not clear.
    #     opt.start_epoch = checkpoint['epoch'] + 1
    # else:
    #     checkpoint = None
    #     model_opt = opt

    vocab = Vocab()
    vocab.load(opt.vocab_save_path)
    vocab_size = vocab.get_vocab_size()

    seq2seq_dataset = Seq2seqDataSet(
        path_conversations=opt.conversations_num_path,
        path_responses=opt.responses_num_path,
        dialog_encoder_vocab_size=vocab_size,
        dialog_encoder_max_length=opt.dialog_encoder_max_length,
        dialog_encoder_vocab=vocab,
        dialog_decoder_vocab_size=vocab_size,
        dialog_decoder_max_length=opt.dialog_encoder_max_length,
        dialog_decoder_vocab=vocab,

        test_split=opt.test_split,  # how many hold out as vali data
        device=device,
        logger=logger
    )

    seq2seq_model = build_model(opt, vocab, vocab, None)
    # tally_parameters(seq2seq_model)
    # check_save_model_path()

    # criterion = nn.CrossEntropyLoss()
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    criterion = nn.NLLLoss()

    # Build optimizer.
    optimizer = build_optim(seq2seq_model, opt, None)

    train_epochs(seq2seq_model=seq2seq_model,
                 seq2seq_dataset=seq2seq_dataset,
                 batch_size=opt.batch_size,
                 epochs=opt.epochs,
                 batch_per_load=10,
                 log_interval=opt.log_interval,
                 optimizer=optimizer,
                 criterion=criterion,
                 vocab=vocab,
                 opt=opt)

    '''
    if opt.mode == 'train':
        train()
    else:
        load_models()

    if opt.mode in ['train', 'continue']:
        s2s.train(batch_size, epochs, lr=learning_rate)
    else:
        if mode == 'eval':
            s2s.build_model_test()
            s2s.evaluate()
        elif mode == 'interact':
            s2s.interact()
            
    '''
