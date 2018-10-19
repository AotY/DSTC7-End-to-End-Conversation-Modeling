# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import logging
import argparse

import numpy as np
import shutil

import torch
import torch.nn as nn
from modules.optim import Optim
from modules.embeddings import Embedding

from misc.vocab import Vocab
from misc.data_set import Seq2seqDataSet
from train_evaluate_opt import data_set_opt, train_seq2seq_opt

from simple_seq2seq_model import Seq2seq

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

# logger file
time_str = time.strftime('%Y-%m-%d_%H:%M')
opt.log_file = opt.log_file.format(time_str)
logger.info('log_file: {}'.format(opt.log_file))

device = torch.device(opt.device)
logging.info("device: %s" % device)

logger.info(opt.dialogue_encoder_max_length)

if opt.seed:
    torch.manual_seed(opt.seed)


def train_epochs(model=None,
                 dataset=None,
                 optimizer=None,
                 criterion=None,
                 vocab=None,
                 opt=None):

    start = time.time()
    max_load = int(np.ceil(dataset.n_train / opt.batch_size))
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        dataset.reset_data('train')
        log_loss_total = 0  # Reset every logger.info_every
        log_accuracy_total = 0
        for load in range(1, max_load + 1):

            # load data
            encoder_inputs, encoder_inputs_length, \
                decoder_inputs, decoder_targets, \
                conversation_texts, response_texts = dataset.load_data(
                    'train', opt.batch_size)

            # train and get cur loss
            loss, accuracy = train(model,
                                   encoder_inputs,
                                   encoder_inputs_length,
                                   decoder_inputs,
                                   decoder_targets,
                                   optimizer,
                                   criterion,
                                   vocab,
                                   opt)

            log_loss_total += float(loss)
            log_accuracy_total += accuracy
            if load % opt.log_interval == 0:
                log_loss_avg = log_loss_total / opt.log_interval
                log_accuracy_avg = log_accuracy_total / opt.log_interval
                logger_str = '\ntrain ------------> epoch: %d %s (%d %d%%) %.4f %.4f' % (epoch, timeSince(start, load / max_load),
                                                                                         load, load / max_load * 100, log_loss_avg,
                                                                                         log_accuracy_avg)
                logger.info(logger_str)
                save_logger(logger_str)
                log_loss_total = 0
                log_accuracy_total = 0

''' start traing '''


def train(model,
          encoder_inputs,
          encoder_inputs_length,
          decoder_inputs,
          decoder_targets,
          optimizer,
          criterion,
          vocab,
          opt):

    # Turn on training mode which enables dropout.
    model.train()

    encoder_outputs, decoder_outputs = model(
        encoder_inputs,
        encoder_inputs_length,
        decoder_inputs[0].view(1, -1),
        decoder_targets,
        opt.batch_size,
        opt.dialogue_encoder_max_length)

    optimizer.zero_grad()

    loss = 0

    # decoder_outputs -> [max_length, batch_size, vocab_sizes]
    decoder_outputs_argmax = torch.argmax(decoder_outputs, dim=2)

    # [max_length, batch_size]
    accuracy = compute_accuracy(decoder_outputs_argmax, decoder_inputs, decoder_targets)

    print('loss dialogue_decoder_outputs shape: {}'.format(decoder_outputs.shape))
    decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])

    decoder_targets = decoder_targets.view(-1)

    loss = criterion(decoder_outputs, decoder_targets)

    # backward
    loss.backward()

    # optimizer
    optimizer.step()

    return loss.item(), accuracy


def compute_accuracy(decoder_outputs_argmax, decoder_inputs, decoder_targets):
    """
    dialogue_decoder_targets: [seq_len, batch_size]
    """
    print('---------------------->\n')
    print(decoder_outputs_argmax)
    print(decoder_inputs)
    print(decoder_targets)

    match_tensor = (decoder_outputs_argmax == decoder_targets).long()
    decoder_mask = (decoder_targets != 0).long()

    accuracy_tensor = match_tensor * decoder_mask

    accuracy = float(torch.sum(accuracy_tensor)) / float(torch.sum(decoder_mask))

    return accuracy


''' save log to file '''


def save_logger(logger_str):
    with open(opt.log_file, 'a', encoding='utf-8') as f:
        f.write(logger_str)

def build_optim(model, opt):
    logger.info('Make optimizer for training.')
    optim = Optim(
        opt.optim_method,
        opt.lr,
        opt.max_norm,
    )

    optim.set_parameters(model.parameters())

    return optim


def build_criterion(padid):
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    #  ignore_index=padid,
    criterion = nn.NLLLoss(reduction='elementwise_mean', ignore_index=padid)
    #  criterion = nn.NLLLoss(reduction='elementwise_mean')

    return criterion


def build_model(opt, vocab):
    logger.info('Building model...')

    model = Seq2seq(
            vocab_size=vocab.get_vocab_size(),
            embedding_size=opt.dialogue_encoder_embedding_size,
            hidden_size=opt.dialogue_encoder_hidden_size,
            dropout_ratio=opt.dialogue_encoder_dropout_probability,
            padding_idx=vocab.padid,
            device=device)

    model = model.to(device)

    print(model)
    return model


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


if __name__ == '__main__':
    vocab = Vocab()
    vocab.load(opt.vocab_save_path)
    vocab_size = vocab.get_vocab_size()
    logger.info("vocab_size -----------------> %d" % vocab_size)

    dataset = Seq2seqDataSet(
        path_conversations_responses_pair=opt.path_conversations_responses_pair,
        dialogue_encoder_max_length=opt.dialogue_encoder_max_length,
        dialogue_encoder_vocab=vocab,
        dialogue_decoder_max_length=opt.dialogue_encoder_max_length,
        dialogue_decoder_vocab=vocab,
        save_path=opt.save_path,
        dialogue_turn_num=opt.dialogue_turn_num,
        eval_split=opt.eval_split,  # how many hold out as eval data
        device=device,
        logger=logger)

    model = build_model(opt, vocab)

    # Build optimizer.
    optimizer = build_optim(model, opt)

    criterion = build_criterion(vocab.padid)

    train_epochs(model=model,
                 dataset=dataset,
                 optimizer=optimizer,
                 criterion=criterion,
                 vocab=vocab,
                 opt=opt)
