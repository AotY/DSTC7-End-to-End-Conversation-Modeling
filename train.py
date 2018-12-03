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
import torch.nn.functional as F

from modules.optim import ScheduledOptimizer
from modules.early_stopping import EarlyStopping

from gensim.models import KeyedVectors

from misc.vocab import Vocab
from cm import ConversationModel
from misc.dataset import Dataset
from train_opt import data_set_opt, model_opt, train_opt

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("Running %s", ' '.join(sys.argv))

# get optional parameters
parser = argparse.ArgumentParser(description=program,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
data_set_opt(parser)
model_opt(parser)
train_opt(parser)
opt = parser.parse_args()

# logger file
time_str = time.strftime('%Y_%m_%d_%H:%M')
opt.log_path = opt.log_path.format(
    opt.model_type, time_str, opt.turn_num)
logger.info('log_path: {}'.format(opt.log_path))

if opt.seed:
    torch.manual_seed(opt.seed)

device = torch.device(opt.device)
logging.info("device: %s" % device)

logger.info('c_max_len: %d' % opt.c_max_len)
logger.info('r_max_len: %d' % opt.r_max_len)


def train_epochs(model,
                 dataset,
                 optimizer,
                 criterion,
                 vocab,
                 early_stopping):

    start = time.time()
    max_load = int(np.ceil(dataset._size_dict['train'] / opt.batch_size))
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        dataset.reset_data('train')
        log_loss_total = 0
        log_accuracy_total = 0

        for load in range(1, max_load + 1):
            # load data
            dec_inputs, dec_targets, dec_inputs_length, dec_inputs_pos, \
                context_texts, response_texts, conversation_ids, \
                f_inputs, f_inputs_length, f_topks_length, facts_texts, \
                h_inputs, h_turns_length, h_inputs_length, h_inputs_pos = dataset.load_data(
                    'train', opt.batch_size)

            # train and get cur loss
            loss, accuracy = train(model,
                                   h_inputs,
                                   h_inputs_length,
                                   h_turns_length,
                                   h_inputs_pos,
                                   f_inputs,
                                   f_inputs_length,
                                   f_topks_length,
                                   dec_inputs,
                                   dec_targets,
                                   dec_inputs_length,
                                   dec_inputs_pos,
                                   optimizer,
                                   criterion)

            log_loss_total += float(loss)
            log_accuracy_total += accuracy
            if load % opt.log_interval == 0:
                log_loss_avg = log_loss_total / opt.log_interval
                log_accuracy_avg = log_accuracy_total / opt.log_interval
                logger_str = '\ntrain --> epoch: %d %s (%d %d%%) loss: %.4f acc: %.4f ppl: %.4f' \
                    % (epoch, timeSince(start, load / max_load),
                       load, load / max_load * 100, log_loss_avg,
                       log_accuracy_avg, math.exp(log_loss_avg))

                logger.info(logger_str)
                save_logger(logger_str)
                log_loss_total = 0
                log_accuracy_total = 0

        # save model of each epoch
        save_state = {
            'loss': log_loss_avg,
            'ppl': math.exp(log_loss_avg),
            'acc': log_accuracy_avg,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.optimizer.state_dict(),
            'opt': opt
        }

        # save checkpoint, including epoch, seq2seq_mode.state_dict() and
        save_checkpoint(state=save_state,
                        is_best=False,
                        filename=os.path.join(opt.model_path, 'epoch-%d_%s_%d_%s.pth' %
                                              (epoch, opt.model_type, opt.turn_num, time_str)))

        # evaluate
        evaluate_loss, evaluate_accuracy = evaluate(model=model,
                                                    dataset=dataset,
                                                    criterion=criterion)

        logger_str = '\nevaluate --> loss: %.4f acc: %.4f ppl: %.4f' % (
            evaluate_loss, evaluate_accuracy, math.exp(evaluate_loss))
        logger.info(logger_str)
        save_logger(logger_str)

        # generate sentence
        #  logger.info('generate...')
        #  decode(model, dataset)

        is_stop = early_stopping.step(evaluate_loss)
        if is_stop:
            logger.info('Early Stopping.')
            sys.exit(0)


''' start traing '''


def train(model,
          h_inputs,
          h_inputs_length,
          h_turns_length,
          h_inputs_pos,
          f_inputs,
          f_inputs_length,
          f_topks_length,
          dec_inputs,
          dec_targets,
          dec_inputs_length,
          dec_inputs_pos,
          optimizer,
          criterion):

    # Turn on training mode which enables dropout.
    model.train()

    # [batch_size, max_len, vocab_size]
    decoder_outputs = model(
        h_inputs,
        h_inputs_length,
        h_turns_length,
        h_inputs_pos,
        f_inputs,
        f_inputs_length,
        f_topks_length,
        dec_inputs,
        dec_inputs_length,
        dec_inputs_pos
    )

    optimizer.zero_grad()

    loss = 0

    # decoder_outputs -> [max_length, batch_size, vocab_sizes]
    decoder_outputs_argmax = torch.argmax(decoder_outputs, dim=2)
    #  accuracy = compute_accuracy(decoder_outputs_argmax, dec_targets)

    # reshape to [max_seq * batch_size, decoder_vocab_size]
    decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])

    dec_targets = dec_targets.view(-1)

    # compute loss
    loss = criterion(decoder_outputs, dec_targets)
    #  print(loss)

    # backward
    loss.backward()

    #  for p, n in zip(model.simple_encoder.rnn.parameters(), model.simple_encoder.rnn._all_weights[0]):
    #  if n[:6] == 'weight':
    #  print('simple: ===========\ngradient:{}\n----------\n{}'.format(n,p.grad))

    #  for p, n in zip(model.self_attn_encoder.rnn.parameters(), model.self_attn_encoder.rnn._all_weights[0]):
    #  if n[:6] == 'weight':
    #  print('self_attn: ===========\ngradient:{}\n----------\n{}'.format(n,p.grad))

    # optimizer
    optimizer.step()
    #  _ = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
    #  optimizer.step()

    return loss.item(), accuracy


'''
evaluate model.
'''


def evaluate(model,
             dataset,
             criterion):

    # Turn on evaluation mode which disables dropout.
    model.eval()
    loss_total = 0
    accuracy_total = 0
    max_load = int(np.floor(dataset._size_dict['test'] / opt.batch_size))
    dataset.reset_data('test', False)
    with torch.no_grad():
        for load in range(1, max_load + 1):
            # load data
            dec_inputs, dec_targets, dec_inputs_length, dec_inputs_pos, \
                context_texts, response_texts, conversation_ids, \
                f_inputs, f_inputs_length, f_topks_length, facts_texts, \
                h_inputs, h_turns_length, h_inputs_length, h_inputs_pos = dataset.load_data(
                    'test', opt.batch_size)

            # train and get cur loss
            decoder_outputs = model(
                h_inputs,
                h_inputs_length,
                h_turns_length,
                h_inputs_pos,
                f_inputs,
                f_inputs_length,
                f_topks_length,
                dec_inputs,
                dec_inputs_length,
                dec_inputs_pos
            )

            # decoder_outputs -> [max_length, batch_size, vocab_sizes]
            decoder_outputs_argmax = torch.argmax(decoder_outputs, dim=2)
            #  accuracy = compute_accuracy(
                #  decoder_outputs_argmax, dec_targets)

            #  Compute loss
            decoder_outputs = decoder_outputs.view(-1,
                                                   decoder_outputs.shape[-1])
            dec_targets = dec_targets.view(-1)

            loss = criterion(decoder_outputs, dec_targets)

            loss_total += loss.item()
            accuracy_total += accuracy

    return loss_total / max_load, accuracy_total / max_load


def decode(model, dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    dataset.reset_data('eval', False)
    max_load = int(np.floor(dataset._size_dict['eval'] / opt.batch_size))
    with torch.no_grad():
        for load in range(1, max_load + 1):
            dec_inputs, dec_targets, dec_inputs_length, dec_inputs_pos, \
                context_texts, response_texts, conversation_ids, \
                f_inputs, f_inputs_length, f_topks_length, facts_texts, \
                h_inputs, h_turns_length, h_inputs_length, h_inputs_pos = dataset.load_data(
                    'eval', opt.batch_size)

            # greedy: [batch_size, r_max_len]
            # beam_search: [batch_sizes, best_n, len]
            beam_outputs, beam_score = model.decode(
                h_inputs,
                h_inputs_length,
                h_turns_length,
                h_inputs_pos,
                f_inputs,
                f_inputs_length,
                f_topks_length
            )

            beam_texts = dataset.generating_texts(beam_outputs,
                                                  decode_type='beam_search')

            # save sentences
            dataset.save_generated_texts(context_texts,
                                         response_texts,
                                         conversation_ids,
                                         beam_texts,
                                         os.path.join(opt.save_path, 'generated/transformer_%s_%s_%d_%s.txt' % (
                                             opt.model_type, opt.decode_type, opt.turn_num, time_str)),
                                         opt.decode_type,
                                         facts_texts)


''' save log to file '''


def save_logger(logger_str):
    with open(opt.log_path, 'a', encoding='utf-8') as f:
        f.write(logger_str)


def build_optimizer(model):
    optimizer = ScheduledOptimizer(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.transformer_size,
        opt.n_warmup_steps,
        opt.max_grad_norm
    )

    return optimizer


def build_criterion(padid):
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    criterion = nn.NLLLoss(reduction='elementwise_mean', ignore_index=padid)
    #  criterion = nn.NLLLoss(reduction='sum', ignore_index=padid)

    return criterion


def build_model(vocab):
    logger.info('Building model...')

    model = ConversationModel(
        opt,
        device,
    )

    model = model.to(device)

    print(model)
    return model


def build_dataset(vocab):
    dataset = Dataset(
        opt,
        vocab,
        device,
        logger)

    return dataset


def save_checkpoint(state, is_best, filename):
    '''
    Saving a model in pytorch.
    :param state: is a dict object, including epoch, optimizer, model etc.
    :param is_best: whether is the best model.
    :param filename: save filename.
    :return:
    '''
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best_%s.pth' % opt.model_type)


def load_fasttext_model(vec_file):
    fasttext = KeyedVectors.load_word2vec_format(vec_file, binary=True)
    return fasttext


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint


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
    # Load checkpoint if we resume from a previous training.
    if opt.checkpoint:
        logger.info('Loading checkpoint from: %s' % opt.checkpoint)
        checkpoint = load_checkpoint(filename=opt.checkpoint)
    else:
        checkpoint = None

    vocab = Vocab()
    vocab.load(opt.vocab_path)
    vocab_size = vocab.size
    opt.vocab_size = int(vocab_size)
    logger.info("vocab_size: %d" % opt.vocab_size)

    dataset = build_dataset(vocab)

    if opt.model_type == 'kg':
        """ computing similarity between conversation and fact """
        offline_filename = os.path.join(
            opt.save_path, 'facts_topk_phrases.%s.pkl' % opt.offline_type)

        if opt.offline_type in ['elastic', 'elastic_tag']:
            dataset.build_similarity_facts_offline(
                offline_filename=offline_filename,
            )
        elif opt.offline_type == 'elmo':
            pass

    model = build_model(vocab)

    # Build optimizer.
    optimizer = build_optimizer(model)

    criterion = build_criterion(vocab.padid)

    early_stopping = EarlyStopping(
        type='min',
        min_delta=0.001,
        patience=3
    )

    '''if load checkpoint'''
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        opt.start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        ppl = checkpoint['ppl']
        acc = checkpoint['acc']
        opt = checkpoint['opt']
        logger_str = '\nevaluate --> loss: %.4f acc: %.4f ppl: %.4f' % (
            loss, acc, ppl)
        logger.info(logger_str)

    if opt.task == 'train':
        train_epochs(model=model,
                     dataset=dataset,
                     optimizer=optimizer,
                     criterion=criterion,
                     vocab=vocab,
                     early_stopping=early_stopping)
    elif opt.task == 'eval':
        evaluate(model,
                 dataset,
                 criterion)
    elif opt.task == 'decode':
        decode(model, dataset, vocab)
    else:
        raise ValueError(
            "task must be train or eval, no %s " % opt.task)
