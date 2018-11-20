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
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.optim import ScheduledOptimizer
from modules.early_stopping import EarlyStopping

from gensim.models import KeyedVectors

from misc.vocab import Vocab
#  from kg_model import KGModel
from hred import HRED
from misc.dataset import Dataset
from train_opt import data_set_opt, model_opt, train_opt
from modules.loss import masked_cross_entropy

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
time_str = time.strftime('%Y-%m-%d_%H:%M')
opt.log_path = opt.log_path.format(
    opt.model_type, time_str, opt.turn_num, opt.turn_type)
logger.info('log_path: {}'.format(opt.log_path))

device = torch.device(opt.device)
logging.info("device: %s" % device)

logging.info("teacher_forcing_ratio: %f" % opt.teacher_forcing_ratio)

if opt.seed:
    torch.manual_seed(opt.seed)

# update max_len
#  if opt.turn_type == 'concat':
    #  opt.max_unroll = opt.max_unroll * opt.turn_num

logger.info('max_unroll: %d' % opt.max_unroll)


def train_epochs(model,
                 dataset,
                 optimizer,
                 criterion,
                 vocab,
                 early_stopping):
    start = time.time()
    max_load = int(np.ceil(dataset.n_train / opt.batch_size))
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        dataset.reset_data('train')
        log_loss_total = 0
        log_accuracy_total = 0
        for load in range(1, max_load + 1):
            # load data
            input_sentences, target_sentences, \
                input_sentence_length, target_sentence_length, \
                input_conversation_length, conversation_texts = dataset.load_data(
                    'train', opt.batch_size)

            # train and get cur loss
            loss, accuracy = train(model,
                                   input_sentences,
                                   target_sentences,
                                   input_sentence_length,
                                   target_sentence_length,
                                   input_conversation_length,
                                   optimizer,
                                   criterion,
                                   vocab)

            log_loss_total += float(loss)
            log_accuracy_total += accuracy
            if load % opt.log_interval == 0:
                log_loss_avg = log_loss_total / opt.log_interval
                log_accuracy_avg = log_accuracy_total / opt.log_interval
                logger_str = '\ntrain --> epoch: %d %s (%d %d%%) loss: %.4f acc: %.4f ppl: %.4f' % (epoch, timeSince(start, load / max_load),
                                                                                                    load, load / max_load * 100, log_loss_avg,
                                                                                                    log_accuracy_avg, math.exp(log_loss_avg))
                logger.info(logger_str)
                save_logger(logger_str)
                log_loss_total = 0
                log_accuracy_total = 0

                generate(model, dataset, vocab)

        # save model of each epoch
        save_state = {
            'loss': log_loss_avg,
            'ppl': math.exp(log_loss_avg),
            'acc': log_accuracy_avg,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.optimizer.state_dict()
        }

        # save checkpoint, including epoch, seq2seq_mode.state_dict() and
        save_checkpoint(
            state=save_state,
            is_best=False,
            filename=os.path.join(
                opt.model_path, 'checkpoint.epoch-%d.pth' % epoch)
        )

        # evaluate
        evaluate_loss, evaluate_accuracy = evaluate(model=model,
                                                    dataset=dataset,
                                                    criterion=criterion)

        logger_str = '\nevaluate ---> loss: %.4f acc: %.4f ppl: %.4f' % (
            evaluate_loss, evaluate_accuracy, math.exp(evaluate_loss))
        logger.info(logger_str)
        save_logger(logger_str)

        # generate sentence
        logger.info('generate...')
        generate(model, dataset, vocab)

        is_stop = early_stopping.step(evaluate_loss)
        if is_stop:
            logger.info('Early Stopping.')
            sys.exit(0)


''' start traing '''


def train(model,
          input_sentences,
          target_sentences,
          input_sentence_length,
          target_sentence_length,
          input_conversation_length,
          optimizer,
          criterion,
          vocab):

    # Turn on training mode which enables dropout.
    model.train()

    # [batch_size, max_unroll, vocab_size]
    sentence_logits = model(
        input_sentences,
        input_sentence_length,
        input_conversation_length,
        target_sentences,
        decode=False
    )
    print('sentence_logits: ', sentence_logits.shape)

    optimizer.zero_grad()

    loss = 0

    batch_loss, n_words = masked_cross_entropy(
        sentence_logits,
        target_sentences,
        target_sentence_length
    )
    print('batch_loss: ', batch_loss)

    # decoder_outputs -> [max_length, batch_size, vocab_sizes]
    #  decoder_outputs_argmax = torch.argmax(decoder_outputs, dim=2)
    #  accuracy = compute_accuracy(decoder_outputs_argmax, decoder_targets)
    accuracy = 0.0

    # backward
    batch_loss.backward()

    # optimizer
    optimizer.step_and_update_lr()

    return batch_loss.item(), accuracy


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
    max_load = int(np.ceil(dataset.n_eval / opt.batch_size))
    dataset.reset_data('test')
    with torch.no_grad():
        for load in range(1, max_load + 1):
            # load data
            input_sentences, target_sentences, \
                input_sentence_length, target_sentence_length, \
                input_conversation_length, conversation_texts = dataset.load_data(
                    'test', opt.batch_size)

            sentence_logits = model(
                input_sentences,
                input_sentence_length,
                input_conversation_length,
                target_sentences
            )

            # decoder_outputs -> [max_length, batch_size, vocab_sizes]
            #  decoder_outputs_argmax = torch.argmax(decoder_outputs, dim=2)
            #  accuracy = compute_accuracy(
            #  decoder_outputs_argmax, decoder_targets)
            accuracy = 0.0

            #  Compute loss
            batch_loss, n_words = masked_cross_entropy(
                sentence_logits,
                target_sentences,
                target_sentence_length
            )

            loss_total += loss.item()
            accuracy_total += accuracy

    return loss_total / max_load, accuracy_total / max_load


def generate(model, dataset, vocab):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    dataset.reset_data('eval')
    max_load = int(np.ceil(dataset.n_eval / opt.batch_size))
    with torch.no_grad():
        for load in range(1, max_load + 1):
            input_sentences, target_sentences, \
                input_sentence_length, target_sentence_length, \
                input_conversation_length, conversation_texts = dataset.load_data(
                    'eval', opt.batch_size)

            # [batch_size, max_unroll]
            beam_outputs = model(
                input_sentences,
                input_sentence_length,
                input_conversation_length,
                target_sentences,
                decode=True
            )

            # generate sentence, and save to file
            #  greedy_texts = dataset.generating_texts(greedy_outputs,
            #  opt.batch_size,
            #  'greedy')

            beam_texts = dataset.generating_texts(beam_outputs,
                                                  opt.batch_size,
                                                  'beam_search')
            # save sentences
            dataset.save_generated_texts(
                conversation_texts,
                beam_texts,
                os.path.join(opt.save_path, 'generated/%s_%s.txt' % (
                    opt.model_type, time_str)),
                opt.decode_type
            )


def compute_accuracy(decoder_outputs_argmax, decoder_targets):
    """
    decoder_targets: [batch_size, max_unroll]
    """
    match_tensor = (decoder_outputs_argmax == decoder_targets).long()

    decoder_mask = (decoder_targets != 0).long()

    accuracy_tensor = match_tensor * decoder_mask

    accuracy = float(torch.sum(accuracy_tensor)) / \
        float(torch.sum(decoder_mask))

    return accuracy


''' save log to file '''


def save_logger(logger_str):
    with open(opt.log_path, 'a', encoding='utf-8') as f:
        f.write(logger_str)


def build_optimizer(model):
    #  scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=7, gamma=0.17)
    optimizer = ScheduledOptimizer(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.encoder_hidden_size,
        opt.n_warmup_steps,
        opt.clip
    )

    return optimizer


def build_criterion(padid):
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    #  criterion = nn.NLLLoss(reduction='elementwise_mean', ignore_index=padid)
    #  criterion = nn.NLLLoss(reduction='sum', ignore_index=padid)
    criterion = None

    return criterion


def build_model(vocab_size, padid):
    logger.info('Building model...')

    pre_trained_weight = None
    if opt.pre_trained_embedding and os.path.exists(opt.pre_trained_embedding):
        logger.info('load pre trained embedding...')
        pre_trained_weight = torch.from_numpy(
            np.load(opt.pre_trained_embedding))

    model = HRED(opt).to(device)
    print(model)
    return model


def build_dataset(vocab):
    dataset = Dataset(
        opt,
        vocab,
        device,
        logger
    )

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
    #  vocab.load(opt.vocab_path.format(opt.model_type))
    vocab.load(opt.vocab_path)
    vocab_size = vocab.size
    logger.info("vocab_size -----------------> %d" % vocab_size)

    opt.vocab_size = vocab_size

    dataset = build_dataset(vocab)

    if opt.model_type == 'kg':
        """ computing similarity between conversation and fact """
        #  filename = os.path.join(opt.save_path, 'topk_facts_embedded.%s.pkl' % 'rake')
        filename = os.path.join(opt.save_path, 'topk_facts_p_embedded.pkl')
        fasttext = None
        wiki_dict = None
        if not os.path.exists(filename):
            fasttext = load_fasttext_model(opt.fasttext_vec)
            wiki_dict = pickle.load(open('./data/facts_p_dict.pkl', 'rb'))

        dataset.build_similarity_facts_offline(
            wiki_dict,
            fasttext,
            opt.pre_embedding_size,
            opt.f_topk,
            filename
        )

    model = build_model(vocab_size, vocab.padid)

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
        logger_str = '\nevaluate -----> loss: %.4f acc: %.4f ppl: %.4f' % (
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
    elif opt.task == 'generate':
        generate(model, dataset, vocab)
    else:
        raise ValueError(
            "task must be train or eval, no %s " % opt.task)
