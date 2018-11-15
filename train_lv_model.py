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

from modules.optim import Optimizer, ScheduledOptimizer
from modules.early_stopping import EarlyStopping

from gensim.models import KeyedVectors

from misc.vocab import Vocab
from lv import LV
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
if opt.turn_type == 'concat':
    opt.c_max_len = opt.c_max_len + opt.turn_num

logger.info('LV model')
logger.info('c_max_len: %d' % opt.c_max_len)
logger.info('r_max_len: %d' % opt.r_max_len)


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
            decoder_inputs, decoder_targets, decoder_inputs_length, \
                conversation_texts, response_texts, \
                f_embedded_inputs, f_embedded_inputs_length, \
                f_ids_inputs, f_ids_inputs_length, facts_texts, \
                h_inputs, h_turns_length, h_inputs_length, h_inputs_position = dataset.load_data(
                    'train', opt.batch_size)

            # train and get cur loss
            loss, accuracy = train(model,
                                   epoch,
                                   h_inputs,
                                   h_turns_length,
                                   h_inputs_length,
                                   h_inputs_position,
                                   decoder_inputs,
                                   decoder_targets,
                                   decoder_inputs_length,
                                   f_embedded_inputs,
                                   f_embedded_inputs_length,
                                   f_ids_inputs,
                                   f_ids_inputs_length,
                                   optimizer,
                                   criterion,
                                   vocab)

            log_loss_total += float(loss)
            log_accuracy_total += accuracy
            if load % opt.log_interval == 0:
                log_loss_avg = log_loss_total / opt.log_interval
                log_accuracy_avg = log_accuracy_total / opt.log_interval
                logger_str = '\ntrain ---> epoch: %d %s (%d %d%%) loss: %.4f acc: %.4f ppl: %.4f' % (epoch, timeSince(start, load / max_load),
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
            'optimizer': optimizer.optimizer.state_dict()
        }

        # save checkpoint, including epoch, seq2seq_mode.state_dict() and
        save_checkpoint(state=save_state,
                        is_best=False,
                        filename=os.path.join(opt.model_path, 'checkpoint.epoch-%d_%s_%d_%s.pth' % (epoch, opt.model_type, opt.turn_num, opt.turn_type)))

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
        decode(model, dataset, vocab)

        is_stop = early_stopping.step(evaluate_loss)
        if is_stop:
            logger.info('Early Stopping.')
            sys.exit(0)


def train(model,
          epoch,
          h_inputs,
          h_turns_length,
          h_inputs_length,
          h_inputs_position,
          decoder_inputs,
          decoder_targets,
          decoder_inputs_length,
          f_embedded_inputs,
          f_embedded_inputs_length,
          f_ids_inputs,
          f_ids_inputs_length,
          optimizer,
          criterion,
          vocab):

    # Turn on training mode which enables dropout.
    model.train()

    # [max_len, batch_size, vocab_size]
    decoder_outputs, mean, logvar = model(
        h_inputs,
        h_turns_length,
        h_inputs_length,
        h_inputs_position,
        decoder_inputs,
        decoder_targets,
        decoder_inputs_length,
        f_embedded_inputs,
        f_embedded_inputs_length,
        f_ids_inputs,
        f_ids_inputs_length,
        opt.batch_size,
        opt.r_max_len,
        opt.teacher_forcing_ratio
    )

    optimizer.zero_grad()
    #  model.zero_grad()

    loss = 0

    # decoder_outputs -> [max_length, batch_size, vocab_sizes]
    decoder_outputs_argmax = torch.argmax(decoder_outputs, dim=2)
    accuracy = compute_accuracy(decoder_outputs_argmax, decoder_targets)

    # reshape to [max_seq * batch_size, decoder_vocab_size]
    decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])

    # , decoder_targets.shape[1])
    decoder_targets = decoder_targets.view(-1)

    # compute loss
    nll_loss = criterion(decoder_outputs, decoder_targets)

    kl_weighted_loss, kl_weight, kl_loss = kl_loss_fn(
        mean,
        logvar,
        kl_anneal=opt.kl_anneal,
        global_step=epoch,
        epoch=epoch,
        kla_k=opt.kla_k,
        kla_x0=opt.kla_x0,
        denom=opt.kla_denom
    )

    loss = nll_loss + kl_weighted_loss

    # backward
    loss.backward()

    # optimizer
    optimizer.step_and_update_lr()

    return_loss = float(loss.item())
    del loss

    return return_loss, accuracy


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
            decoder_inputs, decoder_targets, decoder_inputs_length, \
                conversation_texts, response_texts, \
                f_embedded_inputs, f_embedded_inputs_length, \
                f_ids_inputs, f_ids_inputs_length, facts_texts, \
                h_inputs, h_turns_length, h_inputs_length, h_inputs_position = dataset.load_data(
                    'test', opt.batch_size)

            # train and get cur loss
            decoder_outputs, _, _ = model(
                h_inputs,
                h_turns_length,
                h_inputs_length,
                h_inputs_position,
                decoder_inputs,
                decoder_targets,
                decoder_inputs_length,
                f_embedded_inputs,
                f_embedded_inputs_length,
                f_ids_inputs,
                f_ids_inputs_length,
                opt.batch_size,
                opt.r_max_len,
            )

            # decoder_outputs -> [max_length, batch_size, vocab_sizes]
            decoder_outputs_argmax = torch.argmax(decoder_outputs, dim=2)
            accuracy = compute_accuracy(
                decoder_outputs_argmax, decoder_targets)

            #  Compute loss
            decoder_outputs = decoder_outputs.view(-1,
                                                   decoder_outputs.shape[-1])
            decoder_targets = decoder_targets.view(-1)

            loss = criterion(decoder_outputs,
                             decoder_targets)

            loss_total += loss.item()
            accuracy_total += accuracy

    return loss_total / max_load, accuracy_total / max_load


def decode(model, dataset, vocab):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    dataset.reset_data('eval')
    max_load = int(np.ceil(dataset.n_eval / opt.batch_size))
    with torch.no_grad():
        for load in range(1, max_load + 1):

            decoder_inputs, decoder_targets, decoder_inputs_length, \
                conversation_texts, response_texts, \
                f_embedded_inputs, f_embedded_inputs_length, \
                f_ids_inputs, f_ids_inputs_length, facts_texts, \
                h_inputs, h_turns_length, h_inputs_length, h_inputs_position = dataset.load_data(
                    'eval', opt.batch_size)

            # greedy: [batch_size, r_max_len]
            # beam_search: [batch_sizes, best_n, len]
            decoder_input = torch.ones(
                (1, opt.batch_size), dtype=torch.long, device=device) * vocab.sosid
            greedy_outputs, beam_outputs = model.decode(
                h_inputs,
                h_turns_length,
                h_inputs_length,
                h_inputs_position,
                decoder_input,
                f_embedded_inputs,
                f_embedded_inputs_length,
                f_ids_inputs,
                f_ids_inputs_length,
                opt.decode_type,
                opt.r_max_len,
                vocab.eosid,
                opt.batch_size,
                opt.beam_width,
                opt.best_n)

            # generate sentence, and save to file
            # [max_length, batch_size]
            greedy_texts = dataset.generating_texts(greedy_outputs,
                                                    opt.batch_size,
                                                    'greedy')

            beam_texts = dataset.generating_texts(beam_outputs,
                                                  opt.batch_size,
                                                  'beam_search')
            #  print(beam_texts)

            # save sentences
            dataset.save_generated_texts(conversation_texts,
                                         response_texts,
                                         greedy_texts,
                                         beam_texts,
                                         os.path.join(opt.save_path, 'generated/lv_%s_%s_%s_%d_%s.txt' % (
                                             opt.model_type, opt.decode_type, opt.turn_type, opt.turn_num, time_str)),
                                         opt.decode_type,
                                         facts_texts)


def kl_anneal_function(**kwargs):
    """ Returns the weight of for calculating the weighted KL Divergence."""
    if kwargs['kl_anneal'] == 'logistic':
        """ https://en.wikipedia.org/wiki/Logistic_function """
        weight = float(1 / (1 + np.exp(-kwargs['kla_k'] * (kwargs['global_step'] - kwargs['kla_x0']))))
    elif kwargs['kl_anneal'] == 'step':
        weight = kwargs['epoch'] / kwargs['denom']

    return weight

def kl_loss_fn(mean, logvar, **kwargs):
    """Calculate the ELBO, consiting of the Negative Log Likelihood and KL Divergence. """

    kl_loss = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())).mean()

    kl_weight = kl_anneal_function(**kwargs)

    kl_weighted_loss = kl_weight * kl_loss

    return kl_weighted_loss, kl_weight, kl_loss


def compute_accuracy(decoder_outputs_argmax, decoder_targets):
    """
    decoder_targets: [seq_len, batch_size]
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
    optimizer = ScheduledOptimizer(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.hidden_size,
        opt.n_warmup_steps,
        opt.max_norm
    )

    return optimizer


def build_criterion(padid):
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    criterion = nn.NLLLoss(reduction='elementwise_mean', ignore_index=padid)
    #  criterion = nn.NLLLoss(reduction='sum', ignore_index=padid)

    return criterion

def build_model(vocab_size, padid):
    logger.info('Building model...')

    pre_trained_weight = None
    if opt.pre_trained_embedding and os.path.exists(opt.pre_trained_embedding):
        logger.info('load pre trained embedding...')
        pre_trained_weight = torch.from_numpy(
            np.load(opt.pre_trained_embedding))

    model = LV(
        opt.model_type,
        vocab_size,
        opt.c_max_len,
        opt.pre_embedding_size,
        opt.embedding_size,
        opt.share_embedding,
        opt.rnn_type,
        opt.hidden_size,
        opt.latent_size,
        opt.num_layers,
        opt.encoder_num_layers,
        opt.decoder_num_layers,
        opt.bidirectional,
        opt.turn_num,
        opt.turn_type,
        opt.decoder_type,
        opt.attn_type,
        opt.dropout,
        padid,
        opt.tied,
        device,
        pre_trained_weight
    )

    model = model.to(device)

    print(model)
    return model


def build_dataset(vocab):
    dataset = Dataset(
        opt.model_type,
        opt.pair_path,
        opt.c_max_len,
        opt.r_max_len,
        opt.min_len,
        opt.f_max_len,
        opt.f_topk,
        vocab,
        opt.save_path,
        opt.turn_num,
        opt.min_turn,
        opt.turn_type,
        opt.eval_split,  # how many hold out as eval data
        opt.test_split,
        opt.batch_size,
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
    vocab_size = vocab.get_vocab_size()
    logger.info("vocab_size -----------------> %d" % vocab_size)

    dataset = build_dataset(vocab)

    if opt.model_type == 'kg':
        """ computing similarity between conversation and fact """
        #  filename = os.path.join(opt.save_path, 'topk_facts_embedded.%s.pkl' % 'rake')
        filename = os.path.join(opt.save_path, 'topk_facts_embedded.pkl')
        fasttext = None
        wiki_dict = None
        if not os.path.exists(filename):
            fasttext = load_fasttext_model(opt.fasttext_vec)
            wiki_dict = pickle.load(open('./data/wiki_dict.pkl', 'rb'))

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
        logger_str = '\nevaluate ---------------> loss: %.4f acc: %.4f ppl: %.4f' % (
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
