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
from seq2seq_model import Seq2SeqModel
from misc.data_set import Seq2seqDataSet
from train_evaluate_opt import data_set_opt, train_seq2seq_opt

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
#  time_str = time.strftime('%Y-%m-%d')
opt.log_file = opt.log_file.format(time_str)
logger.info('log_file: {}'.format(opt.log_file))

device = torch.device(opt.device)
logging.info("device: %s" % device)

logging.info("teacher_forcing_ratio: %f" % opt.teacher_forcing_ratio)

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
        for load in range(1, max_load + 1):
            logger_str = '\n*********************** Epoch %i/%i - load %.2f perc **********************' % (
                epoch, opt.epochs, 100 * load / max_load)
            logger.info(logger_str)

            # load data
            dialog_encoder_inputs, dialog_encoder_inputs_length, \
                dialog_decoder_inputs, dialog_decoder_targets, \
                conversation_texts, response_texts = dataset.load_data(
                    'train', opt.batch_size)

            # train and get cur loss
            loss = train(model,
                         dialog_encoder_inputs,
                         dialog_encoder_inputs_length,
                         dialog_decoder_inputs,
                         dialog_decoder_targets,
                         optimizer,
                         criterion,
                         vocab,
                         opt)

            log_loss_total += float(loss)
            if load % opt.log_interval == 0:
                log_loss_avg = log_loss_total / opt.log_interval
                logger_str = '\ntrain -------------------------------> %s (%d %d%%) %.4f' % (timeSince(start, load / max_load),
                                                                                             load, load / max_load * 100,
                                                                                             log_loss_avg)
                logger.info(logger_str)
                save_logger(logger_str)
                log_loss_total = 0

        # evaluate
        evaluate_loss = evaluate(model=model,
                                 dataset=dataset,
                                 criterion=criterion,
                                 opt=opt)

        logger_str = '\nevaluate ---------------------------------> %.4f' % evaluate_loss
        logger.info(logger_str)
        save_logger(logger_str)

        # generate sentence
        generate(model, dataset, opt)

        # save model of each epoch
        save_state = {
            'loss': evaluate_loss,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.optimizer.state_dict()
        }

        # save checkpoint, including epoch, seq2seq_mode.state_dict() and
        # optimizer.state_dict()
        save_checkpoint(state=save_state,
                        is_best=False,
                        filename=os.path.join(opt.model_save_path, 'checkpoint.epoch-%d_seq2seq.pth' % epoch))


''' start traing '''


def train(model,
          dialog_encoder_inputs,
          dialog_encoder_inputs_length,
          dialog_decoder_inputs,
          dialog_decoder_targets,
          optimizer,
          criterion,
          vocab,
          opt):

    # Turn on training mode which enables dropout.
    model.train()

    (dialog_encoder_final_state, dialog_encoder_memory_bank), \
        (dialog_decoder_final_state, dialog_decoder_outputs, dialog_decoder_attns) \
        = model(
        dialog_encoder_inputs=dialog_encoder_inputs,
        dialog_encoder_inputs_length=dialog_encoder_inputs_length,
        dialog_decoder_inputs=dialog_decoder_inputs,
        teacher_forcing_ratio=opt.teacher_forcing_ratio,
        batch_size=opt.batch_size)

    optimizer.zero_grad()

    loss = 0

    # reshape to [max_seq * batch_size, decoder_vocab_size]
    dialog_decoder_outputs = dialog_decoder_outputs.view(
        -1, dialog_decoder_outputs.shape[-1])

    # , dialog_decoder_targets.shape[1])
    dialog_decoder_targets = dialog_decoder_targets.view(-1)

    # compute loss
    loss = criterion(dialog_decoder_outputs, dialog_decoder_targets)

    # backward
    loss.backward()

    # Clip gradients: gradients are modified in place
    #  total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.dialog_decoder_clipnorm)
    #  print('total_norm: {}'.format(total_norm))

    # optimizer
    optimizer.step()

    return loss.item()


''' save log to file '''


def save_logger(logger_str):
    with open(opt.log_file, 'a', encoding='utf-8') as f:
        f.write(logger_str)


'''
evaluate model.
'''


def evaluate(model=None,
             dataset=None,
             criterion=None,
             opt=None):

    # Turn on evaluation mode which disables dropout.
    model.eval()
    loss_total = 0
    max_load = int(np.ceil(dataset.n_eval / opt.batch_size))
    dataset.reset_data('eval')
    with torch.no_grad():
        for load in range(1, max_load + 1):
            # load data
            dialog_encoder_inputs, dialog_encoder_inputs_length, \
                dialog_decoder_inputs, dialog_decoder_targets, \
                conversation_texts, response_texts = dataset.load_data(
                    'eval', opt.batch_size)

            # train and get cur loss
            (dialog_encoder_final_state, dialog_encoder_memory_bank), \
                (dialog_decoder_final_state, dialog_decoder_outputs,
                 dialog_decoder_attns) = model.evaluate(
                dialog_encoder_inputs=dialog_encoder_inputs,  # LongTensor
                dialog_encoder_inputs_length=dialog_encoder_inputs_length,
                batch_size=opt.batch_size)

            # dialog_decoder_outputs -> [max_length, batch_size, vocab_sizes]
            dialog_decoder_outputs_argmax = torch.argmax(
                dialog_decoder_outputs, dim=2)

            #  Compute loss
            dialog_decoder_outputs = dialog_decoder_outputs.view(
                -1, dialog_decoder_outputs.shape[-1])
            dialog_decoder_targets = dialog_decoder_targets.view(-1)

            loss = criterion(dialog_decoder_outputs, dialog_decoder_targets)

            loss_total += loss.item()

    return loss_total / max_load


def generate(model, dataset, opt):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    loss_total = 0
    max_load = int(np.ceil(dataset.n_eval / opt.batch_size))
    dataset.reset_data('eval')
    with torch.no_grad():
        for load in range(1, max_load + 1):
            # load data
            dialog_encoder_inputs, dialog_encoder_inputs_length, \
                dialog_decoder_inputs, dialog_decoder_targets, \
                conversation_texts, response_texts = dataset.load_data(
                    'eval', opt.batch_size)

            # train and get cur loss
            dialog_decoder_outputs_argmax = model.generate(
            dialog_encoder_inputs=dialog_encoder_inputs,  # LongTensor
            dialog_encoder_inputs_length=dialog_encoder_inputs_length,
            batch_size=opt.batch_size)

            # generate sentence, and save to file
            # [max_length, batch_size]
            generated_texts = dataset.generating_texts(
                dialog_decoder_outputs_argmax, opt.batch_size)

            # save sentences
            dataset.save_generated_texts(conversation_texts, response_texts,
                                         generated_texts, os.path.join(opt.save_path, 'seq2seq_generated_texts_{}.txt'.format(time_str)))

    return loss_total / max_load



def build_optim(model, opt):
    logger.info('Make optimizer for training.')
    optim = Optim(
        opt.optim_method,
        opt.lr,
        opt.dialog_encoder_clipnorm,
        # lr_decay=opt.learning_probability_decay,
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


def build_model(opt, dialog_encoder_vocab, dialog_decoder_vocab):
    logger.info('Building model...')

    ''' embedding for encoder and decoder '''
    dialog_encoder_embedding = Embedding(embedding_size=opt.dialog_encoder_embedding_size,
                                         vocab_size=dialog_encoder_vocab.get_vocab_size(),
                                         padding_idx=dialog_encoder_vocab.padid,
                                         dropout_ratio=opt.dialog_encoder_dropout_probability)

    dialog_decoder_embedding = Embedding(embedding_size=opt.dialog_decoder_embedding_size,
                                         vocab_size=dialog_decoder_vocab.get_vocab_size(),
                                         padding_idx=dialog_decoder_vocab.padid,
                                         dropout_ratio=opt.dialog_decoder_dropout_probability)

    ''' load pretrained_weight'''
    if opt.dialog_encoder_pretrained_embedding_path:

        # load pre-trained embedding
        logger.info("Load pre-trained word embeddig: %s ." %
                    opt.dialog_decoder_pretrained_embedding_path)

        dialog_encoder_pretrained_embedding_weight = np.load(
            opt.dialog_decoder_pretrained_embedding_path)
        dialog_decoder_pretrained_embedding_weight = dialog_encoder_pretrained_embedding_weight

        # pretrained_weight is a numpy matrix of shape (num_embedding, embedding_dim)
        dialog_encoder_embedding.set_pretrained_embedding(
            dialog_encoder_pretrained_embedding_weight, fixed=False)

        dialog_decoder_embedding.set_pretrained_embedding(
            dialog_decoder_pretrained_embedding_weight, fixed=False)

    model = Seq2SeqModel(
        dialog_encoder_embedding_size=opt.dialog_encoder_embedding_size,
        dialog_encoder_vocab_size=dialog_encoder_vocab.get_vocab_size(),
        dialog_encoder_hidden_size=opt.dialog_encoder_hidden_size,
        dialog_encoder_num_layers=opt.dialog_encoder_num_layers,
        dialog_encoder_rnn_type=opt.dialog_encoder_rnn_type,
        dialog_encoder_dropout_probability=opt.dialog_encoder_dropout_probability,
        dialog_encoder_max_length=opt.dialog_encoder_max_length,
        dialog_encoder_clipnorm=opt.dialog_encoder_clipnorm,
        dialog_encoder_bidirectional=opt.dialog_encoder_bidirectional,
        dialog_encoder_embedding=dialog_encoder_embedding,

        dialog_decoder_embedding_size=opt.dialog_decoder_embedding_size,
        dialog_decoder_vocab_size=dialog_decoder_vocab.get_vocab_size(),
        dialog_decoder_hidden_size=opt.dialog_decoder_hidden_size,
        dialog_decoder_num_layers=opt.dialog_decoder_num_layers,
        dialog_decoder_rnn_type=opt.dialog_decoder_rnn_type,
        dialog_decoder_dropout_probability=opt.dialog_decoder_dropout_probability,
        dialog_decoder_max_length=opt.dialog_decoder_max_length,
        dialog_decoder_clipnorm=opt.dialog_decoder_clipnorm,
        dialog_decoder_embedding=dialog_decoder_embedding,
        dialog_decoder_pad_id=dialog_decoder_vocab.padid,
        dialog_decoder_sos_id=dialog_decoder_vocab.sosid,
        dialog_decoder_eos_id=dialog_decoder_vocab.eosid,
        dialog_decoder_attention_type=opt.dialog_decoder_attention_type,
		dialog_decoder_type=opt.dialog_decoder_type,
        dialog_decoder_tied=opt.dialog_decoder_tied,
        device=device)

    model = model.to(device)

    print(model)
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    '''
    Saving a model in pytorch.
    :param state: is a dict object, including epoch, optimizer, model etc.
    :param is_best: whether is the best model.
    :param filename: save filename.
    :return:
    '''
    save_path = os.path.join(opt.model_save_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth')


def load_checkpoint(filename='checkpoint.pth'):
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
    vocab.load(opt.vocab_save_path)
    vocab_size = vocab.get_vocab_size()
    logger.info("vocab_size -----------------> %d" % vocab_size)

    dataset = Seq2seqDataSet(
        path_conversations_responses_pair=opt.path_conversations_responses_pair,
        dialog_encoder_max_length=opt.dialog_encoder_max_length,
        dialog_encoder_vocab=vocab,
        dialog_decoder_max_length=opt.dialog_encoder_max_length,
        dialog_decoder_vocab=vocab,
        save_path=opt.save_path,
        eval_split=opt.eval_split,  # how many hold out as eval data
        device=device,
        logger=logger)

    model = build_model(opt, vocab, vocab)

    # Build optimizer.
    optimizer = build_optim(model, opt)

    # criterion = nn.CrossEntropyLoss()
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    criterion = nn.NLLLoss(
        ignore_index=vocab.padid,
        reduction='elementwise_mean')

    '''if load checkpoint'''
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_or_eval == 'train':
        train_epochs(model=model,
                     dataset=dataset,
                     optimizer=optimizer,
                     criterion=criterion,
                     vocab=vocab,
                     opt=opt)
    elif opt.train_or_eval == 'eval':
        evaluate(
            model=model,
            dataset=dataset,
            criterion=criterion,
            opt=opt)
    else:
        raise ValueError(
            "train_or_eval must be train or eval, no %s " % opt.train_or_eval)
