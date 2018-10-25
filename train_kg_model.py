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
import torch.optim as optim

from misc.vocab import Vocab
from kg_model import KGModel
from misc.data_set import Dataset
from train_evaluate_opt import data_set_opt, model_opt, train_opt

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
opt.log_path = opt.log_path.format(opt.model_type, time_str, opt.turn_num, opt.turn_type)
logger.info('log_path: {}'.format(opt.log_path))

device = torch.device(opt.device)
logging.info("device: %s" % device)

logging.info("teacher_forcing_ratio: %f" % opt.teacher_forcing_ratio)

logger.info(opt.max_len)

if opt.seed:
    torch.manual_seed(opt.seed)


def train_epochs(model,
                 data_set,
                 optimizer,
                 criterion,
                 vocab):

    start = time.time()
    max_load = int(np.ceil(data_set.n_train / opt.batch_size))
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        data_set.reset_data('train')
        log_loss_total = 0
        log_accuracy_total = 0
        for load in range(1, max_load + 1):
            # load data
            dialogue_encoder_inputs, dialogue_encoder_inputs_length, \
                dialogue_decoder_inputs, dialogue_decoder_targets, \
                conversation_texts, response_texts, \
                facts_inputs, facts_texts, history_inputs = data_set.load_data('train', opt.batch_size)

            # train and get cur loss
            loss, accuracy = train(model,
                                   history_inputs,
                                   dialogue_encoder_inputs,
                                   dialogue_encoder_inputs_length,
                                   dialogue_decoder_inputs,
                                   dialogue_decoder_targets,
                                   facts_inputs,
                                   optimizer,
                                   criterion,
                                   vocab)

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

        # save model of each epoch
        save_state = {
            'loss': log_loss_avg,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        # save checkpoint, including epoch, seq2seq_mode.state_dict() and
        # optimizer.state_dict()
        save_checkpoint(state=save_state,
                        is_best=False,
                        filename=os.path.join(opt.model_path, 'checkpoint.epoch-%d_%s_%d_%s.pth' % (epoch, opt.model_type, opt.turn_num, opt.turn_type)))

        # evaluate
        evaluate_loss, evaluate_accuracy = evaluate(model=model,
                                                    data_set=data_set,
                                                    criterion=criterion)

        logger_str = '\nevaluate -------------> %.4f %.4f' % (
            evaluate_loss, evaluate_accuracy)
        logger.info(logger_str)
        save_logger(logger_str)

        # generate sentence
        decode(model, data_set, vocab)

''' start traing '''

def train(model,
          history_inputs,
          dialogue_encoder_inputs,
          dialogue_encoder_inputs_length,
          dialogue_decoder_inputs,
          dialogue_decoder_targets,
          facts_inputs,
          optimizer,
          criterion,
          vocab):

    # Turn on training mode which enables dropout.
    model.train()

    # [max_len, batch_size, vocab_size]
    dialogue_decoder_outputs = model(
        history_inputs,
        dialogue_encoder_inputs,
        dialogue_encoder_inputs_length,
        dialogue_decoder_inputs,
        facts_inputs,
        opt.batch_size,
        opt.max_len,
        opt.teacher_forcing_ratio
    )

    optimizer.zero_grad()

    loss=0

    # dialogue_decoder_outputs -> [max_length, batch_size, vocab_sizes]
    dialogue_decoder_outputs_argmax=torch.argmax(dialogue_decoder_outputs, dim=2)
    accuracy=compute_accuracy(dialogue_decoder_outputs_argmax, dialogue_decoder_targets)

    # reshape to [max_seq * batch_size, decoder_vocab_size]
    dialogue_decoder_outputs = dialogue_decoder_outputs.view(-1, dialogue_decoder_outputs.shape[-1])

    # , dialogue_decoder_targets.shape[1])
    dialogue_decoder_targets=dialogue_decoder_targets.view(-1)

    # compute loss
    loss=criterion(dialogue_decoder_outputs, dialogue_decoder_targets)

    # backward
    loss.backward()

    # optimizer
    optimizer.step()

    return loss.item(), accuracy

'''
evaluate model.
'''


def evaluate(model,
             data_set,
             criterion):

    # Turn on evaluation mode which disables dropout.
    model.eval()
    loss_total=0
    accuracy_total=0
    max_load=int(np.ceil(data_set.n_eval / opt.batch_size))
    data_set.reset_data('eval')
    with torch.no_grad():
        for load in range(1, max_load + 1):
            # load data
            dialogue_encoder_inputs, dialogue_encoder_inputs_length, \
                dialogue_decoder_inputs, dialogue_decoder_targets, \
                conversation_texts, response_texts, \
                facts_inputs, facts_texts, history_inputs = data_set.load_data('eval', opt.batch_size)

            # train and get cur loss
            dialogue_decoder_input = torch.ones((1, opt.batch_size), dtype=torch.long, device=device) * vocab.sosid
            dialogue_decoder_outputs=model.evaluate(
                history_inputs,
                dialogue_encoder_inputs,
                dialogue_encoder_inputs_length,
                dialogue_decoder_input,
                facts_inputs,
                opt.max_len,
                opt.batch_size
            )

            # dialogue_decoder_outputs -> [max_length, batch_size, vocab_sizes]
            dialogue_decoder_outputs_argmax=torch.argmax(dialogue_decoder_outputs, dim=2)
            accuracy=compute_accuracy(dialogue_decoder_outputs_argmax, dialogue_decoder_targets)

            #  Compute loss
            dialogue_decoder_outputs=dialogue_decoder_outputs.view(-1, dialogue_decoder_outputs.shape[-1])
            dialogue_decoder_targets=dialogue_decoder_targets.view(-1)

            loss=criterion(dialogue_decoder_outputs,
                             dialogue_decoder_targets)

            loss_total += loss.item()
            accuracy_total += accuracy

    return loss_total / max_load, accuracy_total / max_load


def decode(model, data_set, vocab):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    max_load=int(np.ceil(data_set.n_eval / opt.batch_size))
    data_set.reset_data('eval')
    with torch.no_grad():
        for load in range(1, max_load + 1):
            # load data
            dialogue_encoder_inputs, dialogue_encoder_inputs_length, \
                dialogue_decoder_inputs, dialogue_decoder_targets, \
                conversation_texts, response_texts, \
                facts_inputs, facts_texts, history_inputs = data_set.load_data('eval', opt.batch_size)

            # train and get cur loss
            # greedy: [batch_size, max_len]
            # beam_search: [batch_sizes, best_n, len]
            dialogue_decoder_input = torch.ones((1, opt.batch_size), dtype=torch.long, device=device) * vocab.sosid
            batch_utterances=model.decode(
                history_inputs,
                dialogue_encoder_inputs,  # LongTensor
                dialogue_encoder_inputs_length,
                dialogue_decoder_input,
                facts_inputs,
                opt.decode_type,
                opt.max_len,
                vocab.eosid,
                opt.batch_size,
                opt.beam_width,
                opt.best_n)

            # generate sentence, and save to file
            # [max_length, batch_size]
            batch_texts=data_set.generating_texts(batch_utterances,
                                                   opt.batch_size,
                                                   opt.decode_type)

            # save sentences
            data_set.save_generated_texts(conversation_texts,
                                         response_texts,
                                         batch_texts,
                                         os.path.join(opt.save_path, '%s_generated_texts_%s_%s_%d_%s.txt' % (opt.model_type, opt.decode_type, time_str, opt.turn_num, opt.turn_type)),
                                         opt.decode_type)



def compute_accuracy(dialogue_decoder_outputs_argmax, dialogue_decoder_targets):
    """
    dialogue_decoder_targets: [seq_len, batch_size]
    """
    #  print('---------------------->\n')
    #  print(dialogue_decoder_outputs_argmax)
    #  print(dialogue_decoder_targets)

    match_tensor=(dialogue_decoder_outputs_argmax == dialogue_decoder_targets).long()
    dialogue_decoder_mask=(dialogue_decoder_targets != 0).long()

    accuracy_tensor=match_tensor * dialogue_decoder_mask

    accuracy=float(torch.sum(accuracy_tensor)) / \
        float(torch.sum(dialogue_decoder_mask))

    return accuracy


''' save log to file '''


def save_logger(logger_str):
    with open(opt.log_path, 'a', encoding='utf-8') as f:
        f.write(logger_str)

def build_optim(model):
    optimizer=optim.Adam(model.parameters(),
                           lr=opt.lr,
                           betas=(0.9, 0.999))

    return optimizer


def build_criterion(padid):
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    criterion=nn.NLLLoss(reduction='elementwise_mean',
                           ignore_index=padid)

    return criterion


def build_model(vocab_size, padid):
    logger.info('Building model...')

    model=KGModel(
                opt.model_type,
                vocab_size,
                opt.embedding_size,
                opt.hidden_size,
                opt.num_layers,
                opt.bidirectional,
				opt.turn_num,
				opt.turn_type,
				opt.decoder_type,
                opt.attn_type,
                opt.dropout,
                padid,
                opt.tied,
                device
        )

    model=model.to(device)

    print(model)
    return model


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

def load_pre_trained_embedding(vocab_size, padid, fixed=True):
    ''' embedding for encoder and decoder '''
    embedding = nn.Embedding(vocab_size,
                             opt.embedding_size,
                             padding_idx=padid)

    ''' load pretrained_weight'''
    if opt.pre_trained_embedding:
        pre_trained_embedding = opt.pre_trained_embedding.format(opt.model_type)
        pre_trained_weight = torch.from_numpy(np.load(pre_trained_embedding))
        embedding.weight.data.copy_(pre_trained_weight)
        if fixed:
            embedding.weight.requires_grad = False

    embedding.to(device=device)
    return embedding

def load_checkpoint(filename):
    checkpoint=torch.load(filename)
    return checkpoint

def asMinutes(s):
    m=math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now=time.time()
    s=now - since
    es=s / (percent)
    rs=es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


if __name__ == '__main__':
    # Load checkpoint if we resume from a previous training.
    if opt.checkpoint:
        logger.info('Loading checkpoint from: %s' % opt.checkpoint)
        checkpoint=load_checkpoint(filename=opt.checkpoint)
    else:
        checkpoint=None

    vocab=Vocab()
    vocab.load(opt.vocab_path.format(opt.model_type))
    vocab_size=vocab.get_vocab_size()
    logger.info("vocab_size -----------------> %d" % vocab_size)

    data_set=Dataset(
                    opt.model_type,
                    opt.pair_path,
                    opt.max_len,
                    opt.min_len,
                    opt.fact_max_len,
                    opt.fact_topk,
                    vocab,
                    opt.save_path,
                    opt.turn_num,
                    opt.turn_type,
                    opt.eval_split,  # how many hold out as eval data
                    device,
                    logger)

    if opt.model_type == 'kg':
        """ computing similarity between conversation and fact """
        embedding = load_pre_trained_embedding(vocab_size, vocab.padid)
        filename = os.path.join(opt.save_path, 'topk_facts_embedded.pkl')
        data_set.computing_similarity_facts_offline(opt.embedding_size,
                                                    embedding,
                                                    opt.fact_topk,
                                                    filename)
    model=build_model(vocab_size, vocab.padid)

    # Build optimizer.
    optimizer=build_optim(model)

    criterion=build_criterion(vocab.padid)

    '''if load checkpoint'''
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        opt.start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        logger_str = '\nevaluate ---------------------------------> %.4f' % loss
        logger.info(logger_str)

    if opt.task == 'train':
        train_epochs(model=model,
                     data_set=data_set,
                     optimizer=optimizer,
                     criterion=criterion,
                     vocab=vocab)
    elif opt.task == 'eval':
        evaluate(model,
                data_set,
                criterion)
    elif opt.task == 'decode':
        decode(model, data_set, vocab)
    else:
        raise ValueError(
            "task must be train or eval, no %s " % opt.task)
