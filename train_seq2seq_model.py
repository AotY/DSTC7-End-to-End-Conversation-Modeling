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
opt.log_file = opt.log_file.format(time_str)
logger.info('log_file: {}'.format(opt.log_file))

device = torch.device(opt.device)
logging.info("device: %s" % device)

logging.info("teacher_forcing_ratio: %f" % opt.teacher_forcing_ratio)

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
        logger.info('------------------- epoch: %d ----------------------------' % epoch)
        dataset.reset_data('train')
        log_loss_total = 0  # Reset every logger.info_every
        log_accuracy_total = 0
        for load in range(1, max_load + 1):
            #  logger_str = '\n*********************** Epoch %i/%i - load %.2f perc **********************' % (
                #  epoch, opt.epochs, 100 * load / max_load)
            #  logger.info(logger_str)

            # load data
            dialogue_encoder_inputs, dialogue_encoder_inputs_length, \
                dialogue_decoder_inputs, dialogue_decoder_targets, \
                conversation_texts, response_texts = dataset.load_data(
                    'train', opt.batch_size)

            # train and get cur loss
            loss, accuracy = train(model,
                         dialogue_encoder_inputs,
                         dialogue_encoder_inputs_length,
                         dialogue_decoder_inputs,
                         dialogue_decoder_targets,
                         optimizer,
                         criterion,
                         vocab,
                         opt)

            log_loss_total += float(loss)
            log_accuracy_total += accuracy
            if load % opt.log_interval == 0:
                log_loss_avg = log_loss_total / opt.log_interval
                log_accuracy_avg = log_accuracy_total / opt.log_interval
                logger_str = '\ntrain -------------------------------> %s (%d %d%%) %.4f %.4f' % (timeSince(start, load / max_load),
                                                                                             load, load / max_load * 100,
                                                                                             log_loss_avg, log_accuracy_avg)
                logger.info(logger_str)
                save_logger(logger_str)
                log_loss_total = 0
                log_accuracy_total = 0

        # evaluate
        evaluate_loss, evaluate_accuracy = evaluate(model=model,
                                                    dataset=dataset,
                                                    criterion=criterion,
                                                    opt=opt)

        logger_str = '\nevaluate ---------------------------------> %.4f %.4f' % (evaluate_loss, evaluate_accuracy)
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
          dialogue_encoder_inputs,
          dialogue_encoder_inputs_length,
          dialogue_decoder_inputs,
          dialogue_decoder_targets,
          optimizer,
          criterion,
          vocab,
          opt):

    # Turn on training mode which enables dropout.
    model.train()

    (dialogue_encoder_final_state, dialogue_encoder_memory_bank), \
        (dialogue_decoder_final_state, dialogue_decoder_outputs, dialogue_decoder_attns) \
        = model(
        dialogue_encoder_inputs=dialogue_encoder_inputs,
        dialogue_encoder_inputs_length=dialogue_encoder_inputs_length,
        dialogue_decoder_inputs=dialogue_decoder_inputs,
        teacher_forcing_ratio=opt.teacher_forcing_ratio,
        batch_size=opt.batch_size)

    optimizer.zero_grad()

    loss = 0

    # dialogue_decoder_outputs -> [max_length, batch_size, vocab_sizes]
    dialogue_decoder_outputs_argmax = torch.argmax(dialogue_decoder_outputs, dim=2)
    accuracy = compute_accuracy(dialogue_decoder_outputs_argmax, dialogue_decoder_targets)

    # reshape to [max_seq * batch_size, decoder_vocab_size]
    dialogue_decoder_outputs = dialogue_decoder_outputs.view(-1, dialogue_decoder_outputs.shape[-1])

    # , dialogue_decoder_targets.shape[1])
    dialogue_decoder_targets = dialogue_decoder_targets.view(-1)

    # compute loss
    loss = criterion(dialogue_decoder_outputs, dialogue_decoder_targets)

    # backward
    loss.backward()

    # optimizer
    optimizer.step()

    return loss.item(), accuracy


def compute_accuracy(dialogue_decoder_outputs_argmax, dialogue_decoder_targets):
    """
    dialogue_decoder_targets: [seq_len, batch_size]
    """
    #  print('---------------------->\n')
    #  print(dialogue_decoder_outputs_argmax)
    #  print(dialogue_decoder_targets)

    match_tensor = (dialogue_decoder_outputs_argmax == dialogue_decoder_targets).long()
    dialogue_decoder_mask = (dialogue_decoder_targets != 0).long()

    accuracy_tensor = match_tensor * dialogue_decoder_mask

    accuracy = float(torch.sum(accuracy_tensor)) / float(torch.sum(dialogue_decoder_mask))

    return accuracy



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
            dialogue_encoder_inputs, dialogue_encoder_inputs_length, \
                dialogue_decoder_inputs, dialogue_decoder_targets, \
                conversation_texts, response_texts = dataset.load_data(
                    'eval', opt.batch_size)

            # train and get cur loss
            (dialogue_encoder_final_state, dialogue_encoder_memory_bank), \
                (dialogue_decoder_final_state, dialogue_decoder_outputs,
                 dialogue_decoder_attns) = model.evaluate(
                dialogue_encoder_inputs=dialogue_encoder_inputs,  # LongTensor
                dialogue_encoder_inputs_length=dialogue_encoder_inputs_length,
                batch_size=opt.batch_size)

            # dialogue_decoder_outputs -> [max_length, batch_size, vocab_sizes]
            dialogue_decoder_outputs_argmax = torch.argmax(dialogue_decoder_outputs, dim=2)
            accuracy = compute_accuracy(dialogue_decoder_outputs_argmax, dialogue_decoder_targets)

            #  Compute loss
            dialogue_decoder_outputs = dialogue_decoder_outputs.view(
                -1, dialogue_decoder_outputs.shape[-1])
            dialogue_decoder_targets = dialogue_decoder_targets.view(-1)

            loss = criterion(dialogue_decoder_outputs, dialogue_decoder_targets)

            loss_total += loss.item()
            accuracy_total += accuracy

    return loss_total / max_load, accuracy_total / max_load


def generate(model, dataset, opt):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    loss_total = 0
    max_load = int(np.ceil(dataset.n_eval / opt.batch_size))
    dataset.reset_data('eval')
    with torch.no_grad():
        for load in range(1, max_load + 1):
            # load data
            dialogue_encoder_inputs, dialogue_encoder_inputs_length, \
                dialogue_decoder_inputs, dialogue_decoder_targets, \
                conversation_texts, response_texts = dataset.load_data(
                    'eval', opt.batch_size)

            # train and get cur loss
            # greedy: [batch_size, max_len]
            # beam_search: [batch_sizes, topk, len]
            batch_utterances = model.generate(
                                dialogue_encoder_inputs,  # LongTensor
                                dialogue_encoder_inputs_length,
                                opt.batch_size,
                                opt.beam_width,
                                opt.topk)

            # generate sentence, and save to file
            # [max_length, batch_size]
            batch_texts = dataset.generating_texts(batch_utterances,
                                                       opt.batch_size,
                                                       opt.dialogue_decode_type)

            # save sentences
            dataset.save_generated_texts(conversation_texts,
                                         response_texts,
                                         batch_texts,
                                         os.path.join(opt.save_path, 'seq2seq_generated_texts_%s_%s.txt' % (opt.dialogue_decode_type, time_str)),
                                         opt.dialogue_decode_type)

    return loss_total / max_load



def build_optim(model, opt):
    logger.info('Make optimizer for training.')
    optim = Optim(
        opt.optim_method,
        opt.lr,
        opt.max_norm,
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

def build_criterion(padid):
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    criterion = nn.NLLLoss(
        ignore_index=padid,
        reduction='elementwise_mean')

    return criterion


def build_model(opt, dialogue_encoder_vocab, dialogue_decoder_vocab):
    logger.info('Building model...')

    ''' embedding for encoder and decoder '''
    dialogue_encoder_embedding = Embedding(embedding_size=opt.dialogue_encoder_embedding_size,
                                         vocab_size=dialogue_encoder_vocab.get_vocab_size(),
                                         padding_idx=dialogue_encoder_vocab.padid,
                                         dropout_ratio=opt.dialogue_encoder_dropout_probability)

    dialogue_decoder_embedding = Embedding(embedding_size=opt.dialogue_decoder_embedding_size,
                                         vocab_size=dialogue_decoder_vocab.get_vocab_size(),
                                         padding_idx=dialogue_decoder_vocab.padid,
                                         dropout_ratio=opt.dialogue_decoder_dropout_probability)

    ''' load pretrained_weight'''
    if opt.dialogue_encoder_pretrained_embedding_path:

        # load pre-trained embedding
        logger.info("Load pre-trained word embeddig: %s ." %
                    opt.dialogue_decoder_pretrained_embedding_path)

        dialogue_encoder_pretrained_embedding_weight = np.load(
            opt.dialogue_decoder_pretrained_embedding_path)
        dialogue_decoder_pretrained_embedding_weight = dialogue_encoder_pretrained_embedding_weight

        # pretrained_weight is a numpy matrix of shape (num_embedding, embedding_dim)
        dialogue_encoder_embedding.set_pretrained_embedding(
            dialogue_encoder_pretrained_embedding_weight, fixed=False)

        dialogue_decoder_embedding.set_pretrained_embedding(
            dialogue_decoder_pretrained_embedding_weight, fixed=False)

    model = Seq2SeqModel(
        dialogue_encoder_embedding_size=opt.dialogue_encoder_embedding_size,
        dialogue_encoder_vocab_size=dialogue_encoder_vocab.get_vocab_size(),
        dialogue_encoder_hidden_size=opt.dialogue_encoder_hidden_size,
        dialogue_encoder_num_layers=opt.dialogue_encoder_num_layers,
        dialogue_encoder_rnn_type=opt.dialogue_encoder_rnn_type,
        dialogue_encoder_dropout_probability=opt.dialogue_encoder_dropout_probability,
        dialogue_encoder_max_length=opt.dialogue_encoder_max_length,
        dialogue_encoder_bidirectional=opt.dialogue_encoder_bidirectional,
        dialogue_encoder_embedding=dialogue_encoder_embedding,

        dialogue_decoder_embedding_size=opt.dialogue_decoder_embedding_size,
        dialogue_decoder_vocab_size=dialogue_decoder_vocab.get_vocab_size(),
        dialogue_decoder_hidden_size=opt.dialogue_decoder_hidden_size,
        dialogue_decoder_num_layers=opt.dialogue_decoder_num_layers,
        dialogue_decoder_rnn_type=opt.dialogue_decoder_rnn_type,
        dialogue_decoder_dropout_probability=opt.dialogue_decoder_dropout_probability,
        dialogue_decoder_max_length=opt.dialogue_decoder_max_length,
        dialogue_decoder_embedding=dialogue_decoder_embedding,
        dialogue_decoder_pad_id=dialogue_decoder_vocab.padid,
        dialogue_decoder_sos_id=dialogue_decoder_vocab.sosid,
        dialogue_decoder_eos_id=dialogue_decoder_vocab.eosid,
        dialogue_decoder_attention_type=opt.dialogue_decoder_attention_type,
		dialogue_decode_type=opt.dialogue_decode_type,
        dialogue_decoder_tied=opt.dialogue_decoder_tied,
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
        dialogue_encoder_max_length=opt.dialogue_encoder_max_length,
        dialogue_encoder_vocab=vocab,
        dialogue_decoder_max_length=opt.dialogue_encoder_max_length,
        dialogue_decoder_vocab=vocab,
        save_path=opt.save_path,
        dialogue_turn_num=opt.dialogue_turn_num,
        eval_split=opt.eval_split,  # how many hold out as eval data
        device=device,
        logger=logger)

    model = build_model(opt, vocab, vocab)

    # Build optimizer.
    optimizer = build_optim(model, opt)

    criterion = build_criterion(vocab.padid)


    '''if load checkpoint'''
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        opt.start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        logger_str = '\nevaluate ---------------------------------> %.4f' % loss

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
