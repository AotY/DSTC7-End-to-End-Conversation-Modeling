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

# logger file
time_str = time.strftime('%Y-%m-%d_%H:%M')
opt.log_file = opt.log_file.format(time_str)
logger.info('log_file: {}'.format(opt.log_file))

device = torch.device(opt.device)
logging.info("device: %s" % device)

logging.info("teacher_forcing_ratio: %f" % opt.teacher_forcing_ratio)

torch.manual_seed(opt.seed)

''''''


def train_epochs(seq2seq_model=None,
                 seq2seq_dataset=None,
                 optimizer=None,
                 criterion=None,
                 vocab=None,
                 opt=None):

    start = time.time()

    log_loss_total = 0  # Reset every logger.info_every

    max_load = np.ceil(seq2seq_dataset.n_train /
                       opt.batch_size / opt.batch_per_load)

    for epoch in range(opt.start_epoch, opt.epochs + 1):

        load = 0
        seq2seq_dataset.reset()
        while not seq2seq_dataset.all_loaded('train'):
            load += 1
            logger_str = '\n*********************** Epoch %i/%i - load %.2f perc **********************' % (
                epoch + 1, opt.epochs, 100 * load / max_load)
            logger.info(logger_str)
            #  save_logger(logger_str)

            # load data
            num_samples, dialog_encoder_inputs, \
                dialog_decoder_inputs, dialog_decoder_targets, \
                dialog_encoder_inputs_length, dialog_decoder_inputs_length, \
                conversation_texts, response_texts = seq2seq_dataset.load_data(
                    'train', opt.batch_size * opt.batch_per_load)

            # train and get cur loss
            loss = train(seq2seq_model,
                         dialog_encoder_inputs,
                         dialog_decoder_inputs,
                         dialog_decoder_targets,
                         dialog_encoder_inputs_length,
                         dialog_decoder_inputs_length,
                         optimizer,
                         criterion,
                         num_samples,
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
        evaluate_loss = evaluate(seq2seq_model=seq2seq_model,
                                 seq2seq_dataset=seq2seq_dataset,
                                 criterion=criterion,
                                 opt=opt)

        logger_str = '\nevaluate ---------------------------------> %.4f' % evaluate_loss
        logger.info(logger_str)
        save_logger(logger_str)

        # save model of each epoch
        save_state = {
            'loss': evaluate_loss,
            'epoch': epoch,
            'state_dict': seq2seq_model.state_dict(),
            'optimizer': optimizer.optimizer.state_dict()
        }

        # save checkpoint, including epoch, seq2seq_mode.state_dict() and
        # optimizer.state_dict()
        save_checkpoint(state=save_state,
                        is_best=False,
                        filename=os.path.join(opt.model_save_path, 'checkpoint.epoch-%d.pth' % epoch))


''' start traing '''


def train(seq2seq_model,
          dialog_encoder_inputs,
          dialog_decoder_inputs,
          dialog_decoder_targets,
          dialog_encoder_inputs_length,
          dialog_decoder_inputs_length,
          optimizer,
          criterion,
          num_samples,
          vocab,
          opt):

    # Turn on training mode which enables dropout.
    seq2seq_model.train()

    (dialog_encoder_final_state, dialog_encoder_memory_bank), \
        (dialog_decoder_final_state, dialog_decoder_outputs, dialog_decoder_attns) \
        = seq2seq_model(
        dialog_encoder_inputs=dialog_encoder_inputs,
        dialog_encoder_inputs_length=dialog_encoder_inputs_length,
        dialog_decoder_inputs=dialog_decoder_inputs,
        dialog_decoder_inputs_length=dialog_decoder_inputs_length,
        dialog_decoder_targets=dialog_decoder_targets,
        teacher_forcing_ratio=opt.teacher_forcing_ratio,
        batch_size=num_samples)

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
    #  loss.div(num_samples).backward()
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(seq2seq_model.parameters(), opt.dialog_decoder_clipnorm)

    # optimizer
    optimizer.step()

    #  return batch_loss / torch.sum(dialog_decoder_inputs_length)
    return loss.item()


''' save log to file '''


def save_logger(logger_str):
    with open(opt.log_file, 'a', encoding='utf-8') as f:
        f.write(logger_str)


'''
evaluate model.
'''


def evaluate(seq2seq_model=None,
             seq2seq_dataset=None,
             criterion=None,
             opt=None):

    # Turn on evaluation mode which disables dropout.
    seq2seq_model.eval()

    loss_total = 0
    iter_ = 0
    with torch.no_grad():
        while not seq2seq_dataset.all_loaded('test'):

            iter_ += 1

            # load data
            num_samples, dialog_encoder_inputs, \
                dialog_decoder_inputs, dialog_decoder_targets, \
                dialog_encoder_inputs_length, dialog_decoder_inputs_length, \
                conversation_texts, response_texts = seq2seq_dataset.load_data(
                    'test', opt.batch_size * opt.batch_per_load)

            # train and get cur loss

            (dialog_encoder_final_state, dialog_encoder_memory_bank), \
                (dialog_decoder_final_state, dialog_decoder_outputs,
                 dialog_decoder_attns) = seq2seq_model.evaluate(
                dialog_encoder_inputs=dialog_encoder_inputs,  # LongTensor
                dialog_encoder_inputs_length=dialog_encoder_inputs_length,
                dialog_decoder_inputs=dialog_decoder_inputs,
                dialog_decoder_inputs_length=dialog_decoder_inputs_length,
                batch_size=num_samples)

            #  Compute loss
            dialog_decoder_outputs = dialog_decoder_outputs.view(
                -1, dialog_decoder_outputs.shape[-1])
            dialog_decoder_targets = dialog_decoder_targets.view(-1)

            loss = criterion(dialog_decoder_outputs, dialog_decoder_targets)

            loss_total += loss.item()

            # generate sentence, and save to file
            # dialog_decoder_outputs -> [max_length, batch_size, vocab_sizes]
            dialog_decoder_outputs = torch.argmax(dim=2)
            # [max_length, batch_size]
            generated_texts = seq2seq_dataset.generating_texts(
                dialog_decoder_outputs.detach().cpu())

            # save sentences
            seq2seq_dataset.save_generated_texts(conversation_texts, response_texts, generated_texts,
                                                 os.path.join(opt.save_path, 'generated_texts_{}.txt'.format(time_str)))

    return loss_total / iter_


def dialog(self, input_text):
    source_seq_int = []
    for token in input_text.strip().strip('\n').split(' '):
        source_seq_int.append(
            self.dataset.token2index.get(token, self.dataset.UNK))
    return self._infer(np.atleast_2d(source_seq_int))


def build_optim(model, opt):
    logger.info('Make optimizer for training.')
    optim = Optim(
        opt.optim_method,
        opt.lr,
        opt.dialog_encoder_clipnorm,
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

    ''' embedding for encoder and decoder '''
    dialog_encoder_embedding = Embedding(embedding_size=opt.dialog_encoder_embedding_size,
                                         vocab_size=dialog_encoder_vocab.get_vocab_size(),
                                         padding_idx=dialog_encoder_vocab.padid,
                                         dropout_ratio=opt.dialog_encoder_dropout_rate)

    dialog_decoder_embedding = Embedding(embedding_size=opt.dialog_decoder_embedding_size,
                                         vocab_size=dialog_decoder_vocab.get_vocab_size(),
                                         padding_idx=dialog_decoder_vocab.padid,
                                         dropout_ratio=opt.dialog_decoder_dropout_rate)

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

    seq2seq_model = Seq2SeqModel(
        dialog_encoder_embedding_size=opt.dialog_encoder_embedding_size,
        dialog_encoder_vocab_size=dialog_encoder_vocab.get_vocab_size(),
        dialog_encoder_hidden_size=opt.dialog_encoder_hidden_size,
        dialog_encoder_num_layers=opt.dialog_encoder_num_layers,
        dialog_encoder_rnn_type=opt.dialog_encoder_rnn_type,
        dialog_encoder_dropout_rate=opt.dialog_encoder_dropout_rate,
        dialog_encoder_max_length=opt.dialog_encoder_max_length,
        dialog_encoder_clipnorm=opt.dialog_encoder_clipnorm,
        dialog_encoder_clipvalue=opt.dialog_encoder_clipvalue,
        dialog_encoder_bidirectional=opt.dialog_encoder_bidirectional,
        dialog_encoder_embedding=dialog_encoder_embedding,
        dialog_encoder_pad_id=dialog_encoder_vocab.padid,
        dialog_encoder_tied=opt.dialog_encoder_tied,

        dialog_decoder_embedding_size=opt.dialog_decoder_embedding_size,
        dialog_decoder_vocab_size=dialog_decoder_vocab.get_vocab_size(),
        dialog_decoder_hidden_size=opt.dialog_decoder_hidden_size,
        dialog_decoder_num_layers=opt.dialog_decoder_num_layers,
        dialog_decoder_rnn_type=opt.dialog_decoder_rnn_type,
        dialog_decoder_dropout_rate=opt.dialog_decoder_dropout_rate,
        dialog_decoder_max_length=opt.dialog_decoder_max_length,
        dialog_decoder_clipnorm=opt.dialog_decoder_clipnorm,
        dialog_decoder_clipvalue=opt.dialog_decoder_clipvalue,
        dialog_decoder_bidirectional=opt.dialog_decoder_bidirectional,
        dialog_decoder_embedding=dialog_decoder_embedding,
        dialog_decoder_pad_id=dialog_decoder_vocab.padid,
        dialog_decoder_eos_id=dialog_decoder_vocab.eosid,
        dialog_decoder_attention_type=opt.dialog_decoder_attention_type,
        dialog_decoder_tied=opt.dialog_decoder_tied,
        device=device)

    seq2seq_model = seq2seq_model.to(device)

    print(seq2seq_model)
    return seq2seq_model


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
    logger.info("Loding checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    return checkpoint


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


if __name__ == '__main__':
    # Load checkpoint if we resume from a previous training.

    if opt.checkpoint:
        logger.info('Loading checkpoint from %s' % opt.checkpoint)
        checkpoint = load_checkpoint(filename=opt.checkpoint)
    else:
        checkpoint = None

    vocab = Vocab()
    vocab.load(opt.vocab_save_path)
    vocab_size = vocab.get_vocab_size()
    logger.info("vocab_size -----------------> %d" % vocab_size)

    seq2seq_dataset = Seq2seqDataSet(
        path_conversations_responses_pair=opt.path_conversations_responses_pair,
        dialog_encoder_max_length=opt.dialog_encoder_max_length,
        dialog_encoder_vocab=vocab,
        dialog_decoder_max_length=opt.dialog_encoder_max_length,
        dialog_decoder_vocab=vocab,

        test_split=opt.test_split,  # how many hold out as vali data
        device=device,
        logger=logger
    )

    seq2seq_model = build_model(opt, vocab, vocab, None)

    # Build optimizer.
    optimizer = build_optim(seq2seq_model, opt)

    # criterion = nn.CrossEntropyLoss()
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    criterion = nn.NLLLoss(
        ignore_index=vocab.padid,
        reduction='elementwise_mean'
    )

    '''if load checkpoint'''
    if checkpoint:
        seq2seq_model.load_state_dict(checkpoint['state_dict'])
        optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_or_eval == 'train':
        train_epochs(seq2seq_model=seq2seq_model,
                     seq2seq_dataset=seq2seq_dataset,
                     optimizer=optimizer,
                     criterion=criterion,
                     vocab=vocab,
                     opt=opt)
    elif opt.train_or_eval == 'eval':
        evaluate(
            seq2seq_model=seq2seq_model,
            seq2seq_dataset=seq2seq_dataset,
            criterion=criterion,
            opt=opt)
    else:
        raise ValueError("train_or_eval must be train or eval, no %s " % opt.train_or_eval)
