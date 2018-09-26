# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import random
import logging
import argparse

import numpy as np
import shutil

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

''''''
def train_epochs(seq2seq_model=None,
                 seq2seq_dataset=None,
                 optimizer=None,
                 criterion=None,
                 vocab=None,
                 opt=None):

    start = time.time()

    log_loss_total = 0  # Reset every logger.info_every

    max_load = np.ceil(seq2seq_dataset.n_train / opt.batch_size / opt.batch_per_load)

    for epoch in range(opt.start_epoch, opt.epochs):
        load = 0
        seq2seq_dataset.reset()
        while not seq2seq_dataset.all_loaded('train'):
            load += 1
            logger.info('\n***************** Epoch %i/%i - load %.2f perc *****************' % (epoch + 1, opt.epochs, 100 * load / max_load))

            # load data
            num_samples, \
            encoder_input_data, \
            decoder_input_data, \
            decoder_target_data, \
            conversation_texts, response_texts = seq2seq_dataset.load_data('train', opt.batch_size * opt.batch_per_load)

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

            if load % opt.log_interval == 0:
                log_loss_avg = log_loss_total / opt.log_interval
                log_loss_total = 0
                logger.info('log_loss_avg type : {}'.format(type(log_loss_avg)))
                logger.info('train ----------------------> %s (%d %d%%) %.4f' % (timeSince(start, load / max_load),
                                                               load, load / max_load * 100, log_loss_avg))

        evaluate_loss = evaluate(seq2seq_model=seq2seq_model,
                                 seq2seq_dataset=seq2seq_dataset,
                                 criterion=criterion,
                                 opt=opt)

        logger.info('evaluate ----> %.4f' % evaluate_loss)

        # save model of each epoch
        save_state = {
            'epoch': epoch,
            'state_dict': seq2seq_model.state_dict(),
            'optimizer': optimizer.optimizer.state_dict()
        }
        save_checkpoint(save_state, False,
                        filename=os.path.join(opt.model_save_path, 'checkpoint.epoch-%d.pth' % epoch))


def train(seq2seq_model,
          encoder_input_data,
          decoder_input_data,
          decoder_target_data,
          optimizer,
          criterion,
          num_samples,
          vocab,
          opt):

    # Turn on training mode which enables dropout.
    # seq2seq_model.train()

    encoder_input_lengths = torch.ones((num_samples,)) * opt.dialog_encoder_max_length
    decoder_input_lengths = torch.ones((num_samples,)) * opt.dialog_decoder_max_length

#      if use_gpu:
        #  encoder_input_data = encoder_input_data.cuda()
        #  decoder_target_data = decoder_target_data.cuda()


    (dialog_encoder_final_state, dialog_encoder_memory_bank), \
    (dialog_decoder_memory_bank, dialog_decoder_final_stae, dialog_decoder_attns, dialog_decoder_outputs) \
        = seq2seq_model.forward(
        dialog_encoder_src=encoder_input_data,  # LongTensor
        dialog_encoder_src_lengths=encoder_input_lengths,
        dialog_decoder_tgt=decoder_input_data,
        dialog_decoder_tgt_lengths=decoder_input_lengths,
        use_teacher_forcing=opt.use_teacher_forcing,
        teacher_forcing_ratio=opt.teacher_forcing_ratio,
        batch_size=num_samples)

    optimizer.zero_grad()

    loss = 0

    #  Compute loss
    if dialog_decoder_outputs.is_cuda:
        dialog_decoder_outputs = dialog_decoder_outputs.cpu()
        decoder_target_data = decoder_target_data.cpu()

    dialog_decoder_outputs = dialog_decoder_outputs.view(-1, dialog_decoder_outputs.shape[-1],
                                                         dialog_decoder_outputs.shape[1])

    decoder_target_data = decoder_target_data.view(-1, decoder_target_data.shape[1])

    loss = criterion(dialog_decoder_outputs, decoder_target_data)

    loss.backward()

    optimizer.step()

    return loss.item() / decoder_input_lengths

    # if opt.use_teacher_forcing:
    #     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # else:
    #     use_teacher_forcing = False


'''
evaluate model.
'''


def evaluate(seq2seq_model=None,
             seq2seq_dataset=None,
             criterion=None,
             opt=None):


    # Turn on evaluation mode which disables dropout.
    # seq2seq_model.eval()

    loss_total = 0
    while not seq2seq_dataset.all_loaded('test'):

        # load data
        num_samples, \
        encoder_input_data, \
        decoder_input_data, \
        decoder_target_data, \
        conversation_texts, response_texts = seq2seq_dataset.load_data('test', opt.batch_size * opt.batch_per_load)

        # train and get cur loss
        encoder_input_lengths = torch.ones((num_samples,)) * opt.dialog_encoder_max_length
        decoder_input_lengths = torch.ones((num_samples,)) * opt.dialog_decoder_max_length

        #  if use_gpu:
            #  encoder_input_lengths.cuda()
            #  decoder_input_lengths.cuda()

        (dialog_encoder_final_state, dialog_encoder_memory_bank), \
        (dialog_decoder_memory_bank, dialog_decoder_final_stae, dialog_decoder_attns, dialog_decoder_outputs) \
            = seq2seq_model.forward(
            dialog_encoder_src=encoder_input_data,  # LongTensor
            dialog_encoder_src_lengths=encoder_input_lengths,
            dialog_decoder_tgt=decoder_input_data,
            dialog_decoder_tgt_lengths=decoder_input_lengths,
            use_teacher_forcing=opt.use_teacher_forcing,
            teacher_forcing_ratio=opt.teacher_forcing_ratio,
            batch_size=num_samples)

        #  Compute loss
        if dialog_decoder_outputs.is_cuda:
            dialog_decoder_outputs = dialog_decoder_outputs.cpu()
            decoder_target_data = decoder_target_data.cpu()

        dialog_decoder_outputs = dialog_decoder_outputs.view(-1, dialog_decoder_outputs.shape[-1],
                                                             dialog_decoder_outputs.shape[1])
        decoder_target_data = decoder_target_data.view(-1, decoder_target_data.shape[1])

        loss = criterion(dialog_decoder_outputs, decoder_target_data)

        loss_total += loss.item() / decoder_input_lengths

    return loss_total


def dialog(self, input_text):
    source_seq_int = []
    for token in input_text.strip().strip('\n').split(' '):
        source_seq_int.append(self.dataset.token2index.get(token, self.dataset.UNK))
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

    # load pre-trained embedding
    logger.info("Load pre-trained word embeddig: %s ." % opt.dialog_decoder_pretrained_embedding_path)
    dialog_encoder_pretrained_embedding_weight = np.load(opt.dialog_decoder_pretrained_embedding_path)
    dialog_decoder_pretrained_embedding_weight = dialog_encoder_pretrained_embedding_weight
    logger.info("dialog_encoder_pretrained_embedding_weight shape: {} .".format(dialog_encoder_pretrained_embedding_weight.shape))

    seq2seq_model = Seq2SeqModel(
        dialog_encoder_embedding_size = opt.dialog_encoder_embedding_size,
        dialog_encoder_vocab_size=dialog_encoder_vocab.get_vocab_size(),
        dialog_encoder_hidden_size=opt.dialog_encoder_hidden_size,
        dialog_encoder_num_layers=opt.dialog_encoder_num_layers,
        dialog_encoder_rnn_type=opt.dialog_encoder_rnn_type,
        dialog_encoder_dropout_rate=opt.dialog_encoder_dropout_rate,
        dialog_encoder_max_length=opt.dialog_encoder_max_length,
        dialog_encoder_clipnorm=opt.dialog_encoder_clipnorm,
        dialog_encoder_clipvalue=opt.dialog_encoder_clipvalue,
        dialog_encoder_bidirectional=opt.dialog_encoder_bidirectional,
        dialog_encoder_pretrained_embedding_weight=dialog_encoder_pretrained_embedding_weight,
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
        dialog_decoder_pretrained_embedding_weight=dialog_decoder_pretrained_embedding_weight,
        dialog_decoder_pad_id=dialog_decoder_vocab.padid,
        dialog_decoder_attention_type=opt.dialog_decoder_attention_type,
        dialog_decoder_tied=opt.dialog_decoder_tied)

    seq2seq_model = seq2seq_model.to(device)
#      if use_gpu:
        #  seq2seq_model.set_cuda()

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

    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = load_checkpoint(filename='checkpoint.pth')
        opt.start_epoch = checkpoint['epoch'] + 1
        # checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)

        # I don't like reassigning attributes of opt: it's not clear.
    else:
        checkpoint = None

    vocab = Vocab()
    vocab.load(opt.vocab_save_path)
    vocab_size = vocab.get_vocab_size()
    logger.info("vocab_size -----> %d" % vocab_size)

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

    # Build optimizer.
    optimizer = build_optim(seq2seq_model, opt)

    # criterion = nn.CrossEntropyLoss()
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    criterion = nn.NLLLoss()


    '''if load checkpoint'''
    if checkpoint:
        seq2seq_model.load_state_dict(checkpoint['state_dict'])
        seq2seq_model.eval()
        optimizer.optimizer.load_state_dict(checkpoint['optimizer'])

    train_epochs(seq2seq_model=seq2seq_model,
                 seq2seq_dataset=seq2seq_dataset,
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
