# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import random
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils.data_set import Seq2seqDataSet

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import seaborn

seaborn.set()

import torch
import torch.nn as nn
from torch import optim

from utils.vocab import Vocab
from train_evaluate_opt import data_set_opt, train_seq2seq_opt
from utils.vocab import SOS, EOS, UNK, PAD

from seq2seq_model import Seq2SeqModel

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

if opt.use_teacher_forcing:
    teacher_forcing_ratio = opt.teacher_forcing_ratio
    logging.info("teacher_forcing_ratio: %f" % teacher_forcing_ratio)

torch.manual_seed(opt.seed)


def trainIters(seq2seq_model, seq2seq_dataset, batch_size, epochs, batch_per_load, log_interval=200, plot_every=100,
               lr=0.001):
    start = time.time()
    plot_losses = []

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    seq2seq_model_optimizer = optim.Adam(
        params=seq2seq_model.parameters(),
        lr=lr)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()  # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.

    max_load = np.ceil(seq2seq_dataset.n_train / batch_size / batch_per_load)

    for epoch in range(epochs):

        load = 0
        seq2seq_dataset.reset()
        while not seq2seq_dataset.all_loaded('train'):
            load += 1
            logger.info('\n***** Epoch %i/%i - load %.2f perc *****' % (epoch + 1, epochs, 100 * load / max_load))
            encoder_input_data, decoder_input_data, decoder_target_data, _, _ = seq2seq_dataset.load_data('train',
                                                                                                          batch_size * batch_per_load)
            loss = train(seq2seq_model, encoder_input_data, decoder_input_data,
                         decoder_target_data, seq2seq_model_optimizer, criterion, batch_size)

            print_loss_total += loss
            plot_loss_total += loss

            if load % log_interval == 0:
                print_loss_avg = print_loss_total / log_interval
                print_loss_total = 0
                logger.info('%s (%d %d%%) %.4f' % (timeSince(start, iter / max_load),
                                                   load, load / max_load * 100, print_loss_avg))

            if load % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


def train(seq2seq_model,
          encoder_input_data,
          encoder_input_lengths,
          decoder_input_data,
          decoder_input_lengths,
          decoder_target_data,
          decoder_target_lengths,
          seq2seq_model_optimizer,
          criterion,
          batch_size):


    (dialog_encoder_final_state, dialog_encoder_memory_bank), (
    dialog_decoder_memory_bank, dialog_decoder_final_stae, dialog_decoder_attns) = seq2seq_model.forward(
        dialog_encoder_src=encoder_input_data,  # LongTensor
        dialog_encoder_src_lengths=encoder_input_lengths,
        dialog_decoder_tgt=decoder_input_data,
    )

    seq2seq_model_optimizer.zero_grad()

    loss = 0

    for ei in range(encoder_input_lengths):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS]], device=device)

    decoder_hidden = encoder_hidden

    if opt.use_teacher_forcing:
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    else:
        use_teacher_forcing = False

    # Teacher forcing: Feed the target as the next input
    # if use_teacher_forcing:
    #     for di in range(target_length):
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs)
    #         loss += criterion(decoder_output, decoder_target_data[di])
    #         decoder_input = decoder_target_data[di]  # Teacher forcing
    #
    # else:  # Without teacher forcing: use its own predictions as the next input
    #     for di in range(decoder_target_lengths):
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs)
    #         topv, topi = decoder_output.topk(1)
    #         decoder_input = topi.squeeze().detach()  # detach from history as input
    #
    #         loss += criterion(decoder_output, target_tensor[di])
    #         if decoder_input.item() == EOS:
    #             break

    loss.backward()

    seq2seq_model_optimizer.step()

    return loss.item() / decoder_target_lengths


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def dialog(self, input_text):
    source_seq_int = []
    for token in input_text.strip().strip('\n').split(' '):
        source_seq_int.append(self.dataset.token2index.get(token, self.dataset.UNK))
    return self._infer(np.atleast_2d(source_seq_int))


def interact(self):
    while True:
        print('----- please input -----')
        input_text = input()
        if not bool(input_text):
            break
        print(self.dialog(input_text))


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
    vocab = Vocab()
    vocab.load()

    seq2seq_dataset = Seq2seqDataSet(
        path_conversations=opt.path_conversations,
        path_responses=opt.path_responses,

        dialog_encoder_vocab_size=opt.dialog_encoder_vocab_size,
        dialog_encoder_max_length=opt.dialog_encoder_max_length,
        dialog_encoder_vocab=vocab,

        dialog_decoder_vocab_size=opt.dialog_encoder_vocab_size,
        dialog_decoder_max_length=opt.dialog_encoder_max_length,
        dialog_decoder_vocab=vocab,

        test_split=opt.test_split,  # how many hold out as vali data
        device=device,
        logger=logger
    )

    # load pre-trained embedding
    dialog_encoder_pretrained_embedding_weight = None
    dialog_decoder_pretrained_embedding_weight = None

    seq2seq_model = Seq2SeqModel(
                             dialog_encoder_vocab_size=opt.dialog_encoder_vocab_size,
                             dialog_encoder_hidden_size=opt.dialog_encoder_hidden_size,
                             dialog_encoder_num_layers=opt.dialog_encoder_num_layers,
                             dialog_encoder_rnn_type=opt.dialog_encoder_rnn_type,
                             dialog_encoder_dropout_rate=opt.dialog_encoder_dropout_rate,
                             dialog_encoder_max_length=opt.dialog_encoder_max_length,
                             dialog_encoder_clip_grads=opt.dialog_encoder_clip_grads,
                             dialog_encoder_bidirectional=opt.dialog_encoder_bidirectional,
                             dialog_encoder_pretrained_embedding_weight=dialog_encoder_pretrained_embedding_weight,

                             dialog_decoder_vocab_size=opt.dialog_decoder_vocab_size,
                             dialog_decoder_hidden_size=opt.dialog_decoder_hidden_size,
                             dialog_decoder_num_layers=opt.dialog_decoder_num_layers,
                             dialog_decoder_rnn_type=opt.dialog_decoder_rnn_type,
                             dialog_decoder_dropout_rate=opt.dialog_decoder_dropout_rate,
                             dialog_decoder_max_length=opt.dialog_decoder_max_length,
                             dialog_decoder_clip_grads=opt.dialog_decoder_clip_grads,
                             dialog_decoder_bidirectional=opt.dialog_decoder_bidirectional,
                             dialog_decoder_pretrained_embedding_weight=dialog_decoder_pretrained_embedding_weight)

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
