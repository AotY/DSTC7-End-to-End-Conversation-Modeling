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

from tqdm import tqdm

from modules.optim import ScheduledOptimizer
from modules.early_stopping import EarlyStopping

from gensim.models import KeyedVectors

from misc.vocab import Vocab
from misc.vocab import PAD_ID
from kg_model import KGModel
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
args = parser.parse_args()

# logger file
time_str = time.strftime('%Y_%m_%d_%H:%M')
args.log_path = args.log_path.format(
    args.model_type, args.enc_type, args.c_min, args.c_max, time_str)
logger.info('log_path: {}'.format(args.log_path))

device = torch.device(args.device)
logging.info("device: %s" % device)

if args.seed:
    torch.manual_seed(args.seed)

# update max_len
if args.enc_type == 'qc':
    args.q_max_len = args.q_max_len + args.c_max_len * args.c_max

logger.info('c_max_len: %d' % args.c_max_len)
logger.info('q_max_len: %d' % args.q_max_len)
logger.info('r_max_len: %d' % args.r_max_len)
logger.info('f_max_len: %d' % args.f_max_len)

# update turn min
args.turn_num = args.c_max + 1

def train_epochs(model,
                 dataset,
                 optimizer,
                 criterion,
                 vocab,
                 early_stopping):

    start = time.time()
    max_load = int(np.ceil(dataset._size_dict['train'] / args.batch_size))
    evaluate_loss_list = []
    evaluate_accuracy_list = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        dataset.reset_data('train')
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0
        start = time.time()

        for load in range(1, max_load + 1):
            # load data
            dec_inputs, enc_inputs, enc_inputs_length, \
                enc_turn_length, f_inputs, f_inputs_length, f_topk_length, \
                subreddit_names, conversation_ids, hash_values = dataset.load_data('train')

            # train and get cur loss
            loss, n_correct, n_word = train(model,
                                            enc_inputs,
                                            enc_inputs_length,
                                            enc_turn_length,
                                            dec_inputs,
                                            f_inputs,
                                            f_inputs_length,
                                            f_topk_length,
                                            optimizer,
                                            criterion)

            total_loss += loss

            n_word_total += n_word
            n_word_correct += n_correct

            if load % args.log_interval == 0:
                train_loss = total_loss/n_word_total
                train_accu = n_word_correct/n_word_total

                logger_str = ' ( Training {progress: 3.3f}% ) epoch: {epoch: 2d}, ' \
                    'loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu: 3.3f} %, ' \
                    'elapse: {elapse:3.3f} min'.format(
                        progress=100*(load / max_load),
                        epoch=epoch,
                        loss=train_loss,
                        ppl=math.exp(min(train_loss, 100)),
                        accu=100*train_accu,
                        elapse=(time.time()-start)/60)
                logger.info(logger_str)
                #  print(logger_str)
                save_logger(logger_str)
                total_loss = 0
                n_word_total = 0
                n_word_correct = 0
                start = time.time()

            if load % args.eval_interval == 0:
                # evaluate
                evaluate_loss, evaluate_accuracy = evaluate(model=model,
                                                            dataset=dataset,
                                                            criterion=criterion)
                logger_str = ' (evaluate) epoch: {epoch: 2d},' \
                    'loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu: 3.3f} %'.format(
                        epoch=epoch,
                        loss=evaluate_loss,
                        ppl=math.exp(min(evaluate_loss, 100)),
                        accu=100*evaluate_accuracy)
                logger.info(logger_str)
                save_logger(logger_str)

                # lr update
                optimizer.update(evaluate_loss)
                evaluate_loss_list.append(evaluate_loss)
                evaluate_accuracy_list.append(evaluate_accuracy)

                is_stop = early_stopping.step(evaluate_loss)

                if is_stop:
                    avg_evaluate_loss = np.mean(evaluate_loss_list[-(args.es_patience + 2):])
                    avg_evaluate_accuracy = np.mean(
                        evaluate_accuracy_list[-(args.es_patience + 2):])
                    logger_str = ' (avg evaluate)' \
                        'loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu: 3.3f} %'.format(
                            loss=avg_evaluate_loss,
                            ppl=math.exp(min(avg_evaluate_loss, 100)),
                            accu=100*avg_evaluate_accuracy
                        )
                    save_logger(logger_str)
                    logger.info(logger_str)
                    logger.info('Early Stopping.')
                    save_model(epoch, avg_evaluate_loss, avg_evaluate_accuracy, model, optimizer)
                    sys.exit(0)

        save_model(epoch, evaluate_loss, evaluate_accuracy, model, optimizer)

        # generate sentence
        logger.info('generate...')
        decode(model, dataset, epoch)


''' start traing '''


def train(model,
          enc_inputs,
          enc_inputs_length,
          enc_turn_length,
          dec_inputs,
          f_inputs,
          f_inputs_length,
          f_topk_length,
          optimizer,
          criterion):

    # Turn on training mode which enables dropout.
    model.train()

    optimizer.zero_grad()

    loss = 0

    # [max_len, batch_size, vocab_size]
    dec_outputs = model(
        enc_inputs,
        enc_inputs_length,
        enc_turn_length,
        dec_inputs[:-1, :],
        f_inputs,
        f_inputs_length,
        f_topk_length
    )

    loss, n_correct = cal_performance(
        dec_outputs, dec_inputs[1:, :], label_smoothing=args.label_smoothing)

    non_pad_mask = dec_inputs[1:, :].ne(PAD_ID)

    n_word = non_pad_mask.sum().item()

    # backward
    loss.backward()

    # optimizer
    optimizer.step()

    return loss.item(), n_correct, n_word


'''
evaluate model.
'''


def evaluate(model,
             dataset,
             criterion):
    logger.info('evaluate...')
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    max_load = int(np.ceil(dataset._size_dict['test'] / args.batch_size))
    dataset.reset_data('test', True)
    with torch.no_grad():
        for load in tqdm(range(1, max_load + 1)):
            # load data
            dec_inputs, enc_inputs, enc_inputs_length, \
                enc_turn_length, f_inputs, f_inputs_length, f_topk_length, \
                subreddit_names, conversation_ids, hash_values = dataset.load_data('test')

            dec_outputs = model(
                enc_inputs,
                enc_inputs_length,
                enc_turn_length,
                dec_inputs[:-1, :],
                f_inputs,
                f_inputs_length,
                f_topk_length
            )

            loss, n_correct = cal_performance(
                dec_outputs, dec_inputs[1:, :], label_smoothing=False)

            non_pad_mask = dec_inputs[1:, :].ne(PAD_ID)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

            total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def decode(model, dataset, epoch):
    logger.info('decode...')
    # Turn on evaluation mode which disables dropout.
    model.eval()
    dataset.reset_data('eval', False)
    max_load = int(np.floor(dataset._size_dict['eval'] / args.batch_size))
    dataset.save_ground_truth('eval')
    with torch.no_grad():
        for load in tqdm(range(1, max_load + 1)):
            dec_inputs, enc_inputs, enc_inputs_length, \
                enc_turn_length, f_inputs, f_inputs_length, f_topk_length, \
                subreddit_names, conversation_ids, hash_values = dataset.load_data('eval')

            # greedy: [batch_size, max_len]
            # beam_search: [batch_sizes, best_n, len]
            greedy_outputs, beam_outputs, beam_length = model.decode(
                enc_inputs,
                enc_inputs_length,
                enc_turn_length,
                f_inputs,
                f_inputs_length,
                f_topk_length
            )

            # generate sentence, and save to file
            # [batch_size, max_len]
            greedy_texts = dataset.generating_texts(greedy_outputs, decode_type='greedy')

            # [batch_size, topk, max_len]
            beam_texts = dataset.generating_texts(beam_outputs,
                                                  decode_type='beam_search')

            # save sentences
            dataset.save_generated_texts(
                epoch,
                subreddit_names,
                conversation_ids,
                hash_values,
                enc_inputs,
                f_inputs,
                dec_inputs,
                greedy_texts,
                beam_texts,
                os.path.join(args.save_path, 'generated/%s_%s_%s_%s_%s_%s.txt' % (
                    args.model_type, args.enc_type, epoch, args.c_min, args.c_max, time_str)),
                time_str
            )


def cal_performance(pred, gold, label_smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, label_smoothing)

    pred = pred.max(1)[1]  # [max_len * batch_size]
    gold = gold.contiguous().view(-1)  # [max_len * batch_size]

    non_pad_mask = gold.ne(PAD_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, label_smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)  # [max_len * batch_size]

    if label_smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(
            dim=1, index=gold.view(-1, 1), source=1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(PAD_ID)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(
            pred, gold, ignore_index=PAD_ID, reduction='sum')

    return loss


''' save log to file '''


def save_logger(logger_str):
    with open(args.log_path, 'a', encoding='utf-8') as f:
        f.write(logger_str + '\n')


def build_optimizer(model):
    optim = torch.optim.Adam(
        model.parameters(),
        args.lr,
        betas=(0.9, 0.98),
        eps=1e-09
    )

    #  scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode='min',
        factor=0.1,
        patience=args.lr_patience
    )

    optimizer = ScheduledOptimizer(
        optim,
        scheduler,
        args.max_grad_norm
    )

    return optimizer


def build_criterion(padid):
    # The negative log likelihood loss. It is useful to train a classification problem with `C` classes.
    #  criterion = nn.NLLLoss(reduction='elementwise_mean', ignore_index=padid)
    #  criterion = nn.NLLLoss(reduction='sum', ignore_index=padid)
    criterion = None

    return criterion


def load_fasttext_embedding(fasttext, vocab):
    words_embedded = list()
    for id, word in sorted(vocab.idx2word.items(), key=lambda item: item[0]):
        try:
            word_embedded = torch.from_numpy(fasttext[word])
        except KeyError:
            word_embedded = torch.rand(args.pre_embedding_size)

        word_embedded = word_embedded.to(device)
        words_embedded.append(word_embedded)
    # [vocab_size, pre_embedding_size]
    words_embedded = torch.stack(words_embedded)

    return words_embedded


def build_model(vocab):
    logger.info('Building model...')

    model = KGModel(
        args,
        device,
    ).to(device)

    print(model)
    return model


def build_dataset(vocab):
    dataset = Dataset(
        args,
        vocab,
        device,
        logger)

    return dataset

def save_model(epoch, loss, accuracy, model, optimizer):
    # save model of each epoch
    save_state = {
        'loss': loss,
        'ppl': math.exp(min(loss, 100)),
        'acc': accuracy,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.optimizer.state_dict()
    }

    # save checkpoint, including epoch, seq2seq_mode.state_dict() and
    save_checkpoint(state=save_state,
                    is_best=False,
                    filename=os.path.join(args.model_path, '%s_%s_%s_%s_%s_%s.pth' %
                                            (args.model_type, args.enc_type, epoch,
                                            args.c_min, args.c_max, time_str)))

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
        shutil.copy(filename, 'model_best_%s.pth' % args.model_type)


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
    if args.checkpoint:
        logger.info('Loading checkpoint from: %s' % args.checkpoint)
        checkpoint = load_checkpoint(filename=args.checkpoint)
    else:
        checkpoint = None

    vocab = Vocab()
    vocab.load(args.vocab_path)
    vocab_size = vocab.size
    args.vocab_size = int(vocab_size)
    logger.info("vocab_size: %d" % args.vocab_size)

    dataset = build_dataset(vocab)

    fasttext = None
    pre_trained_weight = None
    if (args.pre_trained_embedding or args.offline_type == 'fasttext') and os.path.exists(args.fasttext_vec):
        logger.info('load pre trained embedding...')
        fasttext = load_fasttext_model(args.fasttext_vec)
        pre_trained_weight = load_fasttext_embedding(fasttext, vocab)

    if args.model_type == 'kg':
        """ computing similarity between conversation and fact """
        offline_filename = os.path.join(
            args.save_path, 'facts_topk_phrases.%s.pkl' % args.offline_type)

        facts_tfidf_dict = pickle.load(open('./data/facts_tfidf_dict.pkl', 'rb'))
        if args.offline_type in ['elastic', 'elastic_tag']:
            dataset.build_similarity_facts_offline(
                offline_filename=offline_filename,
                facts_tfidf_dict=facts_tfidf_dict
            )
        elif args.offline_type == 'fasttext':
            facts_dict = None
            if not os.path.exists(offline_filename):
                facts_dict = pickle.load(open('./data/facts_p_dict.pkl', 'rb'))
                embedding = nn.Embedding.from_pretrained(
                    pre_trained_weight, freeze=True)
                with torch.no_grad():
                    dataset.build_similarity_facts_offline(
                        facts_dict,
                        offline_filename,
                        embedding=embedding,
                    )
                del embedding
            else:
                dataset.load_similarity_facts(offline_filename)
        elif args.offline_type == 'elmo':
            pass

    model = build_model(vocab)

    # Build optimizer.
    optimizer = build_optimizer(model)

    criterion = build_criterion(vocab.padid)

    early_stopping = EarlyStopping(
        type='min',
        min_delta=0.001,
        patience=args.es_patience
    )

    '''if load checkpoint'''
    epoch = 1
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        args.start_epoch = epoch + 1
        loss = checkpoint['loss']
        ppl = checkpoint['ppl']
        acc = checkpoint['acc']
        logger_str = ' (checkpoint) epoch: {epoch: 2d},' \
            'loss: {loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu: 3.3f} %'.format(
                epoch=epoch,
                loss=loss,
                ppl=math.exp(min(loss, 100)),
                accu=100*acc)
        logger.info(logger_str)

    if args.task == 'train':
        train_epochs(model=model,
                     dataset=dataset,
                     optimizer=optimizer,
                     criterion=criterion,
                     vocab=vocab,
                     early_stopping=early_stopping)
    elif args.task == 'evaluate':
        evaluate(model,
                 dataset,
                 criterion)
    elif args.task == 'decode':
        decode(model, dataset, epoch)
    else:
        raise ValueError(
            "task must be train or eval, no %s " % args.task)
