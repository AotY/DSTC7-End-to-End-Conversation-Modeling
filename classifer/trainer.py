#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
This is the loadable Seq2Seq trainer libray that is in charge
of trainning details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:
    * accuracy
    * perplexity
    * elapsed time
    * MRR, MAP, NDCG, RC,
    """
    def __init__(self, loss=0, n_sample=0, n_correct=0, scores=[], pos_neg_num=10):
        self.loss = loss
        self.n_sample = n_sample
        self.n_correct = n_correct
        self.scores = scores
        self.start_time = time.time()
        self.pos_neg_num = pos_neg_num # postive number + negative_num 
    
    def set_scores(self, scores):
        self.scores = scores 

    def update(self, stat):
        self.loss += stat.loss
        self.n_sample += stat.n_sample
        self.n_correct += stat.n_correct
        self.scores += stat.scores

    def accuracy(self):
        return 100 * (self.n_correct / self.n_sample)

    def xent(self):
        return self.loss / self.n_sample

    def elapsed_time(self):
        return time.time() - self.start_time
    
    def recall_ks(self):
        """recall@N, #### Recall@N
        """
        probas = self.scores
        recall_k = {}
        if isinstance(probas, list):
            probas = probas
        elif isinstance(probas, torch.Tensor):
            probas = probas.numpy().tolist()
        else:
            print("")
        for group_size in [2, 10]:
            recall_k[group_size] = {}
            for k in [1, 2, 5]:
                if k < group_size:
                    recall_k[group_size][k] = self.recall(probas, k, group_size)
        return recall_k

    def recall(self, probas, k, group_size):
        test_size = self.pos_neg_num
        n_batches = len(probas) // test_size
        n_correct = 0
        for i in xrange(n_batches):
            batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
            indices = np.argpartition(batch, -k)[-k:]
            if 0 in indices:
                n_correct += 1
        return n_correct*1.0 / (len(probas) / test_size)
    
    def MRR(self, probas):
        # Mean Reciprocal Rank (MRR)
        n_batches = len(probas) // self.pos_neg_num
        ranks = []
        for i in xrange(n_batches):
            batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
            indices = (np.argsort(batch)).tolist()
            indices.reverse()
            ranks.append( 1. / (indices.index(0) + 1.) )
        # 
        return sum(ranks) / len(ranks)
    
    def MAP(self, probas):
        # Mean Average Precision (MAP) # NDCG(Normalized Discounted Cumulative Gain)
        pass 

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.
        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; xent: %6.4f; " +
               "%3.0f sample /s; %6.0f s elapsed ") %
              (epoch, batch, n_batches,
               self.accuracy(),
               self.xent(),
               self.n_sample / (t + 1e-5),
               time.time() - start))
        #if len(self.scores) > 0:
        #    print("Recall@K on different group", self.recall_ks())
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_sample / t)
        experiment.add_scalar_value(prefix + "_lr", lr)
     
    def log_tensorboard(self, prefix, writer, lr, step):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper",  self.n_sample / t, step)
        writer.add_scalar(prefix + "/lr", lr, step)


"""
Single Input Trainer
"""

class SingleTrainer(object):
    """
    Class that controls the training process.
    Args:
            model(:py:class:`classifier.Model`): translation model to train
            optimizer(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            criterion(:obj: nn.loss, nn.BECLoss, nn.CrossEntropy)
    """

    def __init__(self, model, optimizer, criterion, use_cuda, norm_method="sents"):
        # Basic attributes.
        self.model = model
        self.optimizer = optimizer
        self.norm_method = norm_method
        self.progress_step = 0
        self.use_cuda = use_cuda
        self.criterion = criterion

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging
        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        # Set model in training mode.
        self.model.train()
        total_stats = Statistics(loss=0, n_sample=0, n_correct=0,
                                scores=[], pos_neg_num=10)
        report_stats = Statistics(loss=0, n_sample=0, n_correct=0,
                                scores=[], pos_neg_num=10)

        for sub_dataset in train_iter:
            batch_num = 0
            num_batch = len(sub_dataset.examples) / sub_dataset.batch_size
            for batch in sub_dataset:
                src, src_lengths, labels = batch 
                if self.use_cuda:
                    src, src_lengths, labels= (src.cuda(),
                                              src_lengths.cuda(), labels.cuda()) 
                self.optimizer.optimizer.zero_grad()
                _, batch_size = src.size()
                batch_stats, loss, _ = self.forward_step(src, src_lengths, labels)
                # 4. Update the parameters and statistics.
                loss.div(batch_size).backward()
                self.optimizer.step()
                self.progress_step += 1

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                batch_num += 1
                if report_func is not None:
                    report_stats = report_func(
                        epoch, batch_num, num_batch,
                        self.progress_step,
                        total_stats.start_time, 0.001,
                        report_stats)
                # 
        return total_stats

    def epoch_step(self, ppl, epoch):
        return self.optimizer.update_learning_rate(ppl, epoch)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets
        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        if self.model.output_size == 1:
            outputs_label = torch.floor( scores.squeeze() + 0.5 )
            num_correct = torch.eq(outputs_label, target).float().sum()
        else:
            pred = scores.max(1)[1]
            num_correct = pred.eq(target).sum()
        
        return Statistics(loss=loss[0], n_sample=target.size(0), 
            n_correct=num_correct, scores=[], pos_neg_num=10)

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        valid_stats = Statistics(loss=0, n_sample=0, n_correct=0,
                                 scores=[], pos_neg_num=10)

        for sub_dataset in valid_iter:
            for batch in sub_dataset:
                src, src_lengths, labels = batch 
                if self.use_cuda:
                    src, src_lengths, labels= (src.cuda(),
                                              src_lengths.cuda(), labels.cuda()) 
                batch_stats, loss, outputs = self.forward_step(src, src_lengths, labels)
                # Update statistics.
                batch_stats.set_scores([outputs.data.cpu().numpy()]) ### ??????
                valid_stats.update(batch_stats)
        # Set model back to training mode.
        self.model.train()

        return valid_stats

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.
        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        model_state_dict = real_model.state_dict()

        checkpoint = {
            'model': model_state_dict,
            'opt': opt,
            'epoch': epoch,
            'optim': self.optimizer,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.xent(), epoch))

    def forward_step(self, src, src_lengths, labels):
        # 2. F-prop all but generator.
        outputs, enc_final = \
                self.model(src, src_lengths)
        # 3. Compute loss 
        loss = self.criterion(outputs, labels)

        loss_data = loss.data.clone()
        batch_stats = self._stats(loss_data, outputs.data, labels.data)
        # 
        return batch_stats, loss, outputs


"""
Double Trainer
"""


class DoubleTrainer(object):
    """
    Class that controls the training process.
    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train
            optimizer(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            norm_method(string): normalization methods: [sents|tokens]
    """

    def __init__(self, model, optimizer, criterion, use_cuda, norm_method="sents"):
        # Basic attributes.
        self.model = model
        self.optimizer = optimizer
        self.norm_method = norm_method
        self.progress_step = 0
        self.use_cuda = use_cuda
        self.criterion = criterion

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging
        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        # Set model in training mode.
        self.model.train()
        total_stats = Statistics(loss=0, n_sample=0, n_correct=0,
                                scores=[], pos_neg_num=10)
        report_stats = Statistics(loss=0, n_sample=0, n_correct=0,
                                scores=[], pos_neg_num=10)

        for sub_dataset in train_iter:
            batch_num = 0
            num_batch = len(sub_dataset.examples) / sub_dataset.batch_size
            for batch in sub_dataset:
                q_src, q_src_lengths, r_src, r_src_lengths, labels = batch 
                if self.use_cuda:
                    q_src, q_src_lengths, r_src, r_src_lengths, labels= (q_src.cuda(),
                                              q_src_lengths.cuda(), r_src.cuda(),
                                              r_src_lengths.cuda(), labels.cuda()) 
                self.optimizer.optimizer.zero_grad()
                _, batch_size = q_src.size()
                batch_stats, loss, _ = self.forward_step(q_src, q_src_lengths, 
                                        r_src, r_src_lengths, labels)
                # 4. Update the parameters and statistics.
                loss.div(batch_size).backward()
                self.optimizer.step()
                self.progress_step += 1

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                batch_num += 1
                if report_func is not None:
                    report_stats = report_func(
                        epoch, batch_num, num_batch,
                        self.progress_step,
                        total_stats.start_time, 0.001,
                        report_stats)
                # 
        return total_stats

    def epoch_step(self, ppl, epoch):
        return self.optimizer.update_learning_rate(ppl, epoch)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets
        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        if self.model.output_size == 1:
            outputs_label = torch.floor( F.sigmoid(scores).squeeze() + 0.5 )
            num_correct = torch.eq(outputs_label, target).float().sum()
        else:
            pred = scores.max(1)[1]
            num_correct = pred.eq(target).sum()
        
        return Statistics(loss=loss[0], n_sample=target.size(0),
               n_correct=num_correct, scores=[], pos_neg_num=10)

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        valid_stats = Statistics(loss=0., n_sample=0, n_correct=0, scores=[])

        for sub_dataset in valid_iter:
            for batch in sub_dataset:
                q_src, q_src_lengths, r_src, r_src_lengths, labels = batch 
                if self.use_cuda:
                    q_src, q_src_lengths, r_src, r_src_lengths, labels= (q_src.cuda(),
                                              q_src_lengths.cuda(), r_src.cuda(),
                                              r_src_lengths.cuda(),  labels.cuda()) 
                _, batch_size = q_src.size()
                batch_stats, _, outputs = self.forward_step(q_src, q_src_lengths,
                                    r_src, r_src_lengths, labels)
                # Update statistics.
                # batch_stats.set_scores([outputs.data.cpu().numpy()])
                batch_stats.set_scores(outputs.data.cpu().numpy().tolist())
                valid_stats.update(batch_stats)
        # Set model back to training mode.
        self.model.train()

        return valid_stats

    def test(self, test_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        test_stats = Statistics(loss=0., n_sample=0, n_correct=0, scores=[])

        print('valid', len(valid_stats.scores))
        for batch in test_iter:
            q_src, q_src_lengths, r_src, r_src_lengths, labels = batch 
            if self.use_cuda:
                q_src, q_src_lengths, r_src, r_src_lengths, labels= (q_src.cuda(),
                                              q_src_lengths.cuda(), r_src.cuda(),
                                              r_src_lengths.cuda(),  labels.cuda()) 
            _, batch_size = q_src.size()
            batch_stats, _, outputs = self.forward_step(q_src, q_src_lengths,
                                    r_src, r_src_lengths, labels)
            # Update statistics.
            temp = outputs.data.cpu().numpy().tolist()
            batch_stats.set_scores(outputs.data.cpu().numpy().tolist())
            test_stats.update(batch_stats)
        # Set model back to training mode.
        self.model.train()

        return test_stats

    def drop_checkpoint(self, opt, epoch, valid_stats):
        """ Save a resumable checkpoint.
        Args:
            opt (dict): option object
            epoch (int): epoch number
            ### fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        model_state_dict = real_model.state_dict()
        
        checkpoint = {
            'model': model_state_dict,
            'opt': opt,
            'epoch': epoch,
            'optim': self.optimizer,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.xent(), epoch))

    def forward_step(self, q_src, q_src_lengths, r_src, r_src_lengths, labels):
        # 2. F-prop all but generator.
        outputs, enc_final = \
                self.model(q_src, q_src_lengths, r_src, r_src_lengths)
        # 3. Compute loss 
        loss = self.criterion(outputs, labels)

        loss_data = loss.data.clone()
        batch_stats = self._stats(loss_data, outputs.data, labels.data)
        # 
        return batch_stats, loss, outputs 

class PairTrainer(object):
    """
    Class that controls the training process.
    Using MarginRankingLoss
    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train
            optimizer(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            norm_method(string): normalization methods: [sents|tokens]
    """

    def __init__(self, model, optimizer, criterion, use_cuda, norm_method="sents"):
        # Basic attributes.
        self.model = model
        self.optimizer = optimizer
        self.norm_method = norm_method
        self.progress_step = 0
        self.use_cuda = use_cuda
        self.criterion = criterion

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging
        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        # Set model in training mode.
        self.model.train()
        total_stats = Statistics(loss=0, n_sample=0, n_correct=0,
                                scores=[], pos_neg_num=10)
        report_stats = Statistics(loss=0, n_sample=0, n_correct=0,
                                scores=[], pos_neg_num=10)

        for sub_dataset in train_iter:
            batch_num = 0
            num_batch = len(sub_dataset.examples) / sub_dataset.batch_size
            for batch in sub_dataset:
                q_src, q_src_lengths, pos_src, pos_src_lengths, neg_src, neg_src_lengths, labels = batch 
                if self.use_cuda:
                    q_src, q_src_lengths, pos_src, pos_src_lengths, neg_src, neg_src_lengths, labels= (q_src.cuda(),
                                              q_src_lengths.cuda(), pos_src.cuda(),
                                              pos_src_lengths.cuda(), neg_src.cuda(),
                                              neg_src_lengths.cuda(), labels.cuda()) 
                self.optimizer.optimizer.zero_grad()
                _, batch_size = q_src.size()
                batch_stats, loss, _ = self.forward_step(q_src, q_src_lengths, 
                                                pos_src, pos_src_lengths, 
                                                neg_src, neg_src_lengths, labels)
                # 4. Update the parameters and statistics.
                loss.div(batch_size).backward()
                self.optimizer.step()
                self.progress_step += 1

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                batch_num += 1
                if report_func is not None:
                    report_stats = report_func(
                        epoch, batch_num, num_batch,
                        self.progress_step,
                        total_stats.start_time, 0.001,
                        report_stats)
                # 
        return total_stats

    def epoch_step(self, ppl, epoch):
        return self.optimizer.update_learning_rate(ppl, epoch)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets
        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        num_correct = ((scores[0] - scores[1]) > 0).sum()
        
        return Statistics(loss=loss[0], n_sample=target.size(0),
               n_correct=num_correct, scores=[], pos_neg_num=10)

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        valid_stats = Statistics(loss=0., n_sample=0, n_correct=0, scores=[])

        for sub_dataset in valid_iter:
            for batch in sub_dataset:
                q_src, q_src_lengths, pos_src, pos_src_lengths, neg_src, neg_src_lengths, labels = batch 
                if self.use_cuda:
                    q_src, q_src_lengths, pos_src, pos_src_lengths, neg_src, neg_src_lengths, labels= (q_src.cuda(),
                                              q_src_lengths.cuda(), pos_src.cuda(),
                                              pos_src_lengths.cuda(), neg_src.cuda(),
                                              neg_src_lengths.cuda(), labels.cuda()) 
                _, batch_size = q_src.size()
                batch_stats, loss, outputs = self.forward_step(q_src, q_src_lengths, 
                                                pos_src, pos_src_lengths, 
                                                neg_src, neg_src_lengths, labels)
                # Update statistics.
                batch_stats.set_scores(outputs.data.cpu().numpy().tolist())
                valid_stats.update(batch_stats)
        # Set model back to training mode.
        self.model.train()

        return valid_stats

    def test(self, test_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        test_stats = Statistics(loss=0., n_sample=0, n_correct=0, scores=[])

        print('valid', len(valid_stats.scores))
        for batch in test_iter:
            q_src, q_src_lengths, pos_src, pos_src_lengths, neg_src, neg_src_lengths, labels = batch 
            if self.use_cuda:
                q_src, q_src_lengths, pos_src, pos_src_lengths, neg_src, neg_src_lengths, labels= (q_src.cuda(),
                                              q_src_lengths.cuda(), pos_src.cuda(),
                                              pos_src_lengths.cuda(), neg_src.cuda(),
                                              neg_src_lengths.cuda(), labels.cuda()) 
            _, batch_size = q_src.size()
            batch_stats, loss, outputs = self.forward_step(q_src, q_src_lengths, 
                                                pos_src, pos_src_lengths, 
                                                neg_src, neg_src_lengths, labels)
            # Update statistics.
            batch_stats.set_scores(outputs.data.cpu().numpy().tolist())
            test_stats.update(batch_stats)
        # Set model back to training mode.
        self.model.train()

        return test_stats

    def drop_checkpoint(self, opt, epoch, valid_stats):
        """ Save a resumable checkpoint.
        Args:
            opt (dict): option object
            epoch (int): epoch number
            ### fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        model_state_dict = real_model.state_dict()
        
        checkpoint = {
            'model': model_state_dict,
            'opt': opt,
            'epoch': epoch,
            'optim': self.optimizer,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.xent(), epoch))

    def forward_step(self, q_src, q_src_lengths, 
                     pos_src, pos_src_lengths, neg_src, neg_src_lengths, labels):
        # 2. F-prop all but generator.
        pos_outputs, neg_outputs, enc_final = self.model(q_src, q_src_lengths, 
                     pos_src, pos_src_lengths, neg_src, neg_src_lengths)
        # 3. Compute loss 
        outputs = torch.cat((pos_outputs.unsqueeze(1), 
                             neg.outputs.unsqueeze(1)), dim=-1)
        loss = self.criterion(outputs, labels)

        loss_data = loss.data.clone()
        batch_stats = self._stats(loss_data, outputs.data, labels.data)
        # 
        return batch_stats, loss, output.view(-1)

