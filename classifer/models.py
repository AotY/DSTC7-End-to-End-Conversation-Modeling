#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
This file is for several classification architectures,
which are universal model for text classification tasks.
"""


from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class SingleEncoder(nn.Module):
	"""
	This is a general architecture for classification tasks that
	only have one input.

	Generally, such frameworks just contains one Encoder and a score function.

	Score function options:
		- Multi-layer Perception (MLP)
		- Logistic Regreesion (LR) (or Multi-classification)
		- Dual, refer to the Dual model
		- Bilinear of Linear Transformation (obj "nn.Linear")
	"""
	def __init__(self, encoder, input_size, hidden_size,
				output_size, dropout=0.0, score_fn_type='MLP'):

		"""
		Args:
			- encoder: an encoder object
			- input_size: the output_size of encoder
			- hidden_size: the dimension of the score function
			- output_size: the class number of targets
		"""

		super(SingleEncoder, self).__init__()

        self.encoder = encoder
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.score_fn_type = score_fn_type
        self.cls_arch = 'SingleEncoder'

        # define score function
        self.score_fn, self.bias  = self.build_classifier()


    def build_classifier(self):
        if self.score_fn_type == 'MLP':
            return (nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size)),
                None)

        elif self.score_fn_type == 'LR':
            weight = nn.Parameter(torch.Tensor(self.input_size))

            bias = nn.Parameter(torch.Tensor(1))
            # initialization
            stdv = 1. / math.sqrt(weight.size(0))
            weight.data.uniform_(-stdv, stdv)
            bias.data.uniform_(-stdv, stdv)

            return weight, bias
        elif self.score_fn_type == 'DUAL':
            return (nn.Linear(self.input_size, self.hidden_size, bias=True),
                    None)
        else:
            raise ValueError("{} is not valid, ".format(self.score_fn_type))

    def forward(self, src, lengths, state=None):
        """
        Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            - src: a source sequence passed to encoder.
            typically for inputs this will be a padded :obj:`LongTensor`
            of size `[len x batch]`. however, may be an image or other generic input depending on encoder.
            - lengths: (:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            state: (:obj: `DecoderState`, optional): encoder initial state

        Returns:
            - decoder outputs [batch * output_size]
            - final decoder state
        """

        raise NotImplementedError

"""
Single Input

"""
class SingleArch(SingleEncoder):
    """
    This is a general architecture for classification tasks
    that only have one input.
    Generally, such frameworks just contains one Encoder and a
    score function.

    Score function options:
        - Multi-layer Perception (MLP)
        - Logistic Regression (LR)
    """

    def forward(self, src, lengths, state=None):
        """
        Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            - src (:obj:`Tensor`): a source sequence passed to encoder.
            typically for inputs this will be a padded :obj:`LongTensor`
            of size `[len x batch]`. however, may be an image or other generic input depending on encoder.
            - lengths (:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            state: (:obj: `DecoderState`, optional): encoder initial state

        Returns:
            - decoder outputs [batch * output_size]
            - final decoder state
        """

        _, batch_size = src.size()
        encoder_final, memory_bank = self.encoder(src, lengths=lengths)

        if not isinstance(encoder_final, tuple):
            encoder_hidden = (encoder_final, )

        encoder_hidden = encoder_hidden[0].contiguous().view(batch_size, -1)

        if self.score_fn_type == 'MLP':
            outputs = self.score_fn(encoder_hidden)
        elif self.score_fn_type == 'LR':
            outputs = torch.mv(encoder_hidden, self.score_fn) + self.bias
        else:
            raise ValueError("{} is not valid for SingArch, ".format(self.score_fn_type))

        return outputs, encoder_final

"""
Double Input

"""

class DoubleArch(SingleEncoder):
    """
    This is a general architecture for classification tasks
    that have two inputs (for example, one is a query, the other
    is the corresponding response. )

    score function options:
        - MLP
        - LR

        : without any extend feature, like bilinear feature, dot feature

    """

    def __init(self, encoder, input_size, hidden_size,
               hidden_size, output_size, dropout=0.0,
               score_fn_type='DUAL'):

        if score_fn_type in ['MLP', 'LR']:
            input_size = 2 * input_size

        super(DoubleArch, self).__init__(encoder, input_size, hidden_size,
                                         output_size, dropout=dropout,
                                         score_fn_type=score_fn_type)

        self.cls_arch = 'DualArch'


    def forward(self, query_src, query_lengths,
                res_src, res_lengths, state=None):

        _, batch_size = query_src.size()
        query_encoder_final, query_memory_bank = self.encoder(query_src, lengths=query_lengths)
        res_encoder_final, res_memory_bank = self.encoder(query_src, lengths=res_lengths)

        if self.score_fn_type == 'DUAL':
            query_transform = self.score_fn_type(query_encoder_final).unsqueeze(1)
            outputs = torch.bmm(query_transform, res_encoder_final.unsqueeze(-1))

        else:
            query_res_con = torch.cat((query_encoder_final, res_encoder_final), dim=1)

            if self.score_fn_type == 'MLP':
                outputs = self.score_fn(query_res_con)
            elif self.score_fn_type == 'LR':
                outputs = torch.mv(query_res_con, self.score_fn) + self.bias
            else:
                raise ValueError("{} is not valid. ".format(self.score_fn_type))

        return outputs.squeeze(), query_encoder_final


class ExtendDoubleArch(SingleEncoder):
    """ This is a general architecture for classification tasks
    that have two inputs (for example, one is a query,
    the other is the corresponding response.).
    Generally, such frameworks just contains one Encoder and
    a score function.

    score function options:
        - MLP
        - LR

    : with any extended features,
    like bilinear feature,
    element-wise dot or substract feature, inner-product, etc.
    """

    def __init__(self, encoder, input_size, hidden_size,
                 output_size, dropout=0.0, score_fn_type='MLP',
                 bilinear_flag=Flase, dot_flag=False, substract_flag=False,
                 inner_prod_flag=False):
        # update the input size of the score function
        new_input_size = 0
        if dot_flag:
            new_input_size += input_size
        if substract_flag:
            new_inupt_size += input_size
        if inner_prod_flag:
            new_input_size += 1
        if bilinear_flag:
            new_input_size += hidden_size

        super(ExtendDoubleArch, self).__init__(encoder, new_input_size, hidden_size,
                                               output_size, dropout=dropout, score_fn_type=score_fn_type)

        self.bilinear_flag = bilinear_flag
        self.dot_flag = dot_flag
        self.substract_flag = substract_flag
        self.inner_prod_flag = inner_prod_flag

        self.cls_arch = 'ExtendDoubleArch'
        if bilinear_flag:
            self.bilinear = nn.Bilinear(input_size, input_size, self.hidden_size)

        def forward(self, query_src, query_lengths, res_src, res_lengths, state=None):
            _, batch = query_src.size()
            # encoder query and res
            query_encoder_final, query_memory_bank = self.encoder(query_src, lengths=query_lengths)
            res_encoder_final, res_memory_bank = self.encoder(res_src, lengths=res_lengths)

            if not isinstance(query_encoder_final, tuple):
                query_encoder_hidden = (query_encoder_final, )
                res_encoder_final = (res_encoder_final, )
            else:
                query_encoder_hidden = query_encoder_final
                res_encoder_hidden = res_encoder_final

            query_encoder_hidden = query_enocder_hidden[0].contiguous().view(batch_size, -1)
            res_encoder_hidden = res_encoder_hidden[0].contiguous().view(batch_size, -1)

            # get extended features
            extended_feats = []
            if self.substract_flag:
                substract_feat = query_encoder_hidden - res_encoder_hidden
                extended_feats.append(substract_feat)
            if self.dot_flag:
                dot_feat = torch.mul(query_encoder_hidden, res_encoder_hidden)
                extended_feats.append(dot_feat)
            if self.bilinear_flag:
                bilinear_feat = self.bilinear(query_encoder_hidden, res_encoder_hidden)
                extended_feats.append(bilinear_feat)
            if self.inner_prod_flag:
                inner_prob_feat = torch.bmm(query_encoder_hidden.unsqueeze(1),
                                       res_encoder_hidden.unsqueeze(2))
                extended_feats.append(inner_prob_feat.squeeze(-1))

            # concatenate features
            combine_feats = torch.cat(extended_feats, dim=-1)

            if self.score_fn_type == 'MLP':
                outputs = self.score_fn(combine_feats)
            elif self.score_fn_type == 'LR':
                outputs = torch.mv(combine_feats, self.score_fn) + self.bias
            else:
                raise ValueError("{} is not valid for SingleArch, ".format(self.score_fn_type))

            return outputs.squeeze(), query_encoder_final


        """
        Pairwise or Listwise Ranker
        """

        class PairwiseRanker(nn.Module):
            """
            Core trainable object in OpenNMT. Implements a trainble interface
            for a simple, generic encoder + decoder model.

            Args:
                encoder: (:obj:`EncoderBase`): an encoder object
                classifier MLP or other type

            """

            def __init__(self, encoder, input_size, hidden_size,
                         output_size, score_fn_type="MLP"):
                super(PairwiseRanker, self).__init__()

                self.encoder = encoder
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.ranker_arch = 'pairwise'
                self.score_fn_type = score_fn_type

                # define score function

                self.score_fn, self.biar = self.build_classifier()

            def build_classifier(self):
                if self.score_fn_type == 'MLP':
                    return (nn.Sequential(
                        nn.Linear(self.input_size, self.hidden_size),
                        nn.Tanh(),
                        nn.Linear(self.hidden_size, self.output_size)),
                    None)
                elif self.score_fn_type == 'LR':
                    weight = nn.Parameter(torch.Tensor(self.input_size))
                    bias = nn.Parameter(torch.Tensor(1))
                    # initialization
                    stdv = 1. / math.sqrt(weight.size(0))
                    weight.data.uniform_(-stdv, stdv)
                    bias.data.uniform_(-stdv, stdv)
                    return weight, bias
                elif self.score_fn_type == 'DUAL':
                    return (nn.Linear(self.input_size, self.hidden_size, bias=True),
                            None)
                else:
                    raise ValueError("{} is not valid. ".format(self.score_fn_type))


            def forward(self, query_src, query_lengths, pos_src, pos_lengths,
                        neg_src, neg_lengths, state=None):
                """
                Forward propagate a `src` ant `tgt` pair for training.
                Possible initialized with a beginning decoder state.

                Args:
                    - src (:obj:`Tensor`): a source sequence passed to encoder.
                    typically for inputs this will be a padded :obj:`LongTensor`
                    of size `[len x batch]`. however, may be an
                    image or other generic input depending on encoder.
                    - tgt (:obj:`LongTensor`): a target sequence of size `[tgt_len x batch]`.
					- lengths: (:obj:`LongTensor`): the src lengths,
                    pre-padding `[batch]`.
					- state: (:obj:`DecoderState`, optional): encoder initial
                    state
				"""
				_, batch_size = query_src.size()
                # embedding queries and responses.
                query_encoder_final, query_memory_bank = self.encoder(query_src, lengths=query_lengths)
                pos_encoder_final, pos_memory_bank = self.encoder(pos_src, lengths=pos_lengths)
                neg_enc_final, neg_memory_bank = self.encoder(neg_src, lengths=neg_lengths)

                if not isinstance(query_encoder_final, tuple):
                    query_encoder_hidden = (query_encoder_final, )
                    pos_encoder_hidden = (pos_encoder_final, )
                    neg_encoder_hidden = (neg_encoder_final, )
                else:
                    query_encoder_hidden = query_encoder_final
                    pos_encoder_hidden = pos_encoder_final
                    neg_encoder_hidden = neg_encoder_final

                #
                query_encoder_hidden = query_encoder_hidden[0].contiguous().view(batch_size, -1)
                pos_encoder_hidden = pos_encoder_hidden[0].contiguous().view(batch_size, -1)
                neg_encoder_hidden = neg_encoder_hidden[0].contiguous().view(bach_size, -1)

                # compute the relevance between queries and responses
                if self.score_fn_type == 'DUAL':
                    query_transform = self.score_fn(query_encoder_hidden).unsqueeze(1)
                    pos_outputs = torch.bmm(query_transform, pos_encoder_hidden.unsqueeze(-1))
                    neg_outputs = torch.bmm(query_transform, neg_encoder_hidden.unsqueeze(-1))
                else:
                    query_pos_con = torch.cat((query_encoder_hidden, pos_encoder_hidden), dim=-1)
                    query_neg_con = torch.cat((query_encoder_hidden, neg_encoder_hidden), dim=-1)

                    if self.score_fn_type == 'MLP':
                        pos_outputs = self.score_fn(query_pos_con)
                        neg_outputs = self.score_fn(query_neg_con)
                    elif self.score_fn_type == 'LR':
                        pos_outputs = torch.mv(query_pos_con, self.score_fn) + self.bias
                        neg_outputs = torch.mv(query_neg_con, self.score_fn) + self.bias
                    else:
                        raise ValueError("{} is not valid, ".format(self.score_fn_type))
                        
            assert self.output_size == 1

            pos_outputs = torch.clamp(F.sigmoid(pos_outputs), 1e-7, 1.0-1e-7)
            neg_outputs = torch.clamp(F.sigmoid(neg_outputs), 1e-7, 1.0-1e-7)

            return pos_outputs, neg_outputs, query_encoder_final






