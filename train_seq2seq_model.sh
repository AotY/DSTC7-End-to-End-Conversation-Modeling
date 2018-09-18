#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5,6,7

python train_seq2seq_model.py \
    --conversations_num_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/train.conversations.num.txt \
    --responses_num_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/train.responses.num.txt \
    --save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/ \
    --vocab_save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/vocab_word2idx.dict \
    --dialog_encoder_hidden_size 300 \
    --dialog_encoder_num_layers 2 \
    --dialog_encoder_rnn_type RNN \
    --dialog_encoder_dropout_rate 0.8 \
    --dialog_encoder_max_length 32 \
    --dialog_encoder_clip_grads 1 \
    --dialog_encoder_bidirectional \
    --dialog_encoder_pretrained_embedding_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/google_vec_for_vocab.80004.300d.npy \
    --dialog_decoder_hidden_size 300 \
    --dialog_decoder_num_layers 2 \
    --dialog_decoder_rnn_type RNN \
    --dialog_decoder_dropout_rate 0.8 \
    --dialog_decoder_max_length 32 \
    --dialog_decoder_clip_grads 1 \
    --dialog_decoder_bidirectional \
    --dialog_decoder_pretrained_embedding_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/google_vec_for_vocab.80004.300d.npy \
    --dialog_decoder_attention_type dot \
    --lr 0.001 \
    --epochs 5 \
    --batch_size 128 \
    --use_teacher_forcing \
    --teacher_forcing_ratio 0.5 \
    --seed 7 \
    --device cuda \
    --log_interval 200 \
    --save ./models/seq2seq.model.pt \
    --test_split 0.2 \
    --optim_method adam \


/