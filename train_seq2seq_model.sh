#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5,6,7

python train_seq2seq_model.py \
    --path_conversations_responses_pair /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/conversations_responses.pair.txt \
    --save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/ \
    --vocab_save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/vocab_word2idx.dict \
    --dialog_encoder_embedding_size 300 \
    --dialog_encoder_hidden_size 300 \
    --dialog_encoder_num_layers 2 \
    --dialog_encoder_rnn_type LSTM \
    --dialog_encoder_dropout_rate 0.8 \
    --dialog_encoder_max_length 50 \
    --dialog_encoder_clipnorm 50.0 \
    --dialog_encoder_clipvalue 0.5 \
    --dialog_encoder_bidirectional \
    --dialog_encoder_pretrained_embedding_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/fasttext_vec_for_vocab.50004.300d.npy \
    --dialog_encoder_tied \
    --dialog_decoder_embedding_size 300 \
    --dialog_decoder_hidden_size 300 \
    --dialog_decoder_num_layers 2 \
    --dialog_decoder_rnn_type LSTM \
    --dialog_decoder_dropout_rate 0.8 \
    --dialog_decoder_max_length 50 \
    --dialog_decoder_clipnorm 50.0 \
    --dialog_decoder_clipvalue 0.5 \
    --dialog_decoder_bidirectional \
    --dialog_decoder_pretrained_embedding_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/fasttext_vec_for_vocab.50004.300d.npy \
    --dialog_decoder_attention_type general \
    --dialog_decoder_tied \
    --lr 0.001 \
    --epochs 5 \
    --batch_size 128 \
    --teacher_forcing_ratio 0.5 \
    --seed 7 \
    --device cuda \
    --log_interval 50 \
    --log_file ./logs/train_seq2seq_model_{}.log \
    --model_save_path ./models \
    --test_split 0.2 \
    --optim_method adam \
    --batch_per_load 1 \
    --start_epoch 1 \
    --train_or_eval train \
    --checkpoint ./models/checkpoint.epoch-4.pth \ 


    /
