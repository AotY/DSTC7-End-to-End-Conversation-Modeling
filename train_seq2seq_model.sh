#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=6

python train_seq2seq_model.py \
    --path_conversations_responses_pair /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/conversations_responses.pair.txt \
    --save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/ \
    --vocab_save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/vocab_word2idx_seq2seq.dict \
    --dialog_encoder_embedding_size 300 \
    --dialog_encoder_hidden_size 512 \
    --dialog_encoder_num_layers 2 \
    --dialog_encoder_rnn_type LSTM \
    --dialog_encoder_dropout_probability 0.5 \
    --dialog_encoder_max_length 32 \
    --dialog_encoder_clipnorm 50.0 \
    --dialog_encoder_bidirectional \
    --dialog_encoder_pretrained_embedding_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/fasttext_vec_for_vocab_seq2seq.50004.300d.npy \
    --dialog_encoder_tied \
    --dialog_decoder_embedding_size 300 \
    --dialog_decoder_hidden_size 512 \
    --dialog_decoder_num_layers 2 \
    --dialog_decoder_rnn_type LSTM \
    --dialog_decoder_dropout_probability 0.5 \
    --dialog_decoder_max_length 32 \
    --dialog_decoder_clipnorm 50.0 \
    --dialog_decoder_pretrained_embedding_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/fasttext_vec_for_vocab_seq2seq.50004.300d.npy \
    --dialog_decoder_attention_type general \
    --lr 0.005 \
    --epochs 7 \
    --batch_size 256 \
    --teacher_forcing_ratio 0.5 \
    --seed 7 \
    --device cuda \
    --log_interval 100 \
    --log_file ./logs/train_seq2seq_model_{}.log \
    --model_save_path ./models \
    --eval_split 0.2 \
    --optim_method adam \
    --start_epoch 1 \
    --train_or_eval train \
    # --checkpoint ./models/checkpoint.epoch-5.pth  
    # --dialog_decoder_tied \

    /
