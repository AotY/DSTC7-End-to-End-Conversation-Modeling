#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5,6,7

python train_seq2seq_model.py \
    --path_conversations_responses_pair /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/conversations_responses.pair.txt \
    --save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/ \
    --vocab_save_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/vocab_word2idx_seq2seq.dict \
    --dialogue_encoder_embedding_size 300 \
    --dialogue_encoder_hidden_size 512 \
    --dialogue_encoder_num_layers 4 \
    --dialogue_encoder_rnn_type LSTM \
    --dialogue_encoder_dropout_probability 0.5 \
    --dialogue_encoder_max_length 35 \
    --dialogue_encoder_bidirectional \
    --dialogue_encoder_pretrained_embedding_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/fasttext_vec_for_vocab_seq2seq.60004.300d.npy \
    --dialogue_decoder_embedding_size 300 \
    --dialogue_decoder_hidden_size 512 \
    --dialogue_decoder_num_layers 4 \
    --dialogue_decoder_rnn_type LSTM \
    --dialogue_decoder_dropout_probability 0.5 \
    --dialogue_decoder_max_length 35 \
    --dialogue_decoder_pretrained_embedding_path /home/taoqing/Research/DSTC7/DSTC7-End-to-End-Conversation-Modeling/data/fasttext_vec_for_vocab_seq2seq.60004.300d.npy \
    --dialogue_decoder_attention_type general \
	--dialogue_decode_type greedy \
    --dialogue_decoder_tied \
    --dialogue_turn_num 1 \
    --beam_width 10 \
    --topk 2 \
    --lr 0.005 \
    --max_norm 100.0 \
    --epochs 15 \
    --batch_size 128 \
    --teacher_forcing_ratio 0.5 \
    --seed 7 \
    --device cuda:0 \
    --log_interval 20 \
    --log_file ./logs/train_seq2seq_model_{}.log \
    --model_save_path ./models \
    --eval_split 0.2 \
    --optim_method adam \
    --start_epoch 1 \
    --train_or_eval train \
    # --checkpoint ./models/checkpoint.epoch-5_seq2seq.pth \

    /
