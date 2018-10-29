#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_kg.dict \
    --turn_num 3 \
    --turn_type dcgm \
    --embedding_size 300 \
    --pre_trained_embedding data/fasttext_vec_for_vocab_{}.90004.300d.npy \
    --rnn_type GRU \
    --hidden_size 512 \
    --num_layers 2 \
    --dropout 0.5 \
    --max_len 35 \
    --h_max_len 35 \
    --c_max_len 35 \
    --r_max_len 35 \
    --min_len 3 \
    --bidirectional \
    --tied \
	--decoder_type luong \
    --decode_type greedy \
    --beam_width 10 \
    --best_n 5 \
    --attn_type general \
    --f_max_len 50 \
    --f_topk 10 \
    --lr 0.001 \
    --max_norm 80.0 \
    --epochs 5 \
    --batch_size 128 \
    --teacher_forcing_ratio 0.7 \
    --seed 7 \
    --device cuda \
    --log_interval 50 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.1 \
    --start_epoch 1 \
    --task train \
    --model_type seq2seq
    # --checkpoint ./models/checkpoint.epoch-1_seq2seq_3_concat.pth

/
