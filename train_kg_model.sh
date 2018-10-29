#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=6

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_seq2seq.dict \
    --turn_num 3 \
    --turn_type dcgm \
    --embedding_size 300 \
    --pre_trained_embedding data/fasttext_vec_for_vocab_{}.50004.300d.npy \
    --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \
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
    --attn_type dot \
    --f_max_len 50 \
    --f_topk 20 \
    --lr 0.001 \
    --max_norm 80.0 \
    --epochs 7 \
    --batch_size 128 \
    --teacher_forcing_ratio 1.0 \
    --seed 7 \
    --device cuda \
    --log_interval 50 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.1 \
    --start_epoch 1 \
    --task train \
    --model_type seq2seq
    # --checkpoint ./models/checkpoint.epoch-1_kg_3_dcgm.pth

/
