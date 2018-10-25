#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_{}.dict \
    --turn_num 3 \
    --turn_type concat \
    --embedding_size 300 \
    --pre_trained_embedding data/fasttext_vec_for_vocab_{}.70004.300d.npy \
    --hidden_size 512 \
    --num_layers 1 \
    --dropout 0.5 \
    --max_len 35 \
    --min_len 3 \
    --bidirectional \
    --tied \
	--decoder_type normal \
    --decode_type greedy \
    --beam_width 10 \
    --best_n 5 \
    --attn_type general \
    --fact_max_len 50 \
    --fact_topk 20 \
    --lr 0.001 \
    --max_norm 100.0 \
    --epochs 5 \
    --batch_size 128 \
    --teacher_forcing_ratio 1.0 \
    --seed 7 \
    --device cuda \
    --log_interval 50 \
    --log_path ./logs/train_{}_model_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.1 \
    --start_epoch 1 \
    --task train \
    --model_type seq2seq \
    --checkpoint ./models/checkpoint.epoch-1_seq2seq_3_concat.pth

/
