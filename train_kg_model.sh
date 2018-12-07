#!/usr/bin/env bash
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=6

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_kg.40004.dict \
    --vocab_size 4e4 \
    --turn_num 5 \
    --min_turn 2 \
    --turn_type normal_sum \
    --rnn_type GRU \
    --embedding_size 512 \
    --hidden_size 512 \
    --num_layers 2 \
    --encoder_num_layers 2 \
    --decoder_num_layers 2 \
    --dropout 0.0 \
    --c_max_len 50 \
    --r_max_len 50 \
    --min_len 3 \
    --bidirectional \
    --tied \
	--decoder_type luong \
    --decode_type beam_search \
    --beam_size 8 \
    --best_n 3 \
    --f_max_len 120 \
    --f_topk 5 \
    --lr 0.001 \
    --max_grad_norm 10.0 \
    --epochs 15 \
    --batch_size 200 \
    --teacher_forcing_ratio 1.0 \
    --seed 19 \
    --device cuda \
    --eval_interval 90 \
    --log_interval 15 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.0001 \
    --test_split 0.07 \
    --start_epoch 1 \
    --model_type seq2seq \
    --task train \
    --share_embedding \
    --offline_type elastic \
    # --checkpoint models/epoch-1_seq2seq_5_none_2018_12_05_13:45.pth \
    # --smoothing \
    # --pre_embedding_size 300 \
    # --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \
    # --pre_trained_embedding \

/
