#!/usr/bin/env bash
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=5

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_kg.60004.dict \
    --vocab_size 6e4 \
    --turn_num 4 \
    --min_turn 1 \
    --turn_type self_attn \
    --rnn_type GRU \
    --embedding_size 512 \
    --hidden_size 512 \
    --latent_size 0 \
    --num_layers 2 \
    --encoder_num_layers 2 \
    --decoder_num_layers 2 \
    --cnn_kernel_width 3 \
    --dropout 0.2 \
    --c_max_len 35 \
    --r_max_len 35 \
    --min_len 3 \
    --bidirectional \
    --tied \
	--decoder_type luong \
    --decode_type beam_search \
    --beam_size 10 \
    --best_n 3 \
    --f_max_len 35 \
    --f_topk 5 \
    --lr 0.005 \
    --max_grad_norm 5.0 \
    --epochs 15 \
    --batch_size 128 \
    --teacher_forcing_ratio 1.0 \
    --seed 19 \
    --device cuda \
    --log_interval 20 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.0005 \
    --test_split 0.07 \
    --start_epoch 1 \
    --model_type kg \
    --task train \
    --share_embedding \
    --offline_type elastic \
    # --pre_embedding_size 300 \
    # --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \
    # --pre_trained_embedding \
    # --checkpoint models/checkpoint.epoch-1_kg_5_self_attn.pth \

/
