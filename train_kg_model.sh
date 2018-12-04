#!/usr/bin/env bash
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=5

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_kg.40004.dict \
    --vocab_size 4e4 \
    --turn_num 5 \
    --min_turn 2 \
    --turn_type normal \
    --rnn_type GRU \
    --embedding_size 512 \
    --hidden_size 512 \
    --num_layers 2 \
    --encoder_num_layers 2 \
    --decoder_num_layers 2 \
    --dropout 0.2 \
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
    --lr 0.0003 \
    --max_grad_norm 10.0 \
    --epochs 15 \
    --batch_size 200 \
    --teacher_forcing_ratio 1.0 \
    --seed 19 \
    --device cuda \
    --log_interval 50 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.0001 \
    --test_split 0.07 \
    --start_epoch 1 \
    --model_type kg \
    --task train \
    --share_embedding \
    --offline_type elastic \
    # --checkpoint models/epoch-3_kg_5_self_attn_2018_11_30_15:27.pth \
    # --pre_embedding_size 300 \
    # --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \
    # --pre_trained_embedding \

/
