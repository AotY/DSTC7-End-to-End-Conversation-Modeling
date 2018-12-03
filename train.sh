#!/usr/bin/env bash
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=5

python train.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_kg.40004.dict \
    --vocab_size 4e4 \
    --turn_num 5 \
    --min_turn 2 \
    --rnn_type GRU \
    --embedding_size 512 \
    --hidden_size 512 \
    --bidirectional \
    --r_num_layers 2 \
    --t_num_layers 6 \
    --transformer_size 512 \
    --inner_hidden_size 2048 \
    --k_size 64 \
    --v_size 64 \
    --num_heads 8 \
    --n_warmup_steps 4000 \
    --dropout 0.1 \
    --c_max_len 50 \
    --r_max_len 50 \
    --min_len 3 \
    --tied \
    --beam_size 8 \
    --n_best 3 \
    --f_max_len 100 \
    --f_topk 5 \
    --lr 0.0002 \
    --epochs 15 \
    --batch_size 128 \
    --decode_type beam_search \
    --max_grad_norm 5.0 \
    --teacher_forcing_ratio 1.0 \
    --seed 19 \
    --device cuda \
    --log_interval 20 \
    --log_path ./logs/transformer_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.0005 \
    --test_split 0.07 \
    --start_epoch 1 \
    --model_type kg \
    --task train \
    --share_embedding \
    --offline_type elastic \
    # --checkpoint models/epoch-1_kg_4_self_attn_2018_11_26_21:55.pth \
    # --pre_embedding_size 300 \
    # --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \
    # --pre_trained_embedding \

/
