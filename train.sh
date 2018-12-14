#!/usr/bin/env bash
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=6

python train.py \
    --pair_path data/train.convos.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx.60004.dict \
    --c_max 3 \
    --c_min 1 \
    --enc_type qc_seq_h \
    --rnn_type GRU \
    --embedding_size 512 \
    --hidden_size 512 \
    --encoder_num_layers 2 \
    --decoder_num_layers 2 \
    --t_num_layers 2 \
    --transformer_size 512 \
    --inner_hidden_size 1024 \
    --k_size 64 \
    --v_size 64 \
    --num_heads 4 \
    --dropout 0.0 \
    --bidirectional \
    --tied \
    --decode_type beam_search \
    --q_max_len 55 \
    --c_max_len 55 \
    --r_max_len 35 \
    --f_max_len 120 \
    --min_len 3 \
    --beam_size 8 \
    --best_n 3 \
    --f_topk 15 \
    --lr 0.001 \
    --max_grad_norm 15.0 \
    --epochs 15 \
    --batch_size 128 \
    --teacher_forcing_ratio 1.0 \
    --seed 23 \
    --device cuda \
    --eval_interval 1800 \
    --log_interval 90 \
    --lr_patience 2 \
    --es_patience 3 \
    --log_path ./logs/{}_{}_{}_{}_{}.log \
    --model_path ./models \
    --test_split 0.07 \
    --eval_batch 5 \
    --start_epoch 1 \
    --model_type seq2seq \
    --task train \
    --share_embedding \
    --offline_type elastic \
    # --label_smoothing \
    # --checkpoint models/epoch-1_kg_concat_1_4_2018_12_12_18:46.pth \
    # --pre_embedding_size 300 \
    # --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \ 
    # --pre_trained_embedding \

/
