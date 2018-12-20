#!/usr/bin/env bash
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=5

python train.py \
    --pair_path data/train2.convos.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx.60004.dict \
    --c_max 3 \
    --c_min 1 \
    --enc_type qc \
    --rnn_type GRU \
    --embedding_size 512 \
    --hidden_size 512 \
    --encoder_num_layers 2 \
    --decoder_num_layers 2 \
    --t_num_layers 1 \
    --num_heads 1 \
    --transformer_size 512 \
    --inner_hidden_size 1024 \
    --k_size 64 \
    --v_size 64 \
    --dropout 0.1 \
    --bidirectional \
    --tied \
    --decode_type beam_search \
    --q_max_len 60 \
    --c_max_len 55 \
    --r_max_len 38 \
    --f_max_len 120 \
    --min_len 3 \
    --beam_size 8 \
    --best_n 3 \
    --f_topk 15 \
    --lr 0.001 \
    --max_grad_norm 5.0 \
    --epochs 25 \
    --batch_size 64 \
    --teacher_forcing_ratio 1.0 \
    --seed 23 \
    --device cuda \
    --eval_interval 1800 \
    --log_interval 100 \
    --lr_patience 3 \
    --es_patience 8 \
    --log_path ./logs/{}_{}_{}_{}_{}.log \
    --model_path ./models \
    --test_split 0.08 \
    --eval_batch 12 \
    --start_epoch 1 \
    --model_type kg \
    --task train \
    --share_embedding \
    --offline_type elastic \
    # --checkpoint models/kg_qc_9_1_3_2018_12_17_10:48.pth \
    # --label_smoothing \
    # --pre_embedding_size 300 \
    # --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \ 
    # --pre_trained_embedding \

/
