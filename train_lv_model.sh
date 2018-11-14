#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

python train_lv_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_seq2seq.60004.dict \
    --turn_num 4 \
    --min_turn 2 \
    --turn_type self_attn \
    --embedding_size 512 \
    --pre_embedding_size 300 \
    --fasttext_vec /home/taoqing/Research/data/wiki-news-300d-1M-subword.vec.bin \
    --rnn_type GRU \
    --hidden_size 512 \
    --latent_size 128 \
    --kl_anneal logistic \
    --kla_denom 10 \
    --kla_k 0.00025 \
    --kla_x0 15000 \
    --num_layers 2 \
    --encoder_num_layers 2 \
    --decoder_num_layers 2 \
    --dropout 0.6 \
    --c_max_len 35 \
    --r_max_len 35 \
    --min_len 3 \
    --bidirectional \
    --tied \
	--decoder_type luong \
    --decode_type beam_search \
    --beam_width 32 \
    --best_n 10 \
    --attn_type concat \
    --f_max_len 50 \
    --f_topk 7 \
    --lr 0.001 \
    --n_warmup_steps 3000 \
    --max_norm 20.0 \
    --epochs 15 \
    --batch_size 128 \
    --teacher_forcing_ratio 1.0 \
    --seed 7 \
    --device cuda \
    --log_interval 30 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./lv_models \
    --eval_split 0.0007 \
    --test_split 0.06 \
    --start_epoch 1 \
    --task train \
    --model_type seq2seq \
    --share_embedding
    # --checkpoint lv_models/checkpoint.epoch-7_seq2seq_4_sum.pth
    # --pre_trained_embedding data/fasttext_vec_for_vocab_seq2seq.60004.300d.npy \
    # --h_max_len 35 \
    # --max_len 35 \

/
