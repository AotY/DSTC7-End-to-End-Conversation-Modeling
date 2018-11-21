#!/usr/bin/env bash
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=5

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_kg.60004.dict \
    --turn_num 4 \
    --min_turn 1 \
    --turn_type self_attn \
    --embedding_size 512 \
    --pre_embedding_size 300 \
    --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \
    --rnn_type GRU \
    --hidden_size 512 \
    --num_layers 2 \
    --encoder_num_layers 2 \
    --decoder_num_layers 2 \
    --dropout 0.5 \
    --c_max_len 35 \
    --r_max_len 35 \
    --min_len 3 \
    --bidirectional \
    --tied \
	--decoder_type luong \
    --decode_type beam_search \
    --beam_width 8 \
    --best_n 3 \
    --f_max_len 10 \
    --f_topk 20 \
    --lr 0.001 \
    --max_grad_norm 1.0 \
    --epochs 15 \
    --batch_size 128 \
    --teacher_forcing_ratio 1.0 \
    --seed 7 \
    --device cuda \
    --log_interval 30 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.0005 \
    --test_split 0.07 \
    --start_epoch 1 \
    --model_type kg \
    --task train \
    --share_embedding \
    # --checkpoint models/checkpoint.epoch-3_seq2seq_4_self_attn.pth \
    # --pre_trained_embedding data/fasttext_vec_for_vocab_seq2seq.60004.300d.npy \

/
