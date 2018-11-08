#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_seq2seq.dict \
    --turn_num 4 \
    --turn_type hred \
    --embedding_size 512 \
    --pre_embedding_size 300 \
    --fasttext_vec /home/taoqing/Research/data/wiki-news-300d-1M-subword.vec.bin \
    --rnn_type GRU \
    --hidden_size 512 \
    --num_layers 2 \
    --encoder_num_layers 2 \
    --decoder_num_layers 2 \
    --dropout 0.8 \
    --max_len 35 \
    --h_max_len 35 \
    --c_max_len 35 \
    --r_max_len 35 \
    --min_len 3 \
    --bidirectional \
    --tied \
	--decoder_type bahdanau \
    --decode_type beam_search \
    --beam_width 32 \
    --best_n 10 \
    --attn_type concat \
    --f_max_len 50 \
    --f_topk 10 \
    --lr 0.001 \
    --max_norm 50.0 \
    --epochs 10 \
    --batch_size 128 \
    --teacher_forcing_ratio 0.5 \
    --seed 7 \
    --device cuda \
    --log_interval 70 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.0007 \
    --test_split 0.06 \
    --start_epoch 1 \
    --task train \
    --model_type kg \
    --share_embedding \
    # --checkpoint models/checkpoint.epoch-1_seq2seq_4_hred.pth
    # --pre_trained_embedding data/fasttext_vec_for_vocab_seq2seq.60004.300d.npy \

/
