#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=5

python train_kg_model.py \
    --pair_path data/conversations_responses.pair.txt \
    --save_path data/ \
    --vocab_path data/vocab_word2idx_seq2seq.dict \
    --turn_num 3 \
    --turn_type none \
    --embedding_size 300 \
    --pre_trained_embedding data/fasttext_vec_for_vocab_seq2seq.50004.300d.npy \
    --fasttext_vec /home/taoqing/Research/data/crawl-300d-2M-subword.vec.bin \
    --rnn_type GRU \
    --hidden_size 512 \
    --num_layers 2 \
    --dropout 0.5 \
    --max_len 35 \
    --h_max_len 35 \
    --c_max_len 35 \
    --r_max_len 35 \
    --min_len 3 \
    --bidirectional \
    --tied \
	--decoder_type bahdanau \
    --decode_type beam_search \
    --beam_width 64 \
    --best_n 5 \
    --attn_type concat \
    --f_max_len 50 \
    --f_topk 10 \
    --lr 0.0005 \
    --max_norm 80.0 \
    --epochs 7 \
    --batch_size 128 \
    --teacher_forcing_ratio 1.0 \
    --seed 7 \
    --device cuda \
    --log_interval 100 \
    --log_path ./logs/{}_{}_{}_{}.log \
    --model_path ./models \
    --eval_split 0.1 \
    --start_epoch 1 \
    --task train \
    --model_type kg
    # --checkpoint ./models/checkpoint.epoch-3_seq2seq_3_dcgm.pth

/
