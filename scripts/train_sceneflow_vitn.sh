#!/usr/bin/env bash

# trained on 2 x 24GB 3090/4090 GPUs

# # backbone
# CHECKPOINT_DIR=results/train/defomstereo_custom_vitn_sceneflow && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=9990 train_stereo.py \
# --distributed \
# --launcher pytorch \
# --gpu_ids 0 \
# --name defomstereo_custom_vitn_sceneflow \
# --batch_size 4  \
# --num_workers 8  \
# --train_datasets sceneflow \
# --train_folds 1 \
# --n_downsample 2 \
# --train_iters 18 \
# --scale_iters 4 \
# --idepth_scale 0.5 \
# --corr_levels 2 \
# --corr_radius 3 \
# --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
# --scale_corr_radius 2 \
# --dinov2_encoder vitn \
# --image_size 320 512 \
# --lr 0.0001 \
# --n_gru_layers 2 \
# --num_steps 200000 \
# --save_path ${CHECKPOINT_DIR} \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
# # --num_steps 10 \
# # --save_latest_ckpt_freq 1 \
# # --save_ckpt_freq 1 \
# # --val_freq 1 \
# # --corr_radius 4 \
# # --scale_iters 8 \
# # --mixed_precision \ # overflow발생.

# backbone
CHECKPOINT_DIR=results/train/defomstereo_custom_vitn_sceneflow2 && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9990 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 \
--name defomstereo_custom_vitn_sceneflow \
--batch_size 4  \
--num_workers 8  \
--train_datasets sceneflow \
--train_folds 1 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 4 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vitn \
--lr 0.0001 \
--n_gru_layers 3 \
--num_steps 200000 \
--save_path ${CHECKPOINT_DIR} \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
# --image_size 320 512 \
# --num_steps 10 \
# --save_latest_ckpt_freq 1 \
# --save_ckpt_freq 1 \
# --val_freq 1 \
# --corr_radius 4 \
# --scale_iters 8 \
# --mixed_precision \ # overflow발생.



