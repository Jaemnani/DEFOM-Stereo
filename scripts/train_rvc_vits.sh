#!/usr/bin/env bash

# 2 x 24GB 3090/4090 GPUs 환경에 최적화
# 초기 모델: results/train/defomstereo_vitn_sceneflow_test01/checkpoint_latest.pth (또는 최종 step 파일)

################################################################################
# Stage 1: RVC Pretrain 1 (다양한 합성/실제 데이터 혼합 시작)
################################################################################
CHECKPOINT_DIR=results/train/defomstereo_vitn_rvc_pretrain && \
mkdir -p ${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9991 train_stereo.py \
--distributed \
--launcher pytorch \
--gpu_ids 0 1 \
--name defomstereo_vitn_rvc_pretrain \
--batch_size 4  \
--num_workers 8  \
--train_datasets tartan_air sceneflow irs 3dkenburns crestereo falling_things sintel_stereo vkitti2 carla_highres \
--train_folds 1 1 1 1 1 1 3 3 80 \
--num_steps 200000 \
--n_downsample 2 \
--train_iters 18 \
--scale_iters 8 \
--idepth_scale 0.5 \
--corr_levels 2 \
--corr_radius 4 \
--scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
--scale_corr_radius 2 \
--dinov2_encoder vitn \
--image_size 384 768 \
--lr 0.00005 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
# --resume_ckpt results/train/defomstereo_custom_vitn_sceneflow/checkpoint_latest.pth \

################################################################################
# Stage 2: RVC Pretrain 2 (실제 환경 데이터셋 추가 및 비중 확대)
################################################################################
# CHECKPOINT_DIR=results/train/defomstereo_vitn_rvc_pretrain2 && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=9992 train_stereo.py \
# --distributed \
# --launcher pytorch \
# --gpu_ids 0 1 \
# --name defomstereo_vitn_rvc_pretrain2 \
# --batch_size 4  \
# --num_workers 8  \
# --train_datasets tartan_air irs 3dkenburns crestereo vkitti2 carla_highres kitti12 kitti15 middlebury_2005 middlebury_2006 middlebury_2014 middlebury_2021 middlebury_Q middlebury_H eth3d instereo2k booster \
# --train_folds 1 1 1 1 3 30 100 100 200 200 200 200 200 200 1000 20 10 \
# --num_steps 100000 \
# --n_downsample 2 \
# --train_iters 18 \
# --scale_iters 8 \
# --idepth_scale 0.5 \
# --corr_levels 2 \
# --corr_radius 4 \
# --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
# --scale_corr_radius 2 \
# --dinov2_encoder vitn \
# --image_size 384 768 \
# --lr 0.00005 \
# --resume_ckpt results/train/defomstereo_vitn_rvc_pretrain/checkpoint_latest.pth \
# --no_resume_optimizer \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

################################################################################
# Stage 3: RVC Final (특정 데이터셋 정밀 튜닝)
################################################################################
# CHECKPOINT_DIR=results/train/defomstereo_vitn_rvc && \
# mkdir -p ${CHECKPOINT_DIR} && \
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=9993 train_stereo.py \
# --distributed \
# --launcher pytorch \
# --gpu_ids 0 1 \
# --name defomstereo_vitn_rvc \
# --batch_size 4  \
# --num_workers 8  \
# --train_datasets tartan_air irs 3dkenburns crestereo vkitti2 carla_highres kitti12 kitti15 middlebury_2005 middlebury_2006 middlebury_2014 middlebury_2021 middlebury_Q middlebury_H eth3d instereo2k booster \
# --train_folds 1 1 1 1 3 30 2500 2500 200 200 200 200 200 200 1000 20 10 \
# --num_steps 20000 \
# --n_downsample 2 \
# --train_iters 18 \
# --scale_iters 8 \
# --idepth_scale 0.5 \
# --corr_levels 2 \
# --corr_radius 4 \
# --scale_list 0.125 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
# --scale_corr_radius 2 \
# --dinov2_encoder vitn \
# --image_size 384 768 \
# --lr 0.00005 \
# --resume_ckpt results/train/defomstereo_vitn_rvc_pretrain2/checkpoint_latest.pth \
# --no_resume_optimizer \
# 2>&1 | tee -a ${CHECKPOINT_DIR}/train.log