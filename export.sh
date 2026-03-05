# # vitn
# python export_from_demo.py --restore_ckpt results/train/defomstereo_custom_vitn_sceneflow/defomstereo_custom_vitn_sceneflow_200000.pth --dinov2_encoder vitn --corr_radius 3 --n_gru_layers 2

python export.py \
  --restore_ckpt checkpoints/defomstereo_vits_sceneflow.pth \
  --dinov2_encoder vits \
  --corr_radius 4 \
  --n_gru_layers 3 \
  --input_size 224 320

# new
# python export.py \
#   --restore_ckpt results/train/defomstereo_custom_vitn_sceneflow/defomstereo_custom_vitn_sceneflow_200000.pth \
#   --dinov2_encoder vitn \
#   --corr_radius 3 \
#   --n_gru_layers 2 \
#   --input_size 200 320 \
#   --valid_iters 8 \
#   --scale_iters 2
