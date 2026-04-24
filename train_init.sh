# #!/bin/bash

# Project path and config
CONFIG=configs/longlive_train_init.yaml
LOGDIR=logs
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

# CUDA_VISIBLE_DEVICES=4,5,6,7 
torchrun \
  --nproc_per_node=8 \
  --master_port=29506 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --run-name train_vanilla_sf \
  --no-one-logger
