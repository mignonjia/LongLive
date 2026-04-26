#!/bin/bash

# Project path and config
CONFIG=configs/longlive_train_adaptive.yaml
LOGDIR=logs_long_adaptive
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG
# 2178059 hao.zhang@fs-mbz-gpu-368
torchrun \
  --nproc_per_node=8 \
  --master_port=29508 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --run-name longlive_adaptive \
  --no-one-logger
