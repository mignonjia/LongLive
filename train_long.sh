#!/bin/bash

# Project path and config
CONFIG=configs/longlive_train_long.yaml
LOGDIR=logs_long_from_longlive_base_t129
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

torchrun \
  --nproc_per_node=8 \
  --master_port=29502 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --run-name long_sf_from_longlive_base \
  --no-one-logger
