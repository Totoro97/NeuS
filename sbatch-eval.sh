#!/bin/bash

# Extract mesh or render a video (set `--mode`).
# Use the latest checkpoint in $DIR.
# Ideally, $PORT should be different at each run.

#SBATCH --job-name eval-132
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 1:30:0

#SBATCH -p htc,gpu #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#: '
# Set these
DIR=logs/gonzalo_100_layers1x_2GPU_ftTo132-3_initAvg
PORT=23117

# Don't set these
LATEST_CKPT=`ls $DIR/checkpoints | tail -1`
NPROC=1
torchrun \
       --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py \
       --mode validate_mesh \
       --checkpoint_path $DIR/checkpoints/$LATEST_CKPT \
       --extra_config_args 'dataset { images_to_pick = [[0, "default"]] }'
# '

# 018: interpolation_9_31
# 019: interpolation_8_28
# 132: interpolation_15_36
# 134: interpolation_25_9
# 130: 
# 200: 

# 036: interpolation_17_37

