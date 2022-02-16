#!/bin/bash

# Continue training from a checkpoint.
# Suitable for both meta-learning and fine-tuning.

#SBATCH --job-name 100_rank10
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 1-0

#SBATCH -p gpu,htc #_a100 #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 73G

#: '
# Set these
DIR=logs/100_rank10_
PORT=27110
NPROC=1

# Don't set these
LATEST_CKPT=`ls -t $DIR/checkpoints | head -1` 

torchrun \
      --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
      --checkpoint_path $DIR/checkpoints/$LATEST_CKPT \
      --extra_config_args 'dataset { batch_size = ${train.batch_size} }, train { batch_size = 1024 }'
#      --extra_config_args {#'train { learning_rate_reduce_steps = [], end_iter = 30000, warm_up_end = 0 }'
# '

