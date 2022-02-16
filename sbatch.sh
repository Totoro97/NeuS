#!/bin/bash

# Train a 'metamodel' or a single-scene model.

#SBATCH --job-name 134
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 3-0

#SBATCH -p gpu #_a100 #,gpu_devel
#SBATCH --gres gpu:2
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 70G

#: '
CONF=`mktemp`
cp confs/gonzalo_100.conf $CONF1

PORT=25119
NPROC=2
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=$NPROC exp_runner.py --mode train \
--conf $CONF
#--checkpoint_path logs/100_rank50/checkpoints/ckpt_0110000.pth \
#--extra_config_args 'general { base_exp_dir = ./logs/100_rank50_restart/ }, dataset { batch_size = ${train.batch_size} }, train { batch_size = 1024, learning_rate = 0.8e-3, warm_up_end = 1000 }'
# '

