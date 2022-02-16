#!/bin/bash

# Finetune a 'metamodel'.

#SBATCH --job-name gonzalo_100_layers1x_2GPU_ftTo132-3_initAvg
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0-4

#SBATCH -p htc,gpu #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#: '
CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
cp confs/gonzalo_finetune_allLayers.conf $CONF2

PORT=26112
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=1 exp_runner.py --mode train \
--conf $CONF1
torchrun --rdzv_id $PORT --rdzv_endpoint 127.0.0.1:$PORT --nnodes=1 --nproc_per_node=1 exp_runner.py --mode train \
--conf $CONF2
# '

