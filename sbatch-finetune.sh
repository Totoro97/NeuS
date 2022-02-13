#!/bin/bash

#SBATCH --job-name finetune-
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0-7

#SBATCH -p gpu #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#: '
CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
cp confs/gonzalo_finetune_allLayers.conf $CONF2

PORT=26100
torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port $PORT exp_runner.py --mode train --conf $CONF1
torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port $PORT exp_runner.py --mode train --conf $CONF2
# '

