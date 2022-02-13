#!/bin/bash

#SBATCH --job-name gonzalo_100_faster_layers2x
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 6-0

#SBATCH -p gpu #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 70G

#: '
CONF1=`mktemp`
cp confs/gonzalo_100.conf $CONF1

#-m cProfile -o 100_100iters.prof
#nvprof --profile-from-start off -o 100_50scenes.nvprof
torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port 25104 exp_runner.py --mode train \
--checkpoint_path logs/gonzalo_100_faster_layers2x/checkpoints/ckpt_0320000.pth \
--extra_config_args 'dataset { batch_size = ${train.batch_size} }, train { batch_size = 768 }'
#--conf $CONF1
# '

