#!/bin/bash

#SBATCH --job-name 100_rank50
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 5-0

#SBATCH -p gpu #,gpu_devel
#SBATCH --gres gpu:2
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 70G

#: '
CONF1=`mktemp`
cp confs/gonzalo_100.conf $CONF1

#-m cProfile -o 100_100iters.prof
#nvprof --profile-from-start off -o 100_50scenes.nvprof
PORT=25109
NPROC=2
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC --master_port $PORT exp_runner.py --mode train \
--conf $CONF1
#--checkpoint_path logs/gonzalo_100_rank10/checkpoints/ckpt_0320000.pth \
#--extra_config_args 'dataset { batch_size = ${train.batch_size} }, train { batch_size = 768 }'
# '

