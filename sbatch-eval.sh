#!/bin/bash

#SBATCH --job-name eval-019
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0-2

#SBATCH -p htc,gpu #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#: '
PORT=23102
NPROC=1
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC --master_port $PORT exp_runner.py \
       --mode interpolate_8_28 \
       --checkpoint_path logs/019_2022-02-14/checkpoints/ckpt_0220000.pth \
       --extra_config_args 'dataset { images_to_pick = [[0, "default"]] }'
# '

# 018: interpolation_9_31
# 019: interpolation_8_28
# 132: interpolation_15_36
# 134: interpolation_25_9
# 130: 
# 200: 

# 036: interpolation_17_37

