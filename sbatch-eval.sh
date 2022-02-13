#!/bin/bash

#SBATCH --job-name eval-
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0-2

#SBATCH -p gpu #,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 13G

#: '
python3 exp_runner.py \
       --mode validate_mesh \
       --checkpoint_path "logs//checkpoints/ckpt_0010000.pth" \
       --extra_config_args 'dataset { images_to_pick = [[0, "default"]] }'
# '
# 132: 15_36
# 134: 25_9
# 130: 
# 200: 

# 036: 17_37

