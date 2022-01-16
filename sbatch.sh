#!/bin/bash

#SBATCH --job-name gonzalo_multi_skip_noVD_scenewiseLr10
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 1-0

#SBATCH -p gpu_a100
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 15G

python3 exp_runner.py --mode train --conf confs/gonzalo_multi.conf

# dataset { data_dirs = ["./datasets/Gonzalo/132/2021-07-22-12-28-42/portrait_reconstruction/",], images_to_pick = [] },
#python3 exp_runner.py \
#	--mode interpolate_15_36 \
#	--extra_config_args 'dataset { images_to_pick = [] }, model { neus_renderer { n_outside = 0 } }' \
#	--checkpoint_path ./logs/132_noVD/checkpoints/ckpt_0110000.pth

