#!/bin/bash

#SBATCH --job-name rend_WD1e-2_ft
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 2:0:0

#SBATCH -p gpu_a100#,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 15G

#python3 exp_runner.py --mode train --conf confs/gonzalo_finetune_allLayers.conf

# dataset { data_dirs = ["./datasets/Gonzalo/132/2021-07-22-12-28-42/portrait_reconstruction/",], images_to_pick = [] },
python3 exp_runner.py \
	--mode interpolate_15_36 \
	--extra_config_args 'dataset {  data_dirs = ["./datasets/Gonzalo/132/2021-07-22-12-28-42/portrait_reconstruction/",], images_to_pick = [[0, "default"]] }, model { neus_renderer { n_outside = 0 } }' \
	--checkpoint_path ./logs/gonzalo_multi_skip_noVD_noWN_scenewiseWD1e-2_ftTo132-1/checkpoints/ckpt_0015000.pth

