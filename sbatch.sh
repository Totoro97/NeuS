#!/bin/bash

#SBATCH --job-name validate_ft3
#SBATCH --output ./stdout/%A.txt
#SBATCH --time 0:59:0

#SBATCH -p gpu_a100#,gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2
#SBATCH --mem-per-gpu 15G

#: '
CONF1=`mktemp`
cp confs/gonzalo_finetune.conf $CONF1
CONF2=`mktemp`
#cp confs/gonzalo_finetune_allLayers.conf $CONF2

python3 exp_runner.py --mode train --conf $CONF1
#python3 exp_runner.py --mode train --conf $CONF2
# '

#python3 exp_runner.py --mode train --extra_config_args 'dataset { images_to_pick_val = [[0, ["00689", "00479", "00556"]]] }, general { base_exp_dir = ./logs/debug-gonzalo_multi_skip_noVD_noWN_ftTo134-3 }, train { restart_from_iter = 2 }' --checkpoint_path ./logs/gonzalo_multi_skip_noVD_noWN_ftTo134-3/checkpoints/ckpt_0040000.pth
: '
# dataset { data_dirs = ["./datasets/Gonzalo/132/2021-07-22-12-28-42/portrait_reconstruction/",], images_to_pick = [] },
python3 exp_runner.py \
	--mode interpolate_25_9 \
	--extra_config_args 'dataset { images_to_pick = [[0, "default"]] }, model { neus_renderer { n_outside = 0 } }' \
	--checkpoint_path ./logs/gonzalo_multi_skip_noVD_noWN_ftTo134-1_1Stage_manMask/checkpoints/ckpt_0045000.pth
# '
# 132: 15_36
# 134: 25_9
# 
