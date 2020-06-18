#!/bin/bash
#SBATCH --partition=general_gpu_k40
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL

source ~/anaconda3/bin/activate
/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/TrainImageNetRots/train_vgg16_cluster_params1_3.sh
