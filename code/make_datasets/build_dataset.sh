#!/bin/bash
#SBATCH --partition=general_gpu_k80
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

source ~/anaconda3/bin/activate
/cube/neurocube/local/serenceslab/maggie/biasCNN/code/make_datasets/convert_imagenet_rotations_cos.sh

