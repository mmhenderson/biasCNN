#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --mail-user=xxx@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=500000
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

# Compute full multivariate Fisher info measure. Need to run PCA first.

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
CWD=$(pwd)
cd ../../
ROOT=$(pwd)

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

dataset_root=FiltIms14AllSFCos
which_model=vgg_16
# num of versions of this dataset (phases are different)
declare -a sets=(1 2 3 4)

# first define the folder where all checkpoint for this model will be located
model_short=${which_model//_/}

# this specifies the exact file for the trained model we want to look at.
ckpt_file=${ROOT}/checkpoints/vgg16_ckpt/vgg_16.ckpt

echo "evaluating pretrained model"
	
# loop over number of versions of this dataset.
for set in ${sets[@]}
do
	
	dataset_name=${dataset_root}_rand${set}
	
	#source ~/anaconda3/bin/activate
	${CWD}/get_fisher_cov_pretrained_single.sh ${which_model} ${dataset_name} ${ROOT} ${ckpt_file} ${overwrite} ${TEST}
	
done

