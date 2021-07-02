#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --mail-user=xxx@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=500000
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

# Analyze sinusoidal gratings: multiple spatial frequencies.
# This script does all the main analyses, including computing fisher info and single-unit tuning properties.

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
CWD=$(pwd)
cd ../../
ROOT=$(pwd)

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

dataset_root=CosGratingsMultiPhase
which_model=vgg_16
declare -a sf_vals=(0.01 0.02 0.04 0.08 0.14 0.25)

# first define the folder where all checkpoint for this model will be located
model_short=${which_model//_/}

# this specifies the exact file for the trained model we want to look at.
ckpt_file=${ROOT}/checkpoints/vgg16_ckpt/vgg_16.ckpt

# params for the tuning curve fitting
codepath=${ROOT}/code/analysis_code/
training_str=pretrained
model_short=${which_model//_/}
nSamples=1
which_hyperpars=params1
step_num=0

echo "evaluating pretrained model"

for sf in ${sf_vals[@]}
do

	dataset_name=${dataset_root}_SF_${sf}

    ${CWD}/get_tuning_and_fisher_pretrained_single.sh ${which_model} ${dataset_name} ${ROOT} ${ckpt_file} ${overwrite} ${TEST}

	cd ${codepath}

	if [[ $TEST != 1 ]]
	then
		python analyze_orient_tuning_nofit.py ${ROOT} ${model_short} ${training_str} ${dataset_name} ${nSamples} ${which_hyperpars} ${step_num} 
		python analyze_orient_tuning_avgspace.py ${ROOT} ${model_short} ${training_str} ${dataset_name} ${nSamples} ${which_hyperpars} ${step_num} 
	fi

	echo -e "\nfinished analyzing/fitting for $dataset_name\n"

done


