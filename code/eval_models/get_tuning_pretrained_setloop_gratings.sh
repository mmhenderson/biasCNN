#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --mail-user=xxx@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=500000
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
CWD=$(pwd)
cd ../../
ROOT=$(pwd)

rand_seed_jitter=129838

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

dataset_root=CosGratings
which_model=vgg_16
declare -a sets=(CosGratings CosGratings1 CosGratings2 CosGratings3)

# first define the folder where all checkpoint for this model will be located
model_short=${which_model//_/}

# this specifies the exact file for the trained model we want to look at.
ckpt_file=${ROOT}/checkpoints/vgg16_ckpt/vgg_16.ckpt

echo "evaluating pretrained model"

for set in ${sets[@]}
do

	dataset_name=${set}	
	
	#source ~/anaconda3/bin/activate
	${CWD}/get_tuning_pretrained_single.sh ${which_model} ${dataset_name} ${ROOT} ${ckpt_file} ${overwrite} ${TEST}

done

# now do the analysis of these tuning curves, combining across all the image sets...
codepath=${ROOT}/code/analysis_code/
cd ${codepath}

training_str=pretrained
model_short=${which_model//_/}
nSamples=${#sets[@]}
which_hyperpars=params1
step_num=0
python analyze_orient_tuning_NOFIT.py ${ROOT} ${model_short} ${training_str} ${dataset_root} ${nSamples} ${which_hyperpars} ${step_num} ${rand_seed_jitter}

echo "finished analyzing/fitting tuning curves!"

