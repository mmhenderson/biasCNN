#!/bin/bash
#SBATCH --partition=general_short
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
ROOT=/cube/neurocube/local/serenceslab/maggie/
#ROOT=/mnt/neurocube/local/serenceslab/maggie/
rand_seed_jitter=576767

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

dataset_root=FiltIms11Cos_SF_0.14
which_model=vgg_16
declare -a sets=(1 2 3 4)

for set in ${sets[@]}
do

	dataset_name=${dataset_root}_rand${set}	
	
	#source ~/anaconda3/bin/activate
	${ROOT}biasCNN/code/shell_scripts/EvalImageNetRots/get_tuning_single_pixels.sh ${dataset_name} ${ROOT} ${overwrite} ${TEST}

done

# now do the analysis of these tuning curves, combining across all the image sets...
codepath=${ROOT}biasCNN/code/analysis_code/
cd ${codepath}

training_str=scratch_imagenet_rot_${rot}
model_short=pixel
training_str=pixel
which_hyperpars=params1
nSamples=${#sets[@]}
step_num=0

python analyze_orient_tuning_jitter.py ${ROOT} ${model_short} ${training_str} ${dataset_root} ${nSamples} ${which_hyperpars} ${step_num} ${rand_seed_jitter}

echo "finished analyzing/fitting tuning curves!"

