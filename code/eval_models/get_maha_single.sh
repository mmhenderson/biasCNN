#!/bin/bash

# evaluate trained VGG-16 model with the desired dataset

## 	GATHER MY ARGUMENTS AND PRINT THEM OUT FOR VERIFICATION
# amount the images were rotated by
rot=${1}
# what step do we want to use as the final checkpoint?
step_num=${2}
# using shorthand for the full description of model hyperparameters
which_hyperpars=${3}
# what model?
which_model=${4}
# what dataset?
dataset_name=${5}
# what is my file path root?
ROOT=${6}
# what is my log directory?
log_dir=${7}
# am i over-writing or not?
overwrite=${8}
# testing/debugging mode?
TEST=${9}

echo -e "\nSTARTING EVAL WITH NEW DATASET\n"
echo "	step_num=$step_num"
echo "	dataset=$dataset_name"
echo "	rot=$rot"
echo "	which_hyperpars=$which_hyperpars"
echo "	model=$which_model"
echo "	overwrite=$overwrite"
echo "	test=$TEST"

# PATHS
codepath=${ROOT}/code/analysis_code/

echo "	codepath=$codepath"

# this specifies the exact file for the trained model we want to look at.
model_short=${which_model//_/}

# where we saved the reduced activ patterns after PCA
reduced_dir=${ROOT}/activations/${model_short}/scratch_imagenet_rot_${rot}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_reduced

# where we save the results of fisher
save_dir=${ROOT}/code/mahal_dist/${model_short}/scratch_imagenet_rot_${rot}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_reduced_varyncomps
if [[ ! -d ${save_dir} ]] || [[ -z $(ls -A ${save_dir}) ]]
then
	mkdir -p ${save_dir}
	num_save_files=0
else
	save_files=($(ls ${save_dir}/*.npy))
	num_save_files=${#save_files[@]}
	echo "	there are $num_save_files files in saving folder"
fi
if (( $num_save_files < 1 ))
then
	doAna=1
else
	doAna=0

fi

echo "	doAna=$doAna"

echo -e "\nloading from $reduced_dir"
echo -e "\nsaving to $save_dir\n"


# Run fisher info calc on reduced activs
if [[ $overwrite == 1 ]] || [[ $doAna == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then		
		python get_maha_vary_ncomps.py ${reduced_dir} ${save_dir} ${model_short} ${dataset_name}
			
	else
		echo "running separability analysis"
	fi
fi

echo -e "\nFINISHED WITH THIS DATASET EVALUATION\n"
