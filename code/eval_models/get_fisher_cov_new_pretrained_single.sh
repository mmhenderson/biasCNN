#!/bin/bash

# evaluate pre-trained VGG-16 model with the desired dataset

## 	GATHER MY ARGUMENTS AND PRINT THEM OUT FOR VERIFICATION
# what model?
which_model=${1}
# what dataset?
dataset_name=${2}
# what is my file path root?
ROOT=${3}
# what is my ckpt file?
ckpt_file=${4}
# am i over-writing or not?
overwrite=${5}
# testing/debugging mode?
TEST=${6}

echo -e "\nSTARTING EVAL FOR PRETRAINED WITH NEW DATASET\n"
echo "	dataset=$dataset_name"
echo "  ckpt_file=$ckpt_file"
echo "	model=$which_model"
echo "	overwrite=$overwrite"
echo "	test=$TEST"

# PATHS
codepath=${ROOT}/code/analysis_code/

echo "	codepath=$codepath"

# this specifies the exact file for the trained model we want to look at.
model_short=${which_model//_/}

# where we saved the reduced activ patterns after PCA
reduced_dir=${ROOT}/activations/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_reduced

# where we save the results of fisher
save_dir=${ROOT}/code/fisher_info_cov_new/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_reduced_varyncomps
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
		python get_fisher_info_cov_new_vary_ncomps.py ${reduced_dir} ${save_dir} ${model_short} ${dataset_name}
			
	else
		echo "running separability analysis"
	fi
fi

echo -e "\nFINISHED WITH THIS DATASET EVALUATION\n"
