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
fisher_dir=${ROOT}/code/fisher_info/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_reduced
if [[ ! -d ${fisher_dir} ]] || [[ -z $(ls -A ${fisher_dir}) ]]
then
	mkdir -p ${fisher_dir}
	num_fisher_files=0
else
	fisher_files=($(ls ${fisher_dir}/*.npy))
	num_fisher_files=${#fisher_files[@]}
	echo "	there are $num_fisher_files files in fisher folder"
fi
if (( $num_fisher_files < 1 ))
then
	doFisher=1
else
	doFisher=0

fi

echo "	do_fisher=$doFisher"

echo -e "\nloading from $reduced_dir"
echo -e "\nsaving to $fisher_dir\n"


# Run fisher info calc on reduced activs
if [[ $overwrite == 1 ]] || [[ $doFisher == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then
		num_batches=96
		min_var_expl=99
		python get_fisher_info_reduced.py ${reduced_dir} ${fisher_dir} ${model_short} ${dataset_name} ${num_batches} ${min_var_expl}
		
	else
		echo "running fisher"
	fi
fi

echo -e "\nFINISHED WITH THIS DATASET EVALUATION\n"
