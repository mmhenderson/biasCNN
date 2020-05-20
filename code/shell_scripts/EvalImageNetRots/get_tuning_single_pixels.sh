#!/bin/bash

# evaluate trained VGG-16 model with the desired dataset

## 	GATHER MY ARGUMENTS AND PRINT THEM OUT FOR VERIFICATION
# amount the images were rotated by
# what dataset?
dataset_name=${1}
# what is my file path root?
ROOT=${2}
# am i over-writing or not?
overwrite=${3}
# testing/debugging mode?
TEST=${4}
# which version of pixel model
which_model=${5}

echo -e "\nSTARTING EVAL WITH NEW DATASET\n"
echo "	dataset=$dataset_name"
echo "	overwrite=$overwrite"
echo "	test=$TEST"
echo " 	model=$which_model"

# PATHS
codepath=${ROOT}biasCNN/code/analysis_code/
slimpath=${ROOT}tensorflow/models/research/slim/

# where the evaluation image dataset is located
dataset_dir=${ROOT}biasCNN/images/gratings/${dataset_name}
if [ ! -d ${dataset_dir} ]
then
	raise error "dataset not found"
fi

# where we save the tuning curves
tuning_dir=${ROOT}biasCNN/activations/pixel/${which_model}/params1/${dataset_name}/eval_at_ckpt-0_orient_tuning
if [[ ! -d ${tuning_dir} ]] || [[ -z $(ls -A ${tuning_dir}) ]]
then
	mkdir -p ${tuning_dir}
	num_tuning_files=0
else
	tuning_files=($(ls ${tuning_dir}/*.npy))
	num_tuning_files=${#tuning_files[@]}
	echo "	there are $num_tuning_files files in reduced folder"
fi
if (( $num_tuning_files < 1 ))
then
	doReduce=1
else
	doReduce=0
fi

echo "	do_reduce=$doReduce"
echo -e "\nloading dataset from $dataset_dir"
echo -e "\nsaving to $tuning_dir\n"

#source ~/anaconda3/bin/activate
# Calculate single unit orientation tuning curves
if [[ $overwrite == 1 ]] || [[ $doReduce == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then
		python get_orient_tuning_pixels.py ${dataset_dir} ${tuning_dir} ${dataset_name}	${which_model}
	else
		echo "estimating single unit tuning"
	fi
fi

# make sure the job is really finished
tuning_files=($(ls ${tuning_dir}/*.npy))
num_tuning_files=${#tuning_files[@]}
echo "	there are $num_tuning_files files in reduced folder"
if (( $num_tuning_files < 1 ))
then
	reallyDone=0
else
	reallyDone=1
fi

echo -e "\nFINISHED WITH THIS DATASET EVALUATION\n"
