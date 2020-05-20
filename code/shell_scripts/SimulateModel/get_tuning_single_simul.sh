#!/bin/bash

# evaluate trained VGG-16 model with the desired dataset

## 	GATHER MY ARGUMENTS AND PRINT THEM OUT FOR VERIFICATION
# string for the model initialization
init_str=${1}
# what model?
which_model=${2}
# what dataset?
dataset_name=${3}
# what is my file path root?
ROOT=${4}
# what is my log directory?
log_dir=${5}
# am i over-writing or not?
overwrite=${6}
# testing/debugging mode?
TEST=${7}

# what step do we want to use as the final checkpoint?
step_num=0
# using shorthand for the full description of model hyperparameters
which_hyperpars=params1


echo -e "\nSTARTING EVAL WITH NEW DATASET\n"
echo "	init_str=$init_str"
echo "	dataset=$dataset_name"
echo "	model=$which_model"
echo "	overwrite=$overwrite"
echo "	test=$TEST"

# PATHS
codepath=${ROOT}biasCNN/code/analysis_code/
slimpath=${ROOT}tensorflow/models/research/slim/

# this specifies the exact file for the trained model we want to look at.
model_short=${which_model}


# where we save the large complete activation files
save_eval_dir=${ROOT}biasCNN/activations/${model_short}/${init_str}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_full

# where we save the tuning curves
tuning_dir=${ROOT}biasCNN/activations/${model_short}/${init_str}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_orient_tuning
if [[ ! -d ${tuning_dir} ]] || [[ -z $(ls -A ${tuning_dir}) ]]
then
	mkdir -p ${tuning_dir}
	num_tuning_files=0
else
	tuning_files=($(ls ${tuning_dir}/*.npy))
	num_tuning_files=${#tuning_files[@]}
	echo "	there are $num_tuning_files files in reduced folder"
fi
if (( $num_tuning_files < 19 ))
then
	doReduce=1
else
	doReduce=0
	doEval=0
fi

echo "	do_reduce=$doReduce"
echo -e "\nsaving to $tuning_dir\n"

# Calculate single unit orientation tuning curves
if [[ $overwrite == 1 ]] || [[ $doReduce == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then
		num_batches=96
		python get_orient_tuning.py ${save_eval_dir} ${tuning_dir} ${model_short} ${dataset_name} ${num_batches}		 
	else
		echo "estimating single unit tuning"
	fi
fi

# make sure the job is really finished
#tuning_files=($(ls ${tuning_dir}/*.npy))
#num_tuning_files=${#tuning_files[@]}
#echo "	there are $num_tuning_files files in reduced folder"
#if (( $num_tuning_files < 19 ))
#then
#	reallyDone=0
#else
#	reallyDone=1
#fi

# Remove the large activation files after reducing.
#if [[ $TEST != 1 ]] && [[ $reallyDone == 1 ]]
#then
	#echo "finished, but not deleting big files until we check"
	#echo "removing big files now"
	#rm -r ${save_eval_dir}
#fi

echo -e "\nFINISHED WITH THIS DATASET EVALUATION\n"
