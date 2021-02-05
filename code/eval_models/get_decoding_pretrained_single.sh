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

#
# where we saved the reduced activ patterns after PCA
reduced_dir=${ROOT}/activations/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_reduced

# where we save the results of decoding
decoding_dir=${ROOT}/code/decoding/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_reduced
if [[ ! -d ${decoding_dir} ]] || [[ -z $(ls -A ${decoding_dir}) ]]
then
	mkdir -p ${decoding_dir}
	num_dec_files=0
else
	dec_files=($(ls ${decoding_dir}/*.npy))
	num_dec_files=${#dec_files[@]}
	echo "	there are $num_dec_files files in decoding folder"
fi
if (( $num_dec_files < 1 ))
then
	doDecode=1
else
	doDecode=0
	
fi


echo "	do_decode=$doDecode"

echo -e "\nloading dataset from $dataset_dir"
echo -e "\nloading activs from $reduced_dir"
echo -e "\nsaving to $decoding_dir\n"

# Run decoder on reduced activs
if [[ $overwrite == 1 ]] || [[ $doDecode == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then
		num_batches=96
		min_var_expl=99
		python run_decoding.py ${reduced_dir} ${decoding_dir} ${model_short} ${dataset_name} ${num_batches} ${min_var_expl}
		
	else
		echo "running decoding"
	fi
fi

echo -e "\nFINISHED WITH THIS DATASET EVALUATION\n"
