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
slimpath=${ROOT}/code/tf_code/

echo "	codepath=$codepath"
echo "	slimpath=$slimpath"

# this specifies the exact file for the trained model we want to look at.
model_short=${which_model//_/}

# where the evaluation image dataset is located
dataset_dir=${ROOT}/datasets/gratings/${dataset_name}
if [ ! -d ${dataset_dir} ]
then
	raise error "dataset not found"
fi

# where we save the large complete activation files
save_eval_dir=${ROOT}/activations/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_full
if [[ ! -d ${save_eval_dir} ]] || [[ -z $(ls -A ${save_eval_dir}) ]]
then
	mkdir -p ${save_eval_dir}
	num_big_eval_files=0
else
	big_eval_files=($(ls ${save_eval_dir}/batch*.npy))
	num_big_eval_files=${#big_eval_files[@]}
	echo "	there are $num_big_eval_files files in big folder"
fi
if (( $num_big_eval_files < 2304 ))
then
	doEval=1
else
	doEval=0
fi

# where we save the reduced activ patterns after PCA
reduced_dir=${ROOT}/activations/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_reducedall
if [[ ! -d ${reduced_dir} ]] || [[ -z $(ls -A ${reduced_dir}) ]]
then
	mkdir -p ${reduced_dir}
	num_reduced_files=0
else
	reduced_files=($(ls ${reduced_dir}/*.npy))
	num_reduced_files=${#reduced_files[@]}
	echo "	there are $num_reduced_files files in reduced folder"
fi
if (( $num_reduced_files < 42 ))
then
	doReduce=1
else
	doReduce=0
	doEval=0
fi


echo "	do_eval=$doEval"
echo "	do_reduce=$doReduce"

echo -e "\nloading dataset from $dataset_dir"
echo -e "\nloading checkpoint from $ckpt_file"
echo -e "\nsaving to $save_eval_dir and $reduced_dir\n"

# Evaluate the network.
if [[ $overwrite == 1 ]] || [[ $doEval == 1 ]]
then
	cd ${slimpath}
	if [[ $TEST != 1 ]]
	then
		python get_activations_biasCNN.py \
		 --checkpoint_path=${ckpt_file} \
		 --eval_dir=${save_eval_dir} \
		 --dataset_name=${dataset_name} \
		 --dataset_dir=${dataset_dir} \
		 --model_name=${which_model} \
		 --num_batches=96 \
		 --num_classes=1000 \
		 --is_windowed=True
	else
		echo "running evaluation"
	fi
fi


# Reduce the activs with PCA
if [[ $overwrite == 1 ]] || [[ $doReduce == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then
		num_batches=96
		min_var_expl=100
		max_comp_keep=100000
		min_comp_keep=8640
		python reduce_activations.py ${save_eval_dir} ${reduced_dir} ${model_short} ${dataset_name} ${num_batches} ${min_var_expl} ${max_comp_keep} ${min_comp_keep}
		
	else
		echo "reducing activations"
	fi
fi

# make sure the job is really finished
reduced_files=($(ls ${reduced_dir}/*.npy))
num_reduced_files=${#reduced_files[@]}
echo "	there are $num_reduced_files files in reduced activ folder"

if (( $num_reduced_files < 42 ))
then
	reallyDone=0
else
	reallyDone=1
fi

# Remove the large activation files after reducing.
if [[ $TEST != 1 ]] && [[ $reallyDone == 1 ]]
then
	echo "removing big files now"
	rm -r ${save_eval_dir}
fi

echo -e "\nFINISHED WITH THIS DATASET EVALUATION\n"
