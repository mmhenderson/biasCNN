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
slimpath=${ROOT}/code/tf_code/

echo "	codepath=$codepath"
echo "	slimpath=$slimpath"

# this specifies the exact file for the trained model we want to look at.
model_short=${which_model//_/}
# folder where the checkpoint file is located
load_log_dir=${log_dir}model.ckpt-${step_num}

# where the evaluation image dataset is located
dataset_dir=${ROOT}/datasets/gratings/${dataset_name}
if [ ! -d ${dataset_dir} ]
then
	raise error "dataset not found"
fi

# where we save the large complete activation files
save_eval_dir=${ROOT}/activations/${model_short}/scratch_imagenet_rot_${rot}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_full
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

# where we save the results of Fisher info
fish_dir=${ROOT}/saved_analyses/fisher_info/${model_short}/scratch_imagenet_rot_${rot}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_full
if [[ ! -d ${fish_dir} ]] || [[ -z $(ls -A ${fish_dir}) ]]
then
	mkdir -p ${fish_dir}
	num_fish_files=0
else
	fish_files=($(ls ${fish_dir}/*.npy))
	num_fish_files=${#fish_files[@]}
	echo "	there are $num_fish_files files in FI folder"
fi
if (( $num_fish_files < 3 ))
then
	doFish=1
else
	doFish=0
fi


# where we save the tuning curves
tuning_dir=${ROOT}/activations/${model_short}/scratch_imagenet_rot_${rot}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_orient_tuning
if [[ ! -d ${tuning_dir} ]] || [[ -z $(ls -A ${tuning_dir}) ]]
then
	mkdir -p ${tuning_dir}
	num_tuning_files=0
else
	tuning_files=($(ls ${tuning_dir}/*.npy))
	num_tuning_files=${#tuning_files[@]}
	echo "	there are $num_tuning_files files in orient tuning folder"
fi
if (( $num_tuning_files < 63 ))
then
	doTuning=1
else
	doTuning=0
fi

if [[ $doFish != 1 ]] && [[ $doTuning != 1 ]]
then
	doEval=0
fi

echo "	do_eval=$doEval"
echo "	do_fish=$doFish"
echo "	do_tuning=$doTuning"

echo -e "\nloading dataset from $dataset_dir"
echo -e "\nloading checkpoint from $load_log_dir"
echo -e "\nsaving to $save_eval_dir and $fish_dir and $tuning_dir\n"

# Evaluate the network.
if [[ $overwrite == 1 ]] || [[ $doEval == 1 ]]
then
	cd ${slimpath}
	if [[ $TEST != 1 ]]
	then
		python get_activations_biasCNN.py \
		 --checkpoint_path=${load_log_dir} \
		 --eval_dir=${save_eval_dir} \
		 --dataset_name=${dataset_name} \
		 --dataset_dir=${dataset_dir} \
		 --model_name=${which_model} \
		 --num_batches=96 \
		 --append_scope_string=my_scope \
		 --num_classes=1001 \
		 --is_windowed=True
	else
		echo "running evaluation"
	fi
fi


if [[ $overwrite == 1 ]] || [[ $doFish == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then
		num_batches=96
		python get_fisher_info_full.py ${save_eval_dir} ${fish_dir} ${model_short} ${dataset_name} ${num_batches}		 
	else	
		echo "calculating Fisher info"
	fi
fi

if [[ $overwrite == 1 ]] || [[ $doTuning == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then
		num_batches=96		
		python get_orient_tuning_avgspace.py ${save_eval_dir} ${tuning_dir} ${model_short} ${dataset_name} ${num_batches}	 
	else	
		echo "calculating tuning functions"
	fi
fi

# make sure the job is really finished
fish_files=($(ls ${fish_dir}/*.npy))
num_fish_files=${#fish_files[@]}
tuning_files=($(ls ${tuning_dir}/*.npy))
num_tuning_files=${#tuning_files[@]}
echo "	there are $num_tuning_files files in tuning curve folder"
echo "	there are $num_fish_files files in fisher info folder"
if (( $num_fish_files < 3 )) || (( $num_tuning_files < 63 ))
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
