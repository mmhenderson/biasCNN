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

# where we save the results of Fisher info
fish_dir=${ROOT}/saved_analyses/fisher_info/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_full
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
tuning_dir=${ROOT}/activations/${model_short}/pretrained/params1/${dataset_name}/eval_at_ckpt-0_orient_tuning
if [[ ! -d ${tuning_dir} ]] || [[ -z $(ls -A ${tuning_dir}) ]]
then
	mkdir -p ${tuning_dir}
	num_tuning_files=0
else
	tuning_files=($(ls ${tuning_dir}/*.npy))
	num_tuning_files=${#tuning_files[@]}
	echo "	there are $num_tuning_files files in tuning folder"
fi
if (( $num_tuning_files < 63 ))
then
	doTuning=1
else
	doTuning=0
fi

echo "	do_eval=$doEval"
echo "	do_fish=$doFish"
echo "	do_tuning=$doTuning"

echo -e "\nloading dataset from $dataset_dir"
echo -e "\nloading checkpoint from $ckpt_file"
echo -e "\nsaving to $save_eval_dir and $fish_dir and $tuning_dir\n"

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

#source ~/anaconda3/bin/activate
# calculate single unit tuning curves
if [[ $overwrite == 1 ]] || [[ $doTuning == 1 ]]
then
	cd ${codepath}
	if [[ $TEST != 1 ]]
	then
		num_batches=96
		python get_orient_tuning_avgspace.py ${save_eval_dir} ${tuning_dir} ${model_short} ${dataset_name} ${num_batches}		
	else
		echo "estimating single unit tuning"
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
