#!/bin/bash

# train VGG16 model on the desired dataset

which_model=$1
params=$2
rot=$3
init_num=$4
max_steps=$5

echo "script=$0"
echo "which_model=$which_model"
echo "params=$params"
echo "rot=$rot"
echo "init_num=$4"
echo "max_steps=$5"

# Specify the directory i am working in
#ROOT=/usr/local/serenceslab/maggie/
ROOT=/cube/neurocube/local/serenceslab/maggie/

# where is all my tensorflow code?
slimpath=${ROOT}tensorflow/models/research/slim/
# where am i loading tfrecord files from?
dataset_path=${ROOT}biasCNN/datasets/ImageNet/ILSVRC2012/
# where am i saving log files to?
log_path=${ROOT}biasCNN/logs/${which_model//_/}/ImageNet/

dataset_name=imagenet

set -e

flipLR=False
random_scale=False
is_windowed=True
max_number_of_steps=$max_steps
max_checkpoints_to_keep=5
keep_checkpoint_every_n_hours=0.5
batch_size=32

if [[ $params == "params1" ]]
then
  weight_decay=0.00005
  rmsprop_decay=0.90
  rmsprop_momentum=0.80
  learning_rate=0.005
  learning_rate_decay_factor=0.94
elif [[ $params == "params2" ]]
then
  weight_decay=0.00005
  rmsprop_decay=0.90
  rmsprop_momentum=0.80
  learning_rate=0.001
  learning_rate_decay_factor=0.94
fi

dataset_dir=${dataset_path}/tfrecord_rot_${rot}/

if [ ! -d ${dataset_dir} ]
then
	exit
fi

if (( $init_num==0 ))
then
	log_dir="${log_path}"'scratch_imagenet_rot_'"${rot}"'/'"${params}"'/'
else
	log_dir="${log_path}"'scratch_imagenet_rot_'"${rot}"'/'"${params}"'_init'"${init_num}"'/'
fi

# check if dir exists already
if [ -d ${log_dir} ]
then
	# see how far this iteration has already gotten
	allckptfiles=$(ls ${log_dir}model.ckpt-*.meta) 
	declare -a all_numbers=()
	for file in ${allckptfiles[@]}
	do
		second_part=${file/*ckpt-}
		number=${second_part/.*}
		all_numbers+=($number)
	done

	# check if there's a checkpoint past the desired max number
	finished=0
	for number in ${all_numbers[@]}
	do		
		if (( $number>=$max_steps )) 
		then		
			echo "number=${number}"	
			finished=1
		fi
	done
	echo "finished=${finished}"
	if (( $finished==1 ))
	then
		echo "${log_dir}"
		echo "already done up to step $max_steps. Aborting..."
		exit
	fi
else
	# make a brand new directory
	mkdir -p ${log_dir}
fi

# finally, get ready to start training...
echo saving to ${log_dir}

split_name=train

echo ${max_number_of_steps}
cd ${slimpath}
#Train the network.
echo python train_image_classifier_biasCNN.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours} --batch_size=${batch_size}

python train_image_classifier_biasCNN.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours} --batch_size=${batch_size}



