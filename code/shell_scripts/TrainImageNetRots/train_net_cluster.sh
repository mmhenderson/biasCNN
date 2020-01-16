#!/bin/bash

# train VGG16 model on the desired dataset

which_model=$1
params=$2
rot=$3
from_scratch=$4

echo "script=$0"
echo "which_model=$which_model"
echo "params=$params"
echo "rot=$rot"
echo "from_scratch=$4"

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
max_number_of_steps=1000000
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

log_dir="${log_path}"'scratch_imagenet_rot_'"${rot}"'/'"${params}"

# check if this version of the model has been trained already, if it has, then make a brand new folder to restart it
if [ -d ${log_dir} ] && [[ $from_scratch == 1 ]]
then
	echo logdir exists already
	declare -a init_num=1
	log_dir_new=${log_dir}_init${init_num}
	while [ -d ${log_dir_new} ]
	do
		init_num=$((init_num+1))
		log_dir_new=${log_dir}_init${init_num}
	done
	log_dir=${log_dir_new}
	echo new version will be ${log_dir}
fi

mkdir -p ${log_dir}
echo saving to ${log_dir}

split_name=train

echo ${max_number_of_steps}
cd ${slimpath}
#Train the network.
echo python train_image_classifier_biasCNN.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours} --batch_size=${batch_size}

python train_image_classifier_biasCNN.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours} --batch_size=${batch_size}



