#!/bin/bash

# train VGG16 model on the desired dataset

# Specify the directory i am working in
#ROOT=/usr/local/serenceslab/maggie/
ROOT=/cube/neurocube/local/serenceslab/maggie/

# where is all my tensorflow code?
slimpath=${ROOT}tensorflow/models/research/slim/
# where am i loading tfrecord files from?
dataset_path=${ROOT}biasCNN/datasets/ImageNet/ILSVRC2012/
# where am i saving log files to?
log_path=${ROOT}biasCNN/logs/vgg16/ImageNet/

which_model=vgg_16
dataset_name=imagenet

rot=$1

echo "script=$0"
echo "rot=$rot"

set -e

flipLR=False
random_scale=False
is_windowed=True
weight_decay=0.0005
max_number_of_steps=1500000
max_checkpoints_to_keep=5
keep_checkpoint_every_n_hours=0.5
batch_size=32
rmsprop_decay=0.90
rmsprop_momentum=0.90
learning_rate=0.01
learning_rate_decay_factor=0.94

dataset_dir=${dataset_path}/tfrecord_rot_${rot}/

if [ ! -d ${dataset_dir} ]
then
	exit
fi

log_dir="${log_path}"'scratch_vgg16_imagenet_rot_'"${rot}"'/weightdecay_'"${weight_decay}"'_rmspropdecay_'"${rmsprop_decay}"'_rmspropmomentum_'"${rmsprop_momentum}"'_learningrate_'"${learning_rate}"'_learningratedecay_'"${learning_rate_decay_factor}"

# check if this version of the model has been trained already, if it has, then make a brand new folder to restart it
if [ -d ${log_dir} ]
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
echo python train_image_classifier_biasCNN_NEW.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours} --batch_size=${batch_size}

python train_image_classifier_biasCNN_NEW.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours} --batch_size=${batch_size}



