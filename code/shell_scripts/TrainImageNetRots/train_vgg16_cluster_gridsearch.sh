#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#

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

#declare -a rot_list=(0 22 45)
#declare -a rot_list=(0)
rot=0

set -e

flipLR=False
random_scale=False
is_windowed=True
max_number_of_steps=100000
max_checkpoints_to_keep=5
keep_checkpoint_every_n_hours=0.5
batch_size=32

rmsprop_decay=0.90
rmsprop_momentum=0.80
learning_rate_decay_factor=0.94

declare -a learning_rate_list=(0.005 0.001)
declare -a weight_decay_list=(0.0005 0.0001 0.00005)

dataset_name=imagenet

declare -a xx=0

for weight_decay in ${weight_decay_list[@]}
do

	for learning_rate in ${learning_rate_list[@]}
	do
				
		echo ${xx}
		# skip some combinations we've already done before
		if [ ${xx} == 5 ]; then
		#if [ ! ${xx} == 4 ] && [ ! ${xx} == 0 ] && [ ! ${xx} == 1 ] && [ ! ${xx} == 2 ]; then

			dataset_dir=${dataset_path}/tfrecord_rot_${rot}/

			if [ ! -d ${dataset_dir} ]
			then
				exit
				#raise error "dataset not found"
			fi

			log_dir="${log_path}"'scratch_vgg16_imagenet_rot_'"${rot}"'/weightdecay_'"${weight_decay}"'_rmspropdecay_'"${rmsprop_decay}"'_rmspropmomentum_'"${rmsprop_momentum}"'_learningrate_'"${learning_rate}"'_learningratedecay_'"${learning_rate_decay_factor}"'/'

			mkdir -p ${log_dir}
			echo saving to ${log_dir}

			split_name=train

			echo ${max_number_of_steps}
			cd ${slimpath}
			#Train the network.
			echo python train_image_classifier_biasCNN.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours} --batch_size=${batch_size}

			python train_image_classifier_biasCNN.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours} --batch_size=${batch_size}

		
		fi
		xx=$((xx+1))
	done
		
done
