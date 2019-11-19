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
declare -a rot_list=(0)

set -e

flipLR=False
random_scale=False
is_windowed=True
weight_decay=0.00005
max_number_of_steps=1000000
max_checkpoints_to_keep=5
keep_checkpoint_every_n_hours=0.5

declare -a rmsprop_decay_list=(0.90)
declare -a rmsprop_momentum_list=(0.80)
declare -a learning_rate_list=(0.005)
declare -a learning_rate_decay_factor_list=(0.94)

dataset_name=imagenet

declare -a xx=0

for rot in ${rot_list[@]}
do
	for rmsprop_decay in ${rmsprop_decay_list[@]}
	do
		for rmsprop_momentum in ${rmsprop_momentum_list[@]}
		do
			for learning_rate in ${learning_rate_list[@]}
			do
				for learning_rate_decay_factor in ${learning_rate_decay_factor_list[@]}
				do
    				echo ${xx}
				#if [ ! ${xx} == 0 ]; then
	 
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
					echo python train_image_classifier_biasCNN.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours}

					python train_image_classifier_biasCNN.py --train_dir=${log_dir} --dataset_name=${dataset_name} --dataset_split_name=${split_name} --dataset_dir=${dataset_dir} --model_name=${which_model} --max_number_of_steps=${max_number_of_steps} --flipLR=${flipLR} --random_scale=${random_scale} --is_windowed=${is_windowed} --weight_decay=${weight_decay} --rmsprop_decay=${rmsprop_decay} --rmsprop_momentum=${rmsprop_momentum} --learning_rate=${learning_rate} --learning_rate_decay_factor=${learning_rate_decay_factor} --max_checkpoints_to_keep=${max_checkpoints_to_keep} --keep_checkpoint_every_n_hours=${keep_checkpoint_every_n_hours}

					
				#fi
					xx=$((xx+1))
				done
				
			done
		done
	done
done
