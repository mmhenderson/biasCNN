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

# Input my custom paths here
root=/usr/local/serenceslab/maggie/biasCNN/

slimpath=/usr/local/serenceslab/maggie/tensorflow/models/research/slim/

which_model=vgg_16

declare -a rot_list=(0 22 45)

set -e

dataset_name=imagenet

for rot in ${rot_list[@]}
do
    
	dataset_dir=${slimpath}datasets/ILSVRC2012/tfrecord_rot_${rot}

	if [ ! -d ${dataset_dir} ]
	then
		raise error "dataset not found"
	fi

    log_dir=${root}logs/vgg16_imagenet_rot_${rot}
    
    split_name=train

    cd ${slimpath}
    # Train the network.
    python train_image_classifier_biasCNN.py \
      --train_dir=${log_dir} \
      --dataset_name=${dataset_name} \
      --dataset_split_name=${split_name} \
      --dataset_dir=${dataset_dir} \
      --model_name=${which_model} \     
      --max_number_of_steps=400000 \
	  --flipLR=False \
	  --random_scale=False
    
done
