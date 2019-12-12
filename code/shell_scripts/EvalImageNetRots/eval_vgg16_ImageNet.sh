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

# MMH 10/28/19


# Input my custom paths here
root=/usr/local/serenceslab/maggie/biasCNN/
codepath=${root}code/analysis_code/
slimpath=/usr/local/serenceslab/maggie/tensorflow/models/research/slim/

# this is the basic spatial frequency image set
dataset_name=imagenet
dataset_path=${root}datasets/ImageNet/ILSVRC2012/

	
which_model=vgg_16

#declare -a rot_list=(0 22 45)
declare -a rot_list=(0)

# what step do we want to use as the final checkpoint?
step_num=1000000

# loop over rotations (training sets) and eval on same test set
for rot in ${rot_list[@]}
do

    # this specifies the exact file for the trained model we want to look at.
    load_log_dir=${root}logs/vgg16/ImageNet/scratch_vgg16_imagenet_rot_${rot}/weightdecay_0.00005_rmspropdecay_0.90_rmspropmomentum_0.80_learningrate_0.001_learningratedecay_0.94/model.ckpt-${step_num}

    set -e
    
	dataset_dir=${dataset_path}/tfrecord_rot_${rot}/
	if [ ! -d ${dataset_dir} ]
	then
		raise error "dataset not found"
	fi

	# where we save the logs
    save_eval_dir=${root}activations/vgg16/scratch_vgg16_imagenet_rot_${rot}_${dataset_name}_EVAL_ckpt_${step_num}
    if [ ! -d ${save_eval_dir} ]
    then
    	mkdir -p ${save_eval_dir}
    fi

    # Evaluate the network.
    cd ${slimpath}
    python eval_image_classifier_biasCNN_eval_only.py \
     --checkpoint_path=${load_log_dir} \
     --eval_dir=${save_eval_dir} \
     --dataset_name=${dataset_name} \
     --dataset_dir=${dataset_dir} \
     --model_name=${which_model} \
     --num_classes=1001 \
	 --dataset_split_name=validation \
	 --append_scope_string=my_scope

done
