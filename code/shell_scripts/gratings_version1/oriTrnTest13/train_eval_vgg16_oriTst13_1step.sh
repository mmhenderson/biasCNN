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
# This script performs the following operations:
# 1. Evaluates a model that was trained on orientation discrim for gratings.
#	-Evaluate it at 5 different levels of noise, all on images that were not seen during training.
# 2. Saves weights at each layer of this network for various grating inputs.
# 3. Reduces the weights with PCA at each layer, within each noise level.
# 4. Removes the original activation files to save space.

# MMH 11/17/18

# Input my custom paths here
root=/usr/local/serenceslab/maggie/biasCNN/

codepath=${root}code

slimpath=/usr/local/serenceslab/maggie/tensorflow/models/research/slim/

which_model=vgg_16

scopes_to_exclude=vgg_16/fc8,vgg_16/fc8/squeezed
#scopes_to_exclude=final_layer/FC,aux_11/aux_logits/FC
scopes_to_train=${scopes_to_exclude}

flipLR=False

set -e

restore_ckpt_dir=${root}checkpoints/vgg16_ckpt/vgg_16.ckpt

declare -a names=(13a 13b 13c 13d 13e 13f)

for nn in ${names[@]}
do
 
    dataset_name=oriTrn${nn}
    
    dataset_dir=${root}datasets/${dataset_name}
    if [ ! -d ${dataset_dir} ]
    then
    	raise error "dataset not found"
    fi
    log_dir=${root}logs/vgg16_oriTrn${nn}_short
    echo ${dataset_name}
    split_name=train

    cd ${slimpath}
    # Train the network.
    python train_image_classifier_MMH_biasCNN.py \
      --train_dir=${log_dir} \
      --dataset_name=${dataset_name} \
      --dataset_split_name=${split_name} \
      --dataset_dir=${dataset_dir} \
      --model_name=${which_model} \
      --checkpoint_path=${restore_ckpt_dir} \
      --checkpoint_exclude_scopes=${scopes_to_exclude} \
      --trainable_scopes=${scopes_to_train} \
      --max_number_of_steps=1 \
      --batch_size=32 \
      --learning_rate=0.01 \
      --save_interval_secs=60 \
      --save_summaries_secs=60 \
      --log_every_n_steps=100 \
      --optimizer=rmsprop \
      --weight_decay=0.00004
    	
    dataset_name=oriTst${nn}
    dataset_dir=${root}datasets/${dataset_name}
    if [ ! -d ${dataset_dir} ]
    then
    	raise error "dataset not found"
    fi
    
    # where we save the large complete activation files
    save_eval_dir=${root}activations/vgg16_oriTst${nn}_short_full
    if [ ! -d ${save_eval_dir} ]
    then
    	mkdir -p ${save_eval_dir}
    fi
    
    # where we save the results of PCA
    reduced_dir=${root}activations/vgg16_oriTst${nn}_short_reduced
    if [ ! -d ${reduced_dir} ]
    then
    	mkdir -p ${reduced_dir}
    fi
    
    
    echo " loading from $dataset_dir"
    echo " saving to $save_eval_dir and $reduced_dir"
    
    set -e
    
    # Evaluate the network.
    cd ${slimpath}
    python eval_image_classifier_MMH_biasCNN.py \
     --checkpoint_path=${log_dir} \
     --eval_dir=${save_eval_dir} \
     --dataset_name=${dataset_name} \
     --dataset_dir=${dataset_dir} \
     --model_name=${which_model}\
     --num_batches=96
    
    # Reduce the weights with PCA
    cd ${codepath}
    python reduce_weights.py \
     --activ_path=${save_eval_dir} \
     --reduced_path=${reduced_dir} \
     --n_components_keep=500 \
     --num_batches=96 \
     --model_name=${which_model}
    
    # Remove the large activation files after reducing.
    rm -r ${save_eval_dir}
    
done