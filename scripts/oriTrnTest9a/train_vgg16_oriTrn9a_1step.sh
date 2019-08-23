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
cd ${slimpath}

dataset_dir=${root}datasets/oriTrn9a

restore_ckpt_dir=${root}checkpoints/vgg16_ckpt/vgg_16.ckpt
save_log_dir=${root}logs/vgg16_oriTrn9a_short
dataset_name=oriTrn9a

split_name=train

which_model=vgg_16
#which_model=nasnet_large

scopes_to_exclude=vgg_16/fc8,vgg_16/fc8/squeezed
#scopes_to_exclude=final_layer/FC,aux_11/aux_logits/FC
scopes_to_train=${scopes_to_exclude}

flipLR=False

set -e

# Train the network.
python train_image_classifier_MMH_biasCNN.py \
  --train_dir=${save_log_dir} \
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

