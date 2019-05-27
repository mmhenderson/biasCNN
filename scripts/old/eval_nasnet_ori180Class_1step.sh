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
# 2. Saves weights at each layer of this network for various grating inputs.
#
# Usage:
# cd /usr/local/serenceslab/maggie/biasCNN/
# ./eval_inceptionV3_ori180Class.sh

# Input my custom paths here
root=/usr/local/serenceslab/maggie/biasCNN/

slimpath=/usr/local/serenceslab/maggie/tensorflow/models/research/slim/
cd ${slimpath}

dataset_dir=${root}datasets/datasets_Grating_Orient_SF

#restore_ckpt_dir=${root}checkpoints/pnas_ckpt/nasnet-a_large_04_10_2017/model.ckpt
load_log_dir=${root}logs/nasnet_retrained_grating_orient_sf_short_tst

save_eval_dir=${root}weights/nasnet_grating_orient_sf_short_tst

if [ ! -d ${save_weights_dir} ]
then
    mkdir ${save_weights_dir}
fi

dataset_name=grating_orient_sf

split_name=train

#which_model=inception_v3
which_model=nasnet_large

set -e

# Train the network.
python eval_image_classifier_MMH_biasCNN.py \
  --checkpoint_path=${load_log_dir} \
  --eval_dir=${save_eval_dir} \
  --dataset_name=${dataset_name} \
  --dataset_split_name=validation \
  --dataset_dir=${dataset_dir} \
  --model_name=${which_model}

