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

# evaluate VGG-16 model that has been trained by a specified amount, on a 
# specified image set.
# MMH 12/11/19

# Specify the directory i am working in
#ROOT=/usr/local/serenceslab/maggie/
ROOT=/cube/neurocube/local/serenceslab/maggie/

codepath=${ROOT}biasCNN/code/analysis_code/
slimpath=${ROOT}tensorflow/models/research/slim/


which_model=vgg_16
# amount the images were rotated by
rot=0
# what step do we want to use as the final checkpoint?
#step_num=694497
step_num=689475
# using shorthand for the full description of model hyperparameters
which_hyperpars=params1

if [[ $which_hyperpars == "params1" ]]
then
	parsfull=weightdecay_0.00005_rmspropdecay_0.90_rmspropmomentum_0.80_learningrate_0.005_learningratedecay_0.94
elif [[ $which_hyperpars == "params2" ]]
then
	parsfull=weightdecay_0.00005_rmspropdecay_0.90_rmspropmomentum_0.80_learningrate_0.001_learningratedecay_0.94
else
	raise error "params not found"
fi

# this specifies the exact file for the trained model we want to look at.
load_log_dir=${ROOT}/biasCNN/logs/vgg16/ImageNet/scratch_vgg16_imagenet_rot_${rot}/${parsfull}/model.ckpt-${step_num}

# loop over datasets that are almost identical, but have different noise instantiations
declare -i nSets=3

for ss in $(seq 1 $nSets)
do
	dataset_name=SpatFreqGratings${ss}

	dataset_dir=${ROOT}/biasCNN/datasets/gratings/${dataset_name}
	if [ ! -d ${dataset_dir} ]
	then
		raise error "dataset not found"
	fi
	
	# where we save the large complete activation files
	save_eval_dir=${ROOT}/biasCNN/activations/vgg16/scratch_imagenet_rot_${rot}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_full
	if [ ! -d ${save_eval_dir} ]
	then
		mkdir -p ${save_eval_dir}
	fi

	# where we save the results of PCA
	reduced_dir=${ROOT}/biasCNN/activations/vgg16/scratch_imagenet_rot_${rot}/${which_hyperpars}/${dataset_name}/eval_at_ckpt-${step_num}_reduced
	if [ ! -d ${reduced_dir} ]
	then
		mkdir -p ${reduced_dir}
	fi    

	echo " loading from $dataset_dir"
	echo " saving to $save_eval_dir and $reduced_dir"

	set -e

	# Evaluate the network.
	cd ${slimpath}
	python eval_image_classifier_biasCNN.py \
	 --checkpoint_path=${load_log_dir} \
	 --eval_dir=${save_eval_dir} \
	 --dataset_name=${dataset_name} \
	 --dataset_dir=${dataset_dir} \
	 --model_name=${which_model} \
	 --num_batches=96 \
	 --append_scope_string=my_scope \
	 --num_classes=1001


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
