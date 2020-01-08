#!/bin/bash

# evaluate trained VGG-16 model with the desired dataset

# Specify the directory i am working in
ROOT=/usr/local/serenceslab/maggie/
#ROOT=/cube/neurocube/local/serenceslab/maggie/

codepath=${ROOT}biasCNN/code/analysis_code/
slimpath=${ROOT}tensorflow/models/research/slim/

which_model=vgg_16

# this specifies the exact file for the trained model we want to look at.
load_log_dir=${ROOT}biasCNN/checkpoints/vgg16_ckpt/vgg_16.ckpt

# loop over datasets that are almost identical, but have different noise instantiations
declare -i nSets=3

for ss in $(seq 0 $nSets)

do
	dataset_name=SpatFreqGratings${ss}

	dataset_dir=${ROOT}/biasCNN/datasets/gratings/${dataset_name}
	if [ ! -d ${dataset_dir} ]
	then
		raise error "dataset not found"
	fi
	
	# where we save the large complete activation files
	save_eval_dir=${ROOT}/biasCNN/activations/vgg16/pretrained/params1/${dataset_name}/eval_at_ckpt-0_full
	if [ ! -d ${save_eval_dir} ]
	then
		mkdir -p ${save_eval_dir}
	fi

	# where we save the results of PCA
	reduced_dir=${ROOT}/biasCNN/activations/vgg16/pretrained/params1/${dataset_name}/eval_at_ckpt-0_reduced
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
	 --labels_offset=0 \
	 --num_classes=1000



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
