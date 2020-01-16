#!/bin/bash

# evaluate trained VGG-16 model with the desired dataset

# Specify the directory i am working in
#ROOT=/usr/local/serenceslab/maggie/
ROOT=/cube/neurocube/local/serenceslab/maggie/

codepath=${ROOT}biasCNN/code/analysis_code/
slimpath=${ROOT}tensorflow/models/research/slim/

which_model=vgg_16

# amount the images were rotated by
rot="$1"
# what step do we want to use as the final checkpoint?
step_num="$2"
# using shorthand for the full description of model hyperparameters
which_hyperpars="$3"

echo "script=$0"
echo "rot=$rot"
echo "step_num=$step_num"
echo "which_hyperpars=$which_hyperpars"

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
	if [ "${ss}" = "0" ]
	then
		dataset_name=SquareGratings
	else
		dataset_name=SquareGratings${ss}
	fi

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
