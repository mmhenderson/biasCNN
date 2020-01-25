#!/bin/bash

# evaluate trained VGG-16 model with the desired dataset

source activate ~/anaconda3/envs/cuda9

## 	GATHER MY ARGUMENTS AND PRINT THEM OUT FOR VERIFICATION
# amount the images were rotated by
rot=${1}
# what step do we want to use as the final checkpoint?
step_num=${2}
# using shorthand for the full description of model hyperparameters
which_hyperpars=${3}
# where is the checkpoint file, and where will i also save the logs for evaluation?
log_dir=${4}
# what is my file path root?
ROOT=${5}

echo "rot=$rot"
echo "step_num=$step_num"
echo "which_hyperpars=$which_hyperpars"
echo "log_dir=$log_dir"

## SET MORE PATHS/PARAMETERS
codepath=${ROOT}biasCNN/code/analysis_code/
slimpath=${ROOT}tensorflow/models/research/slim/

# these things will always be the same
which_model=vgg_16
dataset_name=SpatFreqGratings

# the specific dataset to evaluate on (same one as training)
dataset_dir=${ROOT}biasCNN/datasets/gratings/SpatFreqGratings/

# this specifies the exact file for the trained model we want to look at.
load_log_dir=${log_dir}model.ckpt-${step_num}	
# where we save the large complete activation files
save_eval_dir=${log_dir}eval_at_ckpt-${step_num}_visualize_test

if [ ! -d ${save_eval_dir} ]
then
	mkdir -p ${save_eval_dir}
fi

echo "evaluating from checkpoint at ${load_log_dir}"
echo "evaluating on images from ${dataset_dir}"
echo "saving to ${save_eval_dir}" 

set -e

# Evaluate the network.
cd ${slimpath}

python eval_image_classifier_biasCNN_visualize.py \
	 --checkpoint_path=${load_log_dir} \
	 --eval_dir=${save_eval_dir} \
	 --dataset_name=${dataset_name} \
	 --dataset_dir=${dataset_dir} \
	 --model_name=${which_model} \
	 --num_batches=96 \
	 --append_scope_string=my_scope \
	 --num_classes=1001 \
	 --is_windowed=True

