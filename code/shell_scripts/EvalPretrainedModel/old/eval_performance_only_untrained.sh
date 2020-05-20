#!/bin/bash
#SBATCH --partition=general_gpu_k40
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

source ~/anaconda3/bin/activate

ROOT=/cube/neurocube/local/serenceslab/maggie/
#ROOT=/mnt/neurocube/local/serenceslab/maggie/
#rot=0
#which_hyperpars=params1

step_num=0

echo "eval on ckpt number ${step_num[@]}"

# where is the checkpoint file, and where will i also save the logs for evaluation?
log_dir=${ROOT}biasCNN/logs/vgg16/ImageNet/scratch_imagenet_stop_early/params1/

## SET MORE PATHS/PARAMETERS
codepath=${ROOT}biasCNN/code/analysis_code/
slimpath=${ROOT}tensorflow/models/research/slim/

# these things will always be the same
which_model=vgg_16
dataset_name=imagenet
dataset_split_name=validation

# the specific dataset to evaluate on (same one as training)
dataset_dir=${ROOT}biasCNN/datasets/ImageNet/ILSVRC2012/tfrecord_rot_0

# this specifies the exact file for the trained model we want to look at.
load_log_dir=${log_dir}model.ckpt-${step_num}	
# where we save the large complete activation files
save_eval_dir=${log_dir}eval_at_ckpt-${step_num}_performance_only

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

python eval_image_classifier_biasCNN_performance_only.py \
	 --checkpoint_path=${load_log_dir} \
	 --eval_dir=${save_eval_dir} \
	 --dataset_name=${dataset_name} \
	 --dataset_dir=${dataset_dir} \
	 --dataset_split_name=${dataset_split_name} \
	 --model_name=${which_model} \
	 --num_batches=96 \
	 --append_scope_string=my_scope \
	 --num_classes=1001
	
