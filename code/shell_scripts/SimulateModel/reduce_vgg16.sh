#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=500000
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
ROOT=/cube/neurocube/local/serenceslab/maggie/biasCNN/
#ROOT=/mnt/neurocube/local/serenceslab/maggie/biasCNN/
code_dir=${ROOT}code/analysis_code/
shell_script_dir=${ROOT}code/shell_scripts/SimulateModel/
cd $code_dir

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

nSets=4
dataset_root=FiltIms5AllSFCos
which_model=vgg16_simul
rand_seed=456456
num_batches=96

# loop over number of versions of this dataset.
for ss in $(seq 1 $nSets)
do
	dataset_name=${dataset_root}_rand${ss}
	
	# save big activation patterns here 
	save_eval_dir=${ROOT}activations/${which_model}/random_normal_weights_${rand_seed}/params1/${dataset_name}/eval_at_ckpt-0_full/
	# save reduced activation patterns here
	reduced_dir=${ROOT}activations/${which_model}/random_normal_weights_${rand_seed}/params1/${dataset_name}/eval_at_ckpt-0_reduced/
	# where we save the tuning curves
	tuning_dir=${ROOT}activations/${model_short}/random_normal_weights_${rand_seed}/params1/${dataset_name}/eval_at_ckpt-0_orient_tuning

	echo -e "\nloading from $save_eval_dir\n"
	echo -e "\nsaving to $reduced_dir\n"

	# do PCA on big patterns
	echo python reduce_activations.py \
	 --activ_path=${save_eval_dir} \
	 --reduced_path=${reduced_dir} \
	 --min_components_keep=10 \
	 --pctVar=95 \
	 --num_batches=${num_batches} \
	 --model_name=${which_model}
	python reduce_activations.py \
	 --activ_path=${save_eval_dir} \
	 --reduced_path=${reduced_dir} \
	 --min_components_keep=10 \
	 --pctVar=95 \
	 --num_batches=${num_batches} \
	 --model_name=${which_model}

	# also measure orient tuning curves
	echo python get_orient_tuning.py ${save_eval_dir} ${tuning_dir} ${which_model} ${dataset_name} ${num_batches}
	python get_orient_tuning.py ${save_eval_dir} ${tuning_dir} ${which_model} ${dataset_name} ${num_batches}

done

training_str=random_normal_weights_${rand_seed}
model_short=${which_model//_/}
nSamples=${#sets[@]}
python analyze_orient_tuning.py ${ROOT} ${model_short} ${training_str} ${dataset_root} ${nSamples} ${which_hyperpars} ${step_num}

echo "finished analyzing/fitting tuning curves!"
