#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=500000
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
ROOT=/cube/neurocube/local/serenceslab/maggie/
#ROOT=/mnt/neurocube/local/serenceslab/maggie/

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

rand_seed=456456
init_str=random_normal_weights_${rand_seed}
which_hyperpars=params1
dataset_root=FiltIms5AllSFCos
which_model=vgg16_simul
step_num=0
declare -a sets=(1 2 3 4)

# first define the folder where all checkpoint for this model will be located
model_short=${which_model}

for set in ${sets[@]}
do

	dataset_name=${dataset_root}_rand${set}	
	
	#source ~/anaconda3/bin/activate
	${ROOT}biasCNN/code/shell_scripts/SimulateModel/get_tuning_single_simul.sh ${init_str} ${which_model} ${dataset_name} ${ROOT} ${log_dir} ${overwrite} ${TEST}

done

# now do the analysis of these tuning curves, combining across all the image sets...
codepath=${ROOT}biasCNN/code/analysis_code/
cd ${codepath}

nSamples=${#sets[@]}
python analyze_orient_tuning.py ${ROOT} ${which_model} ${init_str} ${dataset_root} ${nSamples} ${which_hyperpars} ${step_num}

echo "finished analyzing/fitting tuning curves!"

