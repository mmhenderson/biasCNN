#!/bin/bash
#SBATCH --partition=bigmem_long
#SBATCH --gres=gpu:0
#SBATCH --mail-user=xxx@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=500000
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
CWD=$(pwd)
cd ../../
ROOT=$(pwd)

rand_seed_jitter=434895

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

rot=0_cos
which_hyperpars=params1_init1
dataset_root=FiltIms14AllSFCos
which_model=vgg_16
declare -a sets=(1 2 3 4)

# what steps to evaluate at? will find checkpoint closest to this.
step_approx=400000

# first define the folder where all checkpoint for this model will be located
model_short=${which_model//_/}
log_dir=${ROOT}/logs/${model_short}/ImageNet/scratch_imagenet_rot_${rot}/${which_hyperpars}/

if [ ! -d ${log_dir} ]
then
	echo "ERROR: Your log dir doesn't exist!!"
	exit
fi

# now list all the numbers of the checkpoints in the folder
allckptfiles=$(ls ${log_dir}model.ckpt-*.meta) 
declare -a all_numbers=()
for file in ${allckptfiles[@]}
do
	second_part=${file/*ckpt-}
	number=${second_part/.*}
	all_numbers+=($number)
done


# figure out what is the exact number of the checkpoint of interest
current_number=10000000	# set this to a huge value to start	
orig_number=$current_number
for number in ${all_numbers[@]}
do		
	if (( $number>=$step_approx )) && (( $number<$current_number ))
	then			
		current_number=$number
	fi
done
if [[ $current_number == $orig_number ]]
then
	echo "could not find a checkpoint following $step_approx. Aborting..."
	exit
fi
step_num=$current_number
echo "evaluating on ckpt number ${step_num[@]}"

for set in ${sets[@]}
do

	dataset_name=${dataset_root}_rand${set}	
	
	#source ~/anaconda3/bin/activate
	${CWD}/get_tuning_and_fisher_info_single.sh ${rot} ${step_num} ${which_hyperpars} ${which_model} ${dataset_name} ${ROOT} ${log_dir} ${overwrite} ${TEST}

done

# now do the analysis of these tuning curves, combining across all the image sets...
codepath=${ROOT}/code/analysis_code/
cd ${codepath}

training_str=scratch_imagenet_rot_${rot}
nSamples=${#sets[@]}
python analyze_orient_tuning_jitter.py ${ROOT} ${model_short} ${training_str} ${dataset_root} ${nSamples} ${which_hyperpars} ${step_num} ${rand_seed_jitter}

echo "finished analyzing/fitting tuning curves!"

