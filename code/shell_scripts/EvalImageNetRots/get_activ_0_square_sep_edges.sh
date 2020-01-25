#!/bin/bash
#SBATCH --partition=bigmem_long
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
#ROOT=/cube/neurocube/local/serenceslab/maggie/
ROOT=/mnt/neurocube/local/serenceslab/maggie/

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

rot=0_square
which_hyperpars=params1
dataset_root=SpatFreqGratings
which_model=vgg_16
# num of versions of this dataset (phases are different)
nSets=1

# what steps to evaluate at? make a nice sequence here even though the real checkpoints are not round numbers
start=450000
stop=450000
step=50000
# these are approximate - will keep the first checkpoint after each of these numbers
step2eval_list_approx=($(seq $start $step $stop))

# first define the folder where all checkpoint for this model will be located
model_short=${which_model//_/}
log_dir=${ROOT}/biasCNN/logs/${model_short}/ImageNet/scratch_imagenet_rot_${rot}/${which_hyperpars}/
#log_dir=${ROOT}/biasCNN/logs/${model_short}/ImageNet/scratch_vgg16_imagenet_rot_${rot}/weightdecay_0.00005_rmspropdecay_0.90_rmspropmomentum_0.80_learningrate_0.005_learningratedecay_0.94_init1/

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

# START EVALUATION LOOP
# First looping over steps (ckpts)
for step_range in ${step2eval_list_approx[@]}
do
	# figure out what is the exact number of the checkpoint of interest
	current_number=10000000	# set this to a huge value to start	
	orig_number=$current_number
	for number in ${all_numbers[@]}
	do		
		if (( $number>=$step_range )) && (( $number<$current_number ))
		then			
			current_number=$number
		fi
	done
	if [[ $current_number == $orig_number ]]
	then
		echo "could not find a checkpoint following $step_range. Aborting..."
		exit
	fi
	step_num=$current_number

	echo "evaluating on ckpt number ${step_num[@]}"
	
	# Second loop over number of versions of this dataset.
	for ss in $(seq 0 $nSets)
	do
		if [ "${ss}" = "0" ]
		then
			dataset_name=$dataset_root
		else
			dataset_name=${dataset_root}${ss}
		fi

		#source ~/anaconda3/bin/activate
		${ROOT}biasCNN/code/shell_scripts/EvalImageNetRots/get_activ_single_sep_edges.sh ${rot} ${step_num} ${which_hyperpars} ${which_model} ${dataset_name} ${ROOT} ${log_dir} ${overwrite} ${TEST}
		
	done
done
