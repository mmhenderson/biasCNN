#!/bin/bash
#SBATCH --partition=general_gpu_k80
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

source ~/anaconda3/bin/activate

ROOT=/cube/neurocube/local/serenceslab/maggie/
#ROOT=/mnt/neurocube/local/serenceslab/maggie/
rot=0
which_hyperpars=params1

# what steps to evaluate at? make a nice sequence here even though the real checkpoints are not round numbers
start=600000
stop=750000
step=50000
# these are approximate - will keep the first checkpoint after each of these numbers
step2eval_list_approx=($(seq $start $step $stop))
nReps=3

# now more file path stuff...
# if using the older file naming system, need to make a long string describing the parameters...
if [[ $which_hyperpars == "params1" ]]
then
	parsfull=weightdecay_0.00005_rmspropdecay_0.90_rmspropmomentum_0.80_learningrate_0.005_learningratedecay_0.94
elif [[ $which_hyperpars == "params2" ]]
then
	parsfull=weightdecay_0.00005_rmspropdecay_0.90_rmspropmomentum_0.80_learningrate_0.001_learningratedecay_0.94
else
	raise error "params not found"
fi

log_dir=${ROOT}biasCNN/logs/vgg16/ImageNet/scratch_vgg16_imagenet_rot_${rot}/${parsfull}/
#log_dir=${ROOT}biasCNN/logs/vgg16/ImageNet/scratch_imagenet_rot_${rot}/${which_hyperpars}/

# list all the numbers of the checkpoints in the folder
allckptfiles=$(ls ${log_dir}model.ckpt-*.meta) 
declare -a all_numbers=()
for file in ${allckptfiles[@]}
do
	second_part=${file/*ckpt-}
	number=${second_part/.*}
	all_numbers+=($number)
done

# loop over the number of evaluations i want to do
for step_range in ${step2eval_list_approx[@]}
do
	# figure out what is the exact number of the checkpoint of interest
	current_number=10000000	# set this to a huge value to start
	#orig_number=${current_number}
	for number in ${all_numbers[@]}
	do		

		if (( $number>=$step_range )) && (( $number<$current_number ))
		then			
			current_number=$number
		fi
	done
	step_num=$current_number
	
	for ii in $(seq 0 $nReps)
    do
    	echo "eval on ckpt number ${step_num[@]}, iter ${ii}"
    	/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/EvalImageNetRots/eval_performance_only_TEST.sh ${rot} ${step_num} ${which_hyperpars} ${log_dir} ${ROOT}
  	done

done
