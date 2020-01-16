#!/bin/bash
#SBATCH --partition=general_gpu_k40
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

ROOT=/cube/neurocube/local/serenceslab/maggie/
#ROOT=/mnt/neurocube/local/serenceslab/maggie/
declare -a rot_list=(45 45 45 45)
declare -a step_range_list=(600000 700000 800000)	# will search for the first checkpoint that exists LATER than this one.
declare -a pars_list=(params1 params1 params1 params1)
nEvals=4

source ~/anaconda3/bin/activate

for ii in $(seq 0 $(($nEvals-1)))
do
	
	rot=${rot_list[$ii]}
	step_range=${step_range_list[$ii]}
	which_hyperpars=${pars_list[$ii]}

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

	# figure out which checkpoint to use, based on which ones exist.
	allckptfiles=$(ls ${log_dir}model.ckpt-*.meta) 

	current_number=10000000	# set this to a huge value to start
	orig_number=${current_number}
	for file in ${allckptfiles[@]}
	do
		second_part=${file/*ckpt-}
		number=${second_part/.*}
		if (( $number>$step_range )) && (( $number<$current_number ))
		then
			current_number=$number
		fi
	done
	
	if [[ $current_number == $orig_number ]]
	then		
		echo "can't find a checkpoint past ${step_range}"
	else
		echo "Checkpoint number to use: ${current_number}"
		step_num=${current_number}
	fi
	/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/EvalImageNetRots/eval_performance_only.sh ${rot} ${step_num} ${which_hyperpars} ${log_dir} ${ROOT}

done
