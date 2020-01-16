#!/bin/bash

# delete extra checkpoint files out of my log folder - only
# once the network is fully trained and i know which ckpts are of interest!
ROOT=/mnt/neurocube/local/serenceslab/maggie/

#my_dir=${ROOT}biasCNN/logs/vgg16/ImageNet/scratch_vgg16_imagenet_rot_0/weightdecay_0.00005_rmspropdecay_0.90_rmspropmomentum_0.80_learningrate_0.005_learningratedecay_0.94/
my_dir=${ROOT}biasCNN/logs/vgg16/ImageNet/scratch_vgg16_imagenet_rot_45/weightdecay_0.00005_rmspropdecay_0.90_rmspropmomentum_0.80_learningrate_0.005_learningratedecay_0.94/

echo -e "\nSEARCHING IN THIS DIR:\n${my_dir}\n"

start=500000
stop=750000
step=50000
# these are approximate - will keep the first checkpoint after each of these numbers
step2keep_list_approx=($(seq $start $step $stop))
# this one we also need to keep, since i used it (semi randomly) for getting activations
step2keep_list_approx+=(694497 685119 689830)	

# this we will fill in with the real numbers
declare -a step2keep_list=()
declare -a step2delete_list=()

# first list all the numbers of the checkpoints in the folder
allckptfiles=$(ls ${my_dir}model.ckpt-*.meta) 
declare -a all_numbers=()
for file in ${allckptfiles[@]}
do
	second_part=${file/*ckpt-}
	number=${second_part/.*}
	all_numbers+=($number)
done

# now find the exact ckpt number for each of the approx ones	
for step2keep in ${step2keep_list_approx[@]}
do			
	# search the whole list
	current_number=10000000	# set this to a huge value to start
	orig_number=${current_number}
	for number in ${all_numbers[@]}
	do		
		if (( $number>=$step2keep )) && (( $number<$current_number ))
		then
			current_number=$number
		fi
	done
	
	if [[ $current_number == $orig_number ]]
	then		
		echo "can't find a checkpoint past ${step2keep}"
	else
		#echo "Checkpoint number to KEEP: ${current_number}"
		step2keep_list+=(${current_number})
	fi
	
done

echo -e "KEEPING THESE CHECKPOINTS:\n${step2keep_list[@]}\n"

# now figure out which to delete (all others)
for number in ${all_numbers[@]}
do
	bad=1
	for step2keep in ${step2keep_list[@]}
	do
		if [[ $step2keep == $number ]]
		then
			bad=0	
		fi
	done
	if [[ $bad == 1 ]]
	then
		step2delete_list+=($number)
	fi
done

echo -e "DELETING THESE CHECKPOINTS:\n${step2delete_list[@]}\n"
echo -e "Really delete now? [y/n]"
read response 

if [[ $response == y ]]
then
	echo "Going to delete extra files now!"
	for step2delete in ${step2delete_list[@]}
	do
		files2delete=$(ls ${my_dir}model.ckpt-${step2delete}*)
		echo "${files2delete[@]}"
		for file in ${files2delete[@]}
		do
			echo "deleting: ${file}"
			rm ${file}
		done
	done
fi



