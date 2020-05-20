#!/bin/bash


path=/cube/neurocube/local/serenceslab/maggie/biasCNN/
dataset=CosGratings
bb=9
rand_seed=298484
echo python -c "import get_manual_activ; get_manual_activ.get_activ_single_batch('${path}',$bb,'${dataset}',$rand_seed)"
python -c "import get_manual_activ; get_manual_activ.get_activ_single_batch('${path}',$bb,'${dataset}',$rand_seed)"

#${shell_script_dir}one_batch.sh $ROOT $bb $dataset_name $rand_seed &
		#pwd
		#echo python -c "import get_manual_activ; get_manual_activ.get_activ_single_batch('${ROOT}',$bb,'${dataset_name}',$rand_seed)"
		#python -c "import get_manual_activ; get_manual_activ.get_activ_single_batch('${ROOT}',$bb,'${dataset_name}',$rand_seed)" &


	# loop over image set batches and run them in parallel.
	#for bb in $(seq 0 $num_batches)
	#do
	#	batch_processes[$bb]=$(echo python get_manual_activ.py ${ROOT} ${bb} ${dataset_name} ${rand_seed} ${which_model})
	#done


		#echo $batch_process
		#$batch_process &
		#pwd
