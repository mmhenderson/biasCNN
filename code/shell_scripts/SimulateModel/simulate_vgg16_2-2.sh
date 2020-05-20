#!/bin/bash
#SBATCH --partition=general_short
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=200000
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
ROOT=/cube/neurocube/local/serenceslab/maggie/biasCNN/
code_dir=${ROOT}code/analysis_code/
shell_script_dir=${ROOT}code/shell_scripts/SimulateModel/
cd $code_dir

nSets=4
dataset_root=FiltIms5AllSFCos
which_model=vgg16_simul
rand_seed=456456
num_batches=95

# num of versions of this dataset (phases are different)
low=24
high=47

# loop over number of versions of this dataset.
for ss in $(seq 3 $nSets)
do
	
	dataset_name=${dataset_root}_rand${ss}
	
	for bb in $(seq $low $high)
	do
		echo python get_manual_activ.py ${ROOT} ${bb} ${dataset_name} ${rand_seed} 
		python get_manual_activ.py ${ROOT} ${bb} ${dataset_name} ${rand_seed} ${which_model} 
		
	done
	
done

