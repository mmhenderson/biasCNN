#!/bin/bash
#SBATCH --partition=bigmem_long
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=500000
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

set -e

# GET ACTIVATIONS FOR A MODEL ON MULTIPLE DATASETS (EVALUATION IMAGES)
#ROOT=/cube/neurocube/local/serenceslab/maggie/
ROOT=/mnt/neurocube/local/serenceslab/maggie/

# am i over-writing old folders, or checking which exist already?
overwrite=0
TEST=0

dataset_root=FiltNoiseCos_SF_0.14
which_model=vgg_16
# num of versions of this dataset (phases are different)
nSets=0

# first define the folder where all checkpoint for this model will be located
model_short=${which_model//_/}

# this specifies the exact file for the trained model we want to look at.
ckpt_file=${ROOT}biasCNN/checkpoints/vgg16_ckpt/vgg_16.ckpt

echo "evaluating pretrained model"
	
# loop over number of versions of this dataset.
for ss in $(seq 0 $nSets)
do
	if [ "${ss}" = "0" ]
	then
		dataset_name=$dataset_root
	else
		dataset_name=${dataset_root}${ss}
	fi

	#source ~/anaconda3/bin/activate
	${ROOT}biasCNN/code/shell_scripts/EvalPretrainedModel/get_activ_pretrained_single.sh ${which_model} ${dataset_name} ${ROOT} ${ckpt_file} ${overwrite} ${TEST}
	
done

