#!/bin/bash
#SBATCH --partition=bigmem_long
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

dataset_root=PhaseVaryingCosGratings
which_model=vgg_16
# num of versions of this dataset (phases are different)
#sf_vals=(0.01 0.02 0.04 0.08 0.14 0.25)
declare -a sf_vals=(0.14)


# first define the folder where all checkpoint for this model will be located
model_short=${which_model//_/}

# this specifies the exact file for the trained model we want to look at.
ckpt_file=${ROOT}biasCNN/logs/vgg16/ImageNet/scratch_imagenet_rot_0_stop_early/params1/model.ckpt-0

echo "evaluating un-trained model"
	
# loop over number of versions of this dataset.
for sf in ${sf_vals[@]}
do
	
	dataset_name=${dataset_root}_SF_${sf}

	#source ~/anaconda3/bin/activate
	${ROOT}biasCNN/code/shell_scripts/EvalPretrainedModel/get_activ_un-trained_single.sh ${which_model} ${dataset_name} ${ROOT} ${ckpt_file} ${overwrite} ${TEST}
	
done

