#!/bin/bash
#SBATCH --partition=bigmem_long
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

source ~/anaconda3/bin/activate
/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/EvalPretrainedVGG16/eval_pretrained_vgg16_SquareGratings_cluster.sh

#/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/EvalImageNetRots/#eval_pretrained_vgg16_SpatFreqGratings_cluster.sh 

