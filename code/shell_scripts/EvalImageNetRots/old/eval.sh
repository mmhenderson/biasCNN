#!/bin/bash
#SBATCH --partition=bigmem_long
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

rot=0
step_num=689475
which_hyperpars=params1

source ~/anaconda3/bin/activate
/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/EvalImageNetRots/eval_vgg16_SquareGratings_cluster.sh ${rot} ${step_num} ${which_hyperpars}

#/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/EvalImageNetRots/#eval_vgg16_SpatFreqGratings_cluster.sh ${rot} ${step_num} ${which_hyperpars}

