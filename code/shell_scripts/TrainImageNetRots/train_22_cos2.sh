#!/bin/bash
#SBATCH --partition=general_gpu_p6000
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 

rot=22_cos
model=vgg_16
params=params1
from_scratch=1
declare -a inits=(2)
max_steps=500000

source ~/anaconda3/bin/activate

for init_num in ${inits[@]}
do

	/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/TrainImageNetRots/train_net_cluster.sh $model $params $rot $init_num $max_steps

done
