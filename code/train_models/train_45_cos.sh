#!/bin/bash
#SBATCH --partition=general_gpu_p6000
#SBATCH --gres=gpu:0
#SBATCH --mail-user=xxx@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 

rot=45_cos
model=vgg_16
params=params1
from_scratch=1
declare -a inits=(0 1 2 3)
max_steps=500000

CWD=$(pwd)
cd ../../
ROOT=$(pwd)

source activate cuda9

for init_num in ${inits[@]}
do

	$CWD/train_net_cluster.sh $model $params $rot $init_num $max_steps $ROOT

done
