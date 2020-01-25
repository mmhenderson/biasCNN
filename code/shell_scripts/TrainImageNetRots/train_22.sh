#!/bin/bash
#SBATCH --partition=general_gpu_p6000
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 

model=vgg_16
params=params1
rot=22
from_scratch=0

source ~/anaconda3/bin/activate
/cube/neurocube/local/serenceslab/maggie/biasCNN/code/shell_scripts/TrainImageNetRots/train_net_cluster.sh $model $params $rot $from_scratch 

