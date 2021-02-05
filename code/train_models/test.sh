#!/bin/bash
#SBATCH --partition=general_gpu_p6000
#SBATCH --gres=gpu:0
#SBATCH --mail-user=xxx@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 

CWD=$(pwd)
cd ../../
ROOT=$(pwd)

source activate tf_env

cat /proc/driver/nvidia/version

dpkg -l | grep nvidia-driver 

nvidia-smi
