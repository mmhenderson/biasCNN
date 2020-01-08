#!/bin/bash
#SBATCH --partition=bigmem_long
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL

# location of the local folder on ssrde cluster
root_root=/cube/neurocube/local/
root=${root_root}serenceslab/maggie/biasCNN/
matlab_version=${root_root}MATLAB/R2018b/bin/matlab
${matlab_version} -nodisplay -nosplash -nodesktop -r "cd '${root}code/image_proc_code/';get_orientation_stats_imagenet_rots('${root}', 8);exit"

