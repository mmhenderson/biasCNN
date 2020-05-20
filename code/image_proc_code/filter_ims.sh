#!/bin/bash
#SBATCH --partition=general_short
#SBATCH --gres=gpu:0
#SBATCH --mail-user=mmhender@ucsd.edu
#SBATCH --mail-type=ALL
#SBATCH -o ./sbatch_output/output-%A-%x-%u.out # STDOUT

# location of the local folder on ssrde cluster
root_root=/cube/neurocube/local/
root=${root_root}serenceslab/maggie/biasCNN/
image_set=FiltIms11Cos_SF_0.01_rand1
matlab_version=${root_root}MATLAB/R2018b/bin/matlab
${matlab_version} -nodisplay -nosplash -nodesktop -r "cd '${root}code/image_proc_code/';get_orientation_stats_FiltImageNetIms('${image_set}', '${root}', 'max');exit"

