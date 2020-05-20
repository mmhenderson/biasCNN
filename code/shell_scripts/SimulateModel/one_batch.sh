#!/bin/bash

set -e
## 	GATHER MY ARGUMENTS AND PRINT THEM OUT FOR VERIFICATION
ROOT=${1}
# what dataset?
bb=${2}
# what is my file path root?
dataset_name=${3}
# what is my ckpt file?
rand_seed=${4}

echo "	root=$ROOT"
echo "	bb=$bb"
echo " 	dataset_name=$dataset_name"
echo "  rand_seed=$rand_seed"

echo python -c "import get_manual_activ; get_manual_activ.get_activ_single_batch('${ROOT}',$bb,'${dataset_name}',$rand_seed)"
python -c "import get_manual_activ; get_manual_activ.get_activ_single_batch('${ROOT}',$bb,'${dataset_name}',$rand_seed)" 


