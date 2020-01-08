#!/bin/bash
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Convert my modified ImageNET images into tfrecord format.
# Loops over rotations and saves separate dataset for each.
# MMH 10/29/19

set -e

# Specify the directory i am working in
#ROOT=/usr/local/serenceslab/maggie/
ROOT=/cube/neurocube/local/serenceslab/maggie/
SCRIPT_DIR=${ROOT}tensorflow/models/research/slim/datasets/

# where are the raw images?
IMAGE_DIR=${ROOT}biasCNN/images/ImageNet/ILSVRC2012/

# where will i save the tfrecord files?
DATASET_DIR=${ROOT}biasCNN/datasets/ImageNet/ILSVRC2012/

LABELS_FILE="${IMAGE_DIR}imagenet_lsvrc_2015_synsets.txt"
BOUNDING_BOX_FILE="${IMAGE_DIR}bounding_boxes/imagenet_2012_bounding_boxes.csv"
IMAGENET_METADATA_FILE="${IMAGE_DIR}imagenet_metadata.txt"
 
declare -a rot_list=(0 22 45)

for rot in ${rot_list[@]}
do   

    # Where the training/testing sets of images are now
    TRAIN_DIRECTORY="${IMAGE_DIR}train_rot_${rot}_square/"
    VALIDATION_DIRECTORY="${IMAGE_DIR}validation_rot_${rot}_square/"
    
    # where i'll put all the tfrecord files
    OUTPUT_DIRECTORY="${DATASET_DIR}tfrecord_rot_${rot}_square/"
    
    mkdir -p ${OUTPUT_DIRECTORY}
    
    # Build the TFRecords version of the ImageNet data.
    BUILD_SCRIPT="${SCRIPT_DIR}build_imagenet_data.py"
    
    python ${BUILD_SCRIPT} \
      --train_directory="${TRAIN_DIRECTORY}" \
      --validation_directory="${VALIDATION_DIRECTORY}" \
      --output_directory="${OUTPUT_DIRECTORY}" \
      --imagenet_metadata_file="${IMAGENET_METADATA_FILE}" \
      --labels_file="${LABELS_FILE}" \
      --bounding_box_file="${BOUNDING_BOX_FILE}"

done
