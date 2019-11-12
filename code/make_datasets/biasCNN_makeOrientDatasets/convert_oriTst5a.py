# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Make the dataset (tfrecord) files for grating stimuli. 
    This script will load gratings with different levels of noise. These will be used for model TESTING.
    The training images are completely distinct - even the 0 noise condition has different randomized phase for training and testing sets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
#import random
import sys
import numpy as np
import shutil

import tensorflow as tf

from slim.datasets import dataset_utils

#%% set up some useful constants and parameters

# The percntage of images in the validation set.
_PCT_VAL = 0.10

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 10

dataset_dir_root = '/usr/local/serenceslab/maggie/biasCNN/datasets/oriTst5a/'
image_dir = '/usr/local/serenceslab/maggie/biasCNN/grating_ims_5a/'
  
dataset_name = 'oriTst5a'
# information about the stimuli. 
sf_vals = [2.20]
stim_types = ['Gaussian']
nOri=180
nSF=1
nPhase=16
nType=1
nNoiseLevels = 3
noise_levels = [0.20,0.40,0.60]

# list all the image features in a big matrix, where every row is unique.
noiselist = np.expand_dims(np.repeat(noise_levels, nPhase*nOri*nSF*nType),1)
typelist = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
orilist=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
sflist=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
phaselist=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))

featureMat = np.concatenate((noiselist,typelist,orilist,sflist,phaselist),axis=1)

assert np.array_equal(featureMat, np.unique(featureMat, axis=0))

#%%

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
#    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
#    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

#  def decode_jpeg(self, sess, image_data):
#    image = sess.run(self._decode_jpeg,
#                     feed_dict={self._decode_jpeg_data: image_data})
#    assert len(image.shape) == 3
#    assert image.shape[2] == 3
#    return image

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(image_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """  
  
  # we'll train the model on orientation
  model_labels = orilist
  unlabs = np.unique(model_labels)
  class_names = [] 
  for cc in range(np.size(unlabs)):      
      class_names.append('%.f_deg' % (unlabs[cc]))
    
     
  all_labels = []
  all_filenames = []
#  for nn in np.arange(nNoiseLevels):
  for ii in np.arange(0,np.size(model_labels)):
    subfolder = "SF_%.2f_noise%.2f" % (sf_vals[int(sflist[ii])], noiselist[ii])
    filename = "%s_fixedphase%d_%ddeg.png" % (stim_types[int(typelist[ii])], phaselist[ii]+1, orilist[ii])
    full_fn = os.path.join(image_dir, subfolder, filename)

    all_labels.append(model_labels[ii])
    all_filenames.append(full_fn)
      
  return all_filenames, all_labels, class_names

def _get_dataset_filename(dataset_dir, split_name, num_shards, shard_id):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      dataset_name, split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_labels, dataset_dir, num_shards):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  
  print(split_name)
  
  assert (split_name in ['train', 'validation'] or 'batch' in split_name) 

  num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(num_shards):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, num_shards, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

#            class_name = os.path.basename(os.path.dirname(filenames[i]))
            
            class_id = class_labels[i]
#            class_name = class_names[int(class_id)]
            
            print('\n %s, label %d\n' % (filenames[i],class_id))
            
            example = dataset_utils.image_to_tfexample(
                image_data, b'png', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


#def _clean_up_temporary_files(dataset_dir):
#  """Removes temporary files used to create the dataset.
#
#  Args:
#    dataset_dir: The directory where the temporary files are stored.
#  """
#  filename = _DATA_URL.split('/')[-1]
#  filepath = os.path.join(dataset_dir, filename)
#  tf.gfile.Remove(filepath)
#
#  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
#  tf.gfile.DeleteRecursively(tmp_dir)


#def _dataset_exists(dataset_dir):
#  for split_name in ['train', 'validation','allims']:
#    for shard_id in range(_NUM_SHARDS):
#      output_filename = _get_dataset_filename(
#          dataset_dir, split_name, shard_id)
#      if not tf.gfile.Exists(output_filename):
#        return False
#  return True


def main(argv):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  
#  for nn in range(nNoiseLevels):
      
#      dataset_dir = os.path.join(dataset_dir_root, 'noise%.2f/' % noise_levels[nn])
  
  dataset_dir = dataset_dir_root
  
  # if the folder already exists, we'll automatically delete it and make it again.
  if tf.gfile.Exists(dataset_dir):
    print('deleting')
#        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    shutil.rmtree(dataset_dir, ignore_errors = True)
    tf.gfile.MakeDirs(dataset_dir)
  else:
    tf.gfile.MakeDirs(dataset_dir)

 
#%% get the information for ALL my images (all categories, exemplars, rotations)
    
  all_filenames, all_labels, class_names = _get_filenames_and_classes(image_dir)
 
# save out this list just as a double check that this original order is correct
  np.save(dataset_dir + 'all_filenames.npy', all_filenames)
  np.save(dataset_dir + 'all_labels.npy', all_labels)
  np.save(dataset_dir + 'featureMat.npy', featureMat)

  # Save the test set as a couple "batches" of images. 
  # Doing this manually makes it easy to load them and get their weights later on. 
  n_total_val = np.size(all_labels)
  max_per_batch = int(90);
  num_batches = np.ceil(n_total_val/max_per_batch)
  
  for bb in np.arange(0,num_batches):
     
      bb=int(bb)
      name = 'batch' + str(bb)
      
      if (bb+1)*max_per_batch > np.size(all_filenames):
          batch_filenames = all_filenames[bb*max_per_batch:-1]
          batch_labels = all_labels[bb*max_per_batch:-1]
      else:     
          batch_filenames =all_filenames[bb*max_per_batch:(bb+1)*max_per_batch]
          batch_labels = all_labels[bb*max_per_batch:(bb+1)*max_per_batch]

          assert np.size(batch_labels)==max_per_batch

      _convert_dataset(name, batch_filenames, batch_labels, dataset_dir,num_shards=1)
  
      np.save(dataset_dir + name + '_filenames.npy', batch_filenames)
      np.save(dataset_dir + name + '_labels.npy', batch_labels)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

#  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the noisy grating dataset, with orientation labels!')


if __name__ == "__main__":
  tf.app.run()