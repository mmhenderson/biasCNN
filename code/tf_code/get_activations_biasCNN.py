# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluate model on an image (grating) dataset. 
Pass images through network to get activation patterns that are saved as numpy files. 
Dataset is split into several batches, loop over one batch at a time.
MODIFIED from eval_image_classifier.py (from tf-slim library) by MMH 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.getcwd())

import math
import tensorflow as tf

from datasets import dataset_biasCNN
from nets import nets_factory_biasCNN
nets_factory = nets_factory_biasCNN
from preprocessing import preprocessing_biasCNN

#import os
import numpy as np

slim = tf.contrib.slim

print('Using packages at the following paths:')
print(dataset_biasCNN.__file__)
print(nets_factory.__file__)
print(preprocessing_biasCNN.__file__)
print(tf.__file__)
print(slim.__file__)

#%%
tf.app.flags.DEFINE_boolean(
    'is_windowed', True, 'Boolean for whether images are already scaled and have circular gauss mask imposed. If true, do not resize or crop.')

tf.app.flags.DEFINE_integer(
    'batch_size',90, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_batches',96, 'The number of batches.')

tf.app.flags.DEFINE_integer(
    'num_classes',180,'The number of classes to place images in.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', 1,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'append_scope_string', None, 'The name of the scope used in the checkpoint '
    'file, which is appended at the start of each layer name.')

tf.app.flags.DEFINE_string(
    'eval_dir', '', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', '', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', '', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

print(FLAGS.checkpoint_path)

def main(_):
  if not FLAGS.dataset_dir:
      raise ValueError('You must supply the dataset directory with --dataset_dir')

  num_batches= FLAGS.num_batches

  for bb in np.arange(0,num_batches):
    
      batch_name = 'batch'+str(bb)

      tf.logging.set_verbosity(tf.logging.INFO)
  
      with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
    
        ######################
        # Select the dataset #
        ######################
        dataset = dataset_biasCNN.get_dataset(
            FLAGS.dataset_name, batch_name, FLAGS.dataset_dir, num_classes=FLAGS.num_classes)
    
        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)
    
        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=1,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset
    
    
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_biasCNN.get_preprocessing(
            preprocessing_name,
            is_training=False, flipLR=False,random_scale=False, 
	    is_windowed=FLAGS.is_windowed)
    
        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    
        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    
        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, end_pts = network_fn(images)

        if FLAGS.moving_average_decay:
          variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, tf_global_step)
          variables_to_restore = variable_averages.variables_to_restore(
              slim.get_model_variables())
          variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            if FLAGS.append_scope_string:
                # If I've specified a string for the name of the scope in the checkpoint file, append it here so we can match up the layer names
                variables_to_restore_orig = slim.get_variables_to_restore()    
                variables_to_restore = {}
                for var in variables_to_restore_orig:
                    curr_name = var.op.name
                    if 'global_step' not in curr_name:
                        new_name = FLAGS.append_scope_string + '/' + curr_name
                    else:
                        new_name = curr_name 
                    variables_to_restore[new_name]=  var
            else:                    
                variables_to_restore = slim.get_variables_to_restore()
    
        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)
    
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })
    
        # Print the summaries to screen.
        for name, value in names_to_values.items():
          summary_name = 'eval/%s' % name
          op = tf.summary.scalar(summary_name, value, collections=[])
          op = tf.Print(op, [value], summary_name)
          tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
    
        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
          num_batches = FLAGS.max_num_batches
        else:
          # This ensures that we make a single pass over all of the data.
          num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
    
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
          checkpoint_path = FLAGS.checkpoint_path
    
        tf.logging.info('Evaluating %s' % checkpoint_path)
    
        out = slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            final_op={'logits':logits, 'end_pts':end_pts,'images':images,'labels':labels,'predictions':predictions},
            variables_to_restore=variables_to_restore)
           
    
        end_pts= out['end_pts']
        
        keylist= list(end_pts.keys())
     
        for kk in range(np.size(keylist)):
            keystr = keylist[kk]
            keystr = keystr.replace('/','_') 
            fn2save = FLAGS.eval_dir + '/' + batch_name + '_' + keystr + '.npy'
            np.save(fn2save, end_pts[keylist[kk]])
            
        logits = out['logits']

        labels = out['labels']

        predictions = out['predictions']
    
        fn2save = FLAGS.eval_dir + '/' + batch_name + '_logits.npy'
        np.save(fn2save, logits)

        fn2save = FLAGS.eval_dir + '/' + batch_name + '_labels_orig.npy'
        np.save(fn2save, labels)
          
        fn2save = FLAGS.eval_dir + '/' + batch_name + '_labels_predicted.npy'
        np.save(fn2save, predictions)
         
                                    
if __name__ == '__main__':
  tf.app.run()
