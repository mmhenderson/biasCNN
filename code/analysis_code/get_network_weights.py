#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some code to tensors with their weights from network after some amount of training. 

"""

import tensorflow as tf
import os 

import numpy as np
#import matplotlib.pyplot as plt
#import scipy

#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as prt
from tensorflow.contrib.framework import arg_scope as arg_scope
from tensorflow.python import pywrap_tensorflow

import slim.nets.vgg as vgg
from slim.nets.vgg import vgg_16 as vgg_16

slim = tf.contrib.slim
model='vgg16';  
root = '/usr/local/serenceslab/maggie/biasCNN/';
param_str='params1'
training_str = 'scratch_imagenet_rot_45_cos'
ckpt_num_approx='400000'

#%%
def get_weights_trained_vgg16(training_str,param_str,ckpt_num_approx):
  
  ckpt_dir=os.path.join(root, 'logs', model,'ImageNet',training_str,param_str);
  
  # list all checkpoint files in this directory
  files = os.listdir(ckpt_dir)  
  ckpt_files = [ff for ff in files if 'model.ckpt-' in ff and '.meta' in ff]
  
  # identify the numbers of all saved checkpoints (not necessarily round numbers)
  nums=[ff[np.char.find(ff,'-')+1:np.char.find(ff,'.meta')] for ff in ckpt_files]
  nums=np.sort([int(nn) for nn in nums])
  
  # find the soonest checkpoint after the number specified
  ckpt_num = str(nums[np.where(nums>=int(ckpt_num_approx))[0][0]])
  
  ckptfile = os.path.join(root, 'logs', model,'ImageNet',training_str,param_str,'model.ckpt-%s'%ckpt_num)
  metafile = os.path.join(root, 'logs', model,'ImageNet',training_str,param_str,'model.ckpt-%s.meta'%ckpt_num)
 
  # file to save the extracted weights
  save_path = os.path.join(root,'weights',model,training_str,param_str)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    
  #%Get a list of the names of all tensors of interest, and their sizes

  # list of all the layers i'm interested in
  layers2load= ['conv1_1','conv1_2',
     'conv2_1','conv2_2',
     'conv3_1','conv3_2','conv3_3',
     'conv4_1','conv4_2','conv4_3',
     'conv5_1','conv5_2','conv5_3',
     'fc6',
     'fc7',
     'fc8']
  reader = pywrap_tensorflow.NewCheckpointReader(ckptfile)
  var_to_shape_map = reader.get_variable_to_shape_map()
  
  # long list of every single tensor (includes extra variables)
  tensor_list = list(var_to_shape_map.keys())
#  size_list = list(var_to_shape_map.values())
  
  # make a shorter list of just the layers i am interested in here
  tensor_sizes = dict()
  tensors2load = []
  for kk in range(np.size(layers2load)):
    weights_tensor = [ii for ii in tensor_list if layers2load[kk] in ii and 'weights' in ii and 'RMSProp' not in ii]
    weights_tensor = weights_tensor[0]
    biases_tensor = [ii for ii in tensor_list if layers2load[kk] in ii and 'biases' in ii and 'RMSProp' not in ii]
    biases_tensor = biases_tensor[0]
    
    tensor_sizes[weights_tensor] = var_to_shape_map[weights_tensor]
    tensor_sizes[biases_tensor] = var_to_shape_map[biases_tensor]
    
    tensors2load.append('%s:0'%weights_tensor)
      
  #% get the weights for each of these tensors, at the specified training step
  w_all = []
  
  tf.reset_default_graph()
  
  with tf.Session() as sess:
      
      saver = tf.train.import_meta_graph(metafile)
      saver.restore(sess, ckptfile)
      # get the graph
      g = tf.get_default_graph()
    
      w_all = [sess.run(g.get_tensor_by_name(tt)) for tt in tensors2load]
    
  #% save the result
  save_name =os.path.join(save_path,'AllNetworkWeights_eval_at_ckpt_%s0000.npy'%(ckpt_num_approx[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,w_all)

 #%%
def get_weights_pretrained_vgg16(training_str):

  ckptfile_pretrained = os.path.join(root, 'checkpoints', 'vgg16_ckpt','vgg_16.ckpt')

  # file to save the extracted weights
  save_path = os.path.join(root,'weights',model,training_str,param_str)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # list of all the layers i'm interested in
  layers2load= ['conv1_1','conv1_2',
     'conv2_1','conv2_2',
     'conv3_1','conv3_2','conv3_3',
     'conv4_1','conv4_2','conv4_3',
     'conv5_1','conv5_2','conv5_3',
     'fc6',
     'fc7',
     'fc8']
  reader = pywrap_tensorflow.NewCheckpointReader(ckptfile_pretrained)
  var_to_shape_map = reader.get_variable_to_shape_map()
  
  # long list of every single tensor (includes extra variables)
  tensor_list = list(var_to_shape_map.keys())
#  size_list = list(var_to_shape_map.values())
  
  # make a shorter list of just the layers i am interested in here
  tensor_sizes = dict()
  tensors2load = []
  for kk in range(np.size(layers2load)):
    weights_tensor = [ii for ii in tensor_list if layers2load[kk] in ii and 'weights' in ii and 'RMSProp' not in ii]
    weights_tensor = weights_tensor[0]
    biases_tensor = [ii for ii in tensor_list if layers2load[kk] in ii and 'biases' in ii and 'RMSProp' not in ii]
    biases_tensor = biases_tensor[0]
    
    tensor_sizes[weights_tensor] = var_to_shape_map[weights_tensor]
    tensor_sizes[biases_tensor] = var_to_shape_map[biases_tensor]
    
    tensors2load.append('%s:0'%weights_tensor)
  
  # using the pre-trained model
  # don't have a meta file here so we have to use the .py file to define the graph
  # note this block of code will NOT run for the model once we've re-trained it, 
  # because the names of the layers are slightly different 
  tf.reset_default_graph()
  
  w_all=[]

  with tf.Session() as sess:
      
      graph = tf.get_default_graph()
      with graph.as_default():
    
          with arg_scope(vgg.vgg_arg_scope()):
             # passing a bunch of blank images in, just as a placeholder
             batch_size = 10
             image_size = 224
             inputs = tf.placeholder(tf.float32, shape=(batch_size,
                                                           image_size, image_size, 3))
# 
             [logits, end_pts] = vgg_16(inputs, num_classes=1000,
                         is_training=False,spatial_squeeze=False)
  
          # restore the model from saved checkpt
          saver = tf.train.Saver()
          saver.restore(sess, ckptfile_pretrained)
          
          w_all = [sess.run(graph.get_tensor_by_name(tt)) for tt in tensors2load]

 #% save the result
  save_name =os.path.join(save_path,'AllNetworkWeights_eval_at_ckpt_%s0000.npy'%(ckpt_num_approx[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,w_all)
#%% main function to decide between subfunctions
  
if __name__=='__main__':
  
  if 'pretrained' in training_str:
    get_weights_pretrained_vgg16(training_str)
    
  else:
    get_weights_trained_vgg16(training_str,param_str,ckpt_num_approx)
  