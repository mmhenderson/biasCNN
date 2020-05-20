#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:07:15 2020

@author: mmhender

pass gratings through a network with random initializations, save a matrix of the result
# this is a (slow) manual way to simulate evaluating the VGG16 network

"""

import numpy as np
import conv_ops
import os
from PIL import Image
import load_activations
import sys

def get_rand_weights(root,rand_seed,model='vgg16_simul'):  
  """ Get a set of random weights for each layer of the network.
  """
  
  # file to save the generated weights
  save_path = os.path.join(root,'weights',model,'random_normal_weights_%d'%rand_seed,'params1')
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    
  if 'vgg16' in model:
    nLayers = 19
    info = load_activations.get_info('vgg16','CosGratings')
    layer_names = info['layer_labels'][0:nLayers]
    # information about the network architecture
    num_in_channels=   [3, 64,64,64, 128,128,128,256,256,256,256,512,512,512,512,512,512,512,512]
    num_out_channels = [64,64,64,128,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096]
  else:
    raise ValueError('model name not recognized')

  layer_weights = []
  
  np.random.seed(rand_seed) 
  
  for ll in range(nLayers):    
    if 'conv' in layer_names[ll]:
      weights_rand = np.random.normal(size=[3,3,num_in_channels[ll],num_out_channels[ll]])      
    elif 'pool' in layer_names[ll]:      
      weights_rand = []
    elif 'fc6' in layer_names[ll]:
      weights_rand = np.random.normal(size=[7,7,num_in_channels[ll],num_out_channels[ll]])
  
    layer_weights.append(weights_rand)
    
  #% save the result
  save_name = os.path.join(save_path,'AllNetworkWeights_eval_at_ckpt_00000.npy')
  print('saving to %s\n'%save_name)
  w_all=layer_weights
  np.save(save_name,w_all)  

  return layer_weights

def get_constant_weights(root,model='vgg16_simul_nofilter'):  
  """ define weights all equal to 1.
  """
  
  # file to save the generated weights
  save_path = os.path.join(root,'weights',model,'constant_weights_%d'%rand_seed,'params1')
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    
  if 'vgg16' in model:
    nLayers = 19
    info = load_activations.get_info('vgg16','CosGratings')
    layer_names = info['layer_labels'][0:nLayers]
    # information about the network architecture
    num_in_channels=   [3, 64,64,64, 128,128,128,256,256,256,256,512,512,512,512,512,512,512,512]
    num_out_channels = [64,64,64,128,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096]
  else:
    raise ValueError('model name not recognized')

  layer_weights = []
  
#  np.random.seed(rand_seed) 
  
  for ll in range(nLayers):    
    if 'conv' in layer_names[ll]:
      weights = np.ones(size=[3,3,num_in_channels[ll],num_out_channels[ll]])      
    elif 'pool' in layer_names[ll]:      
      weights = []
    elif 'fc6' in layer_names[ll]:
      weights = np.ones(size=[7,7,num_in_channels[ll],num_out_channels[ll]])
  
    layer_weights.append(weights)
    
  #% save the result
  save_name = os.path.join(save_path,'AllNetworkWeights_eval_at_ckpt_00000.npy')
  print('saving to %s\n'%save_name)
  w_all=layer_weights
  np.save(save_name,w_all)
    
  return layer_weights


def get_activ_single_batch(root,batch_num, image_set, rand_seed,model='vgg16_simul'):
  """ Pass images through this "fake" random convnet 
      Save the resulting activation patterns.
  """
  rand_seed = int(rand_seed)
  
  
  # first get information about the images we are going to load
  info = load_activations.get_info('vgg16', image_set)
  exlist = info['exlist'] 
#  sf_vals = info['sf_vals']
#  contrast_levels = info['contrast_levels']
#  contrastlist =info['contrastlist'] 
  orilist = info['orilist']
#  sflist = info['sflist']
#  phaselist = info['phaselist']
  nIms= np.size(orilist)
  
  # which images are in my current batch?
  num_batches = 96
  batch_size = int(nIms/num_batches)
  which_batch = np.repeat(range(num_batches),batch_size)
  ims2do = np.where(which_batch==batch_num)[0]

  # for now only using the first 19 layers
  layer_names = info['layer_labels'][0:19]
  nLayers = np.size(layer_names)
  
  # path information
  save_folder = os.path.join(root,'activations',model,'random_normal_weights_%s'%rand_seed,'params1',image_set,'eval_at_ckpt-0_full')
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

  # initialize activations
  activ_patterns = []
  
  # get the actual network weights we'll use for this whole batch
  if 'nofilter' in model:
    layer_weights = get_constant_weights(root,model=model)
  else:
    layer_weights = get_rand_weights(root,rand_seed,model=model)
    
  for xx in range(batch_size):
      
      ii = ims2do[xx]
      image_file = os.path.join(root,'images','gratings',image_set,
                             'AllIms', 'FiltImage_ex%d_%ddeg.png'%(exlist[ii]+1,orilist[ii]))
   
      print('batch %d of %d\nloading from %s'%(batch_num,num_batches,image_file))
      im = Image.open(image_file)
      im = np.reshape(np.array(im.getdata()),[224,224,3])
      
      activ = im
      
      # process each layer with this image
      for ll in range(np.size(layer_names)):
        
        if 'conv' in layer_names[ll]:
          weights_rand = layer_weights[ll]
          activ = conv_ops.conv(activ,weights_rand,1)
          
        elif 'pool' in layer_names[ll]:
          
          activ = conv_ops.max_pool(activ,2)
          
        elif 'fc6' in layer_names[ll]:
          weights_rand = layer_weights[ll]
          activ = conv_ops.conv(activ,weights_rand,1)    
    
    
        activ = conv_ops.relu(activ)
        
        # flatten matrix and add this to my big array
        activ_flat = np.ravel(activ)
        
        if ii==ims2do[0]:
          # appending an array to store weights from this layer
          activ_patterns.append(np.zeros([batch_size, np.size(activ_flat)]))
        
        # putting the pattern from this image, this layer into the array
        activ_patterns[ll][xx,:] = activ_flat 
      
      
  for ll in range(nLayers):
      fn2save = os.path.join(save_folder,'batch' + str(int(batch_num)) +'_' + layer_names[ll] +'.npy')
      print('\nSAVING TO %s\n'%fn2save)
      np.save(fn2save,activ_patterns[ll])


if __name__ == '__main__':
  
  root=sys.argv[1]
  batch_num=int(sys.argv[2])
  image_set=sys.argv[3]
  rand_seed=int(sys.argv[4])
  model=sys.argv[5]
  
  print('\nroot is %s'%root)
  print('batch num is %d'%batch_num)
  print('image set is %s'%image_set)
  print('rand seed is %s'%rand_seed)
  print('model is %s\n\n'%model)
  
  get_activ_single_batch(root,batch_num, image_set, rand_seed,model)