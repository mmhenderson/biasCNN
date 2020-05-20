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
from sklearn import decomposition

def get_rand_weights(rand_seed,root,model='vgg16_simul'):  
  """ Get a set of random weights for each layer of the network.
  """
  
  if 'vgg16_simul'==model:
    nLayers = 19
    info = load_activations.get_info('vgg16','CosGratings',root=root)
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
    
  return layer_weights


def get_activ_single_batch(root,batch_num, image_set, rand_seed,model='vgg16_simul'):
  """ Pass images through this "fake" random convnet 
      Save the resulting activation patterns.
  """
  rand_seed = int(rand_seed)
  print('\nroot is %s'%root)
  print('batch num is %d'%batch_num)
  print('image set is %s'%image_set)
  print('rand seed is %s'%rand_seed)
  print('model is %s\n\n'%model)
  
  # first get information about the images we are going to load
  info = load_activations.get_info('vgg16', image_set,root=root)
  exlist = info['exlist'] 
  sf_vals = info['sf_vals']
  contrast_levels = info['contrast_levels']
  contrastlist =info['contrastlist'] 
  orilist = info['orilist']
  sflist = info['sflist']
  phaselist = info['phaselist']
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
  layer_weights = get_rand_weights(rand_seed,root=root,model=model)
  
  for xx in range(batch_size):
      
      ii = ims2do[xx]
      image_file = os.path.join(root,'images','gratings',image_set,
                             'SF_%0.2f_Contrast_%0.2f'%(sf_vals[sflist[ii]], contrast_levels[int(contrastlist[ii][0])]),
                             'Gaussian_phase%d_ex%d_%ddeg.png'%(phaselist[ii][0],exlist[ii]+1,orilist[ii]))
   
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
          
          activ = conv_ops.pool(activ,2)
          
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


def reduce_weights(root,image_set, rand_seed,model='vgg16_simul',min_components_keep=10,pctVar=95,n_components_keep=600):
  """ Reduce big network activations using PCA.
  """  
  
  # path information
  big_folder = os.path.join(root,'activations',model,'random_normal_weights_%s'%rand_seed,'params1',image_set,'eval_at_ckpt-0_full')
  reduced_folder = os.path.join(root,'activations',model,'random_normal_weights_%s'%rand_seed,'params1',image_set,'eval_at_ckpt-0_reduced')
  if not os.path.isdir(reduced_folder):
    os.makedirs(reduced_folder)
  
  path2load = big_folder
  path2save = reduced_folder
  
  if model=='vgg16_simul':
    nLayers = 19
    num_batches = 96
    info = load_activations.get_info('vgg16', image_set,root=root)
    layer_names = info['layer_labels'][0:nLayers]
  else:
    raise ValueError('model name not recognized')
  
  # loop over layers and reduce each layer
  for ll in range(nLayers):
      allw = None
      
      # loop over batches and concatenate them to a big matrix
      for bb in np.arange(0,num_batches):

          file = os.path.join(path2load, 'batch' + str(int(bb)) +'_' + layer_names[ll] +'.npy')
          print('loading from %s\n' % file)
          w = np.squeeze(np.load(file))
          # w will be nIms x nFeatures
          w = np.reshape(w, [np.shape(w)[0], np.prod(np.shape(w)[1:])])
          
          if bb==0:
              allw = w
          else:
              allw = np.concatenate((allw, w), axis=0)
  
      #% Run  PCA on this weight matrix to reduce its size
      pca = decomposition.PCA(n_components = np.min((n_components_keep, np.shape(allw)[1])))
      print('\n STARTING PCA WITH %d COMPONENTS MAX\n'%(n_components_keep))
      print('size of allw before reducing is %d by %d'%(np.shape(allw)[0],np.shape(allw)[1]))
      weights_reduced = pca.fit_transform(allw)   
      
      var_expl = pca.explained_variance_ratio_
      
      n_comp_needed = np.where(np.cumsum(var_expl)>pctVar/100)
      if np.size(n_comp_needed)==0:
        n_comp_needed = n_components_keep
        print('need >%d components to capture %d percent of variance' % (n_comp_needed, pctVar))
      else:
        n_comp_needed = n_comp_needed[0][0]
        print('need %d components to capture %d percent of variance' % (n_comp_needed, pctVar))
        
      if n_comp_needed<min_components_keep:
          n_comp_needed = min_components_keep
    
      weights_reduced = weights_reduced[:,0:n_comp_needed]
    
      print('saving %d components\n'%np.shape(weights_reduced)[1])
      #%% Save the result as a single file
      
      fn2save = os.path.join(path2save, 'allStimsReducedWts_' + layer_names[ll] +'.npy')
      
      np.save(fn2save, weights_reduced)
      print('saving to %s\n' % (fn2save))
      
      fn2save = os.path.join(path2save, 'allStimsVarExpl_' + layer_names[ll] +'.npy')
          
      np.save(fn2save, var_expl)
      print('saving to %s\n' % (fn2save))
      
  return