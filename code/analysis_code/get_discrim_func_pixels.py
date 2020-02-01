#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate the discriminability curve (orientation discriminability versus orientation)
for activations at each layer of network, within each spatial frequency. Save the result. 

Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import os

import numpy as np

from copy import deepcopy

import load_activations

import classifiers_custom as classifiers    

from PIL import Image

from sklearn import decomposition
#%% set up paths and decide what datasets to look at here

root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))

#dataset_all = 'CircGratings'
#dataset_all = 'SpatFreqGratings'
dataset_all = 'SquareGratings'
nSets = 4

model = 'pixel'
nPix = 224**2

# looping over the datasets
for dd in range(nSets): 
  
  if dd==0:
    dataset = dataset_all
  else:
    dataset = '%s%d'%(dataset_all,dd)
    
  info = load_activations.get_info(model,dataset)

  nLayers = info['nLayers']
  nSF = info['nSF']
  orilist = info['orilist']
  sflist = info['sflist']
  phaselist=  info['phaselist']
  contrastlist = np.int8(info['contrastlist'])
  exlist = info['exlist']
  sf_vals = info['sf_vals']
  contrast_levels = info['contrast_levels']
  phase_vals = info['phase_vals']
  
  # treat the orientation space as a 0-360 space since we have to go around the 180 space twice to account for phase.    
  orilist_adj = deepcopy(orilist)
  orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
  ori_axis = np.arange(0.5, 360,1)

  nIms = np.size(info['orilist'])
  
  #%% first make the "activation matrix" which is just pixel values
  w = np.zeros([nIms, nPix])
  print('loading images...\n')
  for ii in range(nIms):
    
    im_file = os.path.join(root,'images','gratings',dataset,
                           'SF_%0.2f_Contrast_%0.2f'%(sf_vals[sflist[ii]], contrast_levels[contrastlist[ii][0]]),
                           'Gaussian_phase%d_ex%d_%ddeg.png'%(phaselist[ii][0],exlist[ii]+1,orilist[ii]))
#    print('loading from %s\n'%im_file)
    image = Image.open(im_file)
    
    #convert to grayscale    
    image = image.convert('L')
    
    # unwrap to a vector 
    w[ii,:] = image.getdata()
  
  
  n_components_keep = 800
  pctVar=95
  print('reducing activations with pca...\n')
  pca = decomposition.PCA(n_components = np.min((n_components_keep, np.shape(w)[1])))
  weights_reduced = pca.fit_transform(w)  
  var_expl = pca.explained_variance_ratio_        
  n_comp_needed = np.where(np.cumsum(var_expl)>pctVar/100)[0][0]
  
  w = weights_reduced[:,0:n_comp_needed]
#  # now, removing all pixels that will be the same over images
#  # (these are where the background is masked to zero)
#  good_inds = np.where([w[0,:]!=w[1,:]])[1]
#  
#  w = w[:,good_inds]
  #%% now get discriminability curves for each spatial freq.
  
  discrim = np.zeros([nLayers,nSF,360])
  
  for sf in range(nSF):

    inds = np.where(sflist==sf)[0]
    
    dat = w[inds,:]
    
    
    ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
      
    discrim[0,sf,:] = np.squeeze(disc)
    
        
  save_path = os.path.join(root,'code','discrim_func',model,dataset)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # checkpoint number gets rounded here to make it easier to find later
  save_name =os.path.join(save_path,'Discrim_func_pixels.npy')
  print('saving to %s\n'%save_name)
  
  np.save(save_name,discrim)
