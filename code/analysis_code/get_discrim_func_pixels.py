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

#dataset_all='FiltImsAllSFCos'
#dataset_all = 'CosGratings'
dataset_all = 'CircGratings'
#dataset_all = 'SpatFreqGratings'
#dataset_all = 'SquareGratings'
nSets = 4

model = 'pixel'
nPix = 224**2

# looping over the datasets
#for dd in range(nSets): 
for dd in np.arange(0,nSets):
  
  if 'FiltIms' in dataset_all:
    dataset = '%s_rand%d'%(dataset_all,dd+1)
  else:
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
#  contrastlist = np.int8(info['contrastlist'])
  exlist = info['exlist']
  sf_vals = info['sf_vals']
#  contrast_levels = info['contrast_levels']
#  phase_vals = info['phase_vals']
  nSF_here = np.size(np.unique(sflist))
  
  if info['nPhase']==2:
    # treat the orientation space as a 0-360 space since we have to go around the 180 space twice to account for phase.    
    orilist_adj = deepcopy(orilist)
    orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
    ori_axis = np.arange(0.5, 360,1)
    nOri=360
  else:
    orilist_adj=orilist
    ori_axis = np.arange(0.5, 180,1)
    nOri=180
    
  nIms = np.size(info['orilist'])
  
  #%% first make the "activation matrix" which is just pixel values
  w = np.zeros([nIms, nPix])
  print('loading images...\n')
  for ii in range(nIms):
    
    if 'FiltIms' in dataset:
      im_file = os.path.join(root,'images','gratings',dataset,'AllIms','FiltImage_ex%d_%ddeg.png'%(exlist[ii]+1,orilist[ii]))
    else:
      im_file = os.path.join(root,'images','gratings',dataset,
                           'SF_%0.2f_Contrast_0.80'%(sf_vals[sflist[ii]]),
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
  n_comp_needed = np.where(np.cumsum(var_expl)>pctVar/100)[0]
  if np.size(n_comp_needed)==0:
    n_comp_needed = n_components_keep
  else:
    n_comp_needed = n_comp_needed[0]
    
  w = weights_reduced[:,0:n_comp_needed]
#  # now, removing all pixels that will be the same over images
#  # (these are where the background is masked to zero)
#  good_inds = np.where([w[0,:]!=w[1,:]])[1]
#  
#  w = w[:,good_inds]
  #%% now get discriminability curves for each spatial freq.
  
  discrim = np.zeros([nLayers,nSF_here,180])
  discrim5 = np.zeros([nLayers,nSF_here,180])
  
  for sf in range(nSF_here):

    inds = np.where(sflist==sf)[0]
    
    dat = w
    
    ori_axis, disc = classifiers.get_discrim_func(dat[inds,:],orilist_adj[inds])
    if nOri==360:
      disc = np.reshape(disc,[2,180])
      disc = np.mean(disc,axis=0)
    discrim[0,sf,:] = np.squeeze(disc)
  
    ori_axis, disc = classifiers.get_discrim_func_step_out(dat[inds,:],orilist_adj[inds],step_size=5)
    if nOri==360:
      disc = np.reshape(disc,[2,180])
      disc = np.mean(disc,axis=0)
    discrim5[0,sf,:] = np.squeeze(disc)
        
  save_path = os.path.join(root,'code','discrim_func',model,dataset)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # checkpoint number gets rounded here to make it easier to find later
  save_name =os.path.join(save_path,'Discrim_func_pixels.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,discrim)

# checkpoint number gets rounded here to make it easier to find later
  save_name =os.path.join(save_path,'Discrim_func_5degsteps_pixels.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,discrim5)