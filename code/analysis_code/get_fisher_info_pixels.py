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
dataset_all='FiltIms11Cos_SF_0.01'
nSets = 4

model = 'pixel'
nPix = 224**2
sf_vals = [0.01, 0.02, 0.04, 0.08, 0.14, 0.25]

  
# values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)

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
      
  save_path = os.path.join(root,'code','fisher_info',model,dataset)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
      
  info = load_activations.get_info(model,dataset)

  # extract some things from info
  orilist = info['orilist']
  exlist=info['exlist']
  sflist = info['sflist']
  if info['nPhase']==1:
    nOri=180
    orilist_adj = orilist
  else:
    nOri=360
    orilist_adj = deepcopy(orilist)
    phaselist = info['phaselist']
    orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
   
  nLayers = info['nLayers']
  nSF = info['nSF']
  if 'AllSF' in dataset_all:
    sf2do = [0]
  else:
    sf2do = np.arange(0,nSF);

  nIms = np.size(info['orilist'])
  
  #%% first make the "activation matrix" which is just pixel values
  w = np.zeros([nIms, nPix])
  print('loading images...\n')
  for ii in range(nIms):
    
    if 'FiltIms' in dataset:
      im_file = os.path.join(root,'images','gratings',dataset,'AllIms','FiltImage_ex%d_%ddeg.png'%(exlist[ii]+1,orilist[ii]))
    else:
      im_file = os.path.join(root,'images','gratings',dataset,
                           'SF_%0.2f_Contrast_0.80'%(sf_vals[sflist[ii][0]]),
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
  
  nLayers=1
  fisher_info = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  deriv2 = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  varpooled = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
    
  for sf in sf2do:

    dat = w
        
    inds = np.where(sflist==sf)[0]
    
    for dd in range(np.size(delta_vals)):
        
      ori_axis, fi, d, v = classifiers.get_fisher_info(dat[inds,:],orilist_adj[inds],delta=delta_vals[dd])
      if nOri==360:
        fi = np.reshape(fi,[2,180])
        fi = np.mean(fi,axis=0)
        d = np.reshape(d,[2,180])
        d = np.mean(d,axis=0)
        v = np.reshape(v,[2,180])
        v = np.mean(v,axis=0)
        
      fisher_info[0,sf,:,dd] = np.squeeze(fi)
      deriv2[0,sf,:,dd] = np.squeeze(d)
      varpooled[0,sf,:,dd] = np.squeeze(v)
        
  save_name =os.path.join(save_path,'Fisher_info_pixels.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,fisher_info)
  
  save_name =os.path.join(save_path,'Deriv_sq_pixels.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,deriv2)
  
  save_name =os.path.join(save_path,'Pooled_var_pixels.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,varpooled)