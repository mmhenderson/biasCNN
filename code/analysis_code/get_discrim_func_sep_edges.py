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

#%% set up paths and decide what datasets to look at here

root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))


dataset_all = 'SpatFreqGratings'
#dataset_all = 'SquareGratings'

model='vgg16'
param_str='params1'
training_str='scratch_imagenet_rot_45_square'

ckpt_strs=['450000']

nCheckpoints = np.size(ckpt_strs)

part_strs=['center_units','edge_units']
nParts = np.size(part_strs)

 # searching for folders corresponding to this data set
dataset_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str))
good = [ii for ii in range(np.size(dataset_dirs)) if dataset_all in dataset_dirs[ii]]

# looping over the datasets
for dd in good:
   
  # loop over checkpoints
  for tr in range(nCheckpoints):
 
    ckpt_str = ckpt_strs[tr]
 
    # also searching for all evaluations at different timepts (nearby, all are around the optimal point)
    ckpt_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dataset_dirs[dd]))
    nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in ckpt_dirs]
    
    # compare the first two characters
    good2 = [jj for jj in range(np.size(ckpt_dirs)) if 'reduced_sep_edges' in ckpt_dirs[jj] and ckpt_str[0:2] in nums[jj][0:2]]
    if np.size(good2)<1:
      print('no files found for %s %s'%(dataset_dirs[dd],ckpt_str))
      continue
    else:
      assert(np.size(good2)==1)
    ckpt_dir = ckpt_dirs[good2[0]]
   
    ckpt_num= ckpt_dir.split('_')[2][5:]
    w, varexpl, info = load_activations.load_activ_sep_edges(model, dataset_dirs[dd], training_str, param_str, ckpt_num)
   
    #%% now get the discriminability curve for each layer.
    
    nLayers = info['nLayers']
    nSF = info['nSF']
    orilist = info['orilist']
    sflist = info['sflist']
    phaselist=  info['phaselist']
    
    # treat the orientation space as a 0-360 space since we have to go around the 180 space twice to account for phase.    
    orilist_adj = deepcopy(orilist)
    orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
    ori_axis = np.arange(0.5, 360,1)
    
    # loop over parts (center/edges)
    for pp in range(nParts):
  
      discrim = np.zeros([nLayers,nSF,360])
      for ll in range(nLayers):
        for sf in range(nSF):
          
          dat = w[ll][pp]
          
          inds = np.where(sflist==sf)[0]
          
          ori_axis, disc = classifiers.get_discrim_func(dat[inds,:],orilist_adj[inds])
            
          discrim[ll,sf,:] = np.squeeze(disc)
          
        
      save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset_dirs[dd])
      if not os.path.exists(save_path):
        os.makedirs(save_path)
    
      # saving a single file for each of center/edge units
      # checkpoint number gets rounded here to make it easier to find later
      save_name =os.path.join(save_path,'Discrim_func_eval_at_ckpt_%s0000_%s.npy'%(ckpt_num[0:2], part_strs[pp]))
      print('saving to %s\n'%save_name)
      
      np.save(save_name,discrim)
