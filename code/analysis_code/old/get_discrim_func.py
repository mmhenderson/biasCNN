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

#dataset_all='FiltNoiseCos'
#dataset_all = 'FiltNoiseSquare'
dataset_all = 'FiltNoiseCos_SF_0.14'
#dataset_all = 'PhaseVaryingCosGratings_SF_0.01'
#dataset_all = 'CosGratings'
#dataset_all = 'SpatFreqGratings'
#dataset_all = 'SquareGratings'

model='vgg16'
param_str='params1'
#training_str='scratch_imagenet_rot_45_square'
#training_str='untrained'
#training_str='pretrained'
training_str = 'scratch_imagenet_rot_0_stop_early'

ckpt_strs=['0']
#ckpt_strs = ['350000','400000','450000']
#ckpt_strs = ['400000']

nCheckpoints = np.size(ckpt_strs)

 # searching for folders corresponding to this data set
dataset_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str))
if ('Shift' in dataset_all or 
  'PhaseVarying' in dataset_all or 
  'Small' in dataset_all or 
  'Big' in dataset_all or
  'SF' in dataset_all or
  'Smoothed' in dataset_all): 
  good = [ii for ii in range(np.size(dataset_dirs)) if dataset_all in dataset_dirs[ii]]
else: 
  good = [ii for ii in range(np.size(dataset_dirs)) \
          if dataset_all in dataset_dirs[ii] and  
          'Shift' not in dataset_dirs[ii] and 
          'PhaseVarying' not in dataset_dirs[ii] and 
          'Small' not in dataset_dirs[ii] and 
          'SF' not in dataset_dirs[ii] and 
          'Big' not in dataset_dirs[ii] and
          'Smoothed' not in dataset_dirs[ii]]
sf_vals = np.round(np.logspace(np.log10(0.02),np.log10(.4),6)*140/224, 2)
  
# looping over the datasets
for dd in good:
   
  # loop over checkpoints
  for tr in range(nCheckpoints):
    
    n_comp = []
 
    ckpt_str = ckpt_strs[tr]
 
    # also searching for all evaluations at different timepts (nearby, all are around the optimal point)
    ckpt_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dataset_dirs[dd]))
    nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in ckpt_dirs]
    
    # compare the first two characters
    
    good2 = [jj for jj in range(np.size(ckpt_dirs)) if 'reduced' in ckpt_dirs[jj] and not 'sep_edges' in ckpt_dirs[jj] and ckpt_str[0:2] in nums[jj][0:2]]
    assert(np.size(good2)==1)
    ckpt_dir = ckpt_dirs[good2[0]]
   
    ckpt_num= ckpt_dir.split('_')[2][5:]
    w, varexpl, info = load_activations.load_activ(model, dataset_dirs[dd], training_str, param_str, ckpt_num)
   
    #%% now get the discriminability curve for each layer.
    
    nLayers = info['nLayers']
    nSF = info['nSF']
    orilist = info['orilist']
    sflist = info['sflist']
    if nSF==1:
      sf_ind = np.where(info['sf_vals'][0]==sf_vals)[0]
      assert(np.size(sf_ind)==1)
      sflist = sf_ind*np.ones(np.shape(sflist))     
      nSF = 6
    else:
      sf_ind = np.arange(0,6)
      
    if 'phase_complement_list' in info:
      phaselist = info['phase_complement_list']
    else:
      phaselist=  info['phaselist']
    
    if info['nPhase']==1:
      orilist_adj= orilist
      nOri=180
    else:      
      # treat the orientation space as a 0-360 space since we have to go around the 180 space twice to account for phase.    
      orilist_adj = deepcopy(orilist)
      orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
      nOri=360
      
    ori_axis = np.arange(0.5, nOri,1)

    discrim = np.zeros([nLayers,nSF,nOri])
    discrim5 = np.zeros([nLayers,nSF,nOri])
    discrim_binned = np.zeros([nLayers,nSF,nOri])
    
    for ll in range(nLayers):
      n_comp.append(np.shape(w[ll][0])[1])
      for sf in sf_ind:
        
        if np.size(w[ll][0])==0:
          continue
        dat = w[ll][0]
        
        inds = np.where(sflist==sf)[0]
        
        ori_axis, disc = classifiers.get_discrim_func(dat[inds,:],orilist_adj[inds])
          
        discrim[ll,sf,:] = np.squeeze(disc)
        
        ori_axis, disc_binned = classifiers.get_discrim_func_binned(dat[inds,:],orilist_adj[inds],bin_size=5)
        
        discrim_binned[ll,sf,:] = np.squeeze(disc_binned)
               
        ori_axis, disc5 = classifiers.get_discrim_func_step_out(dat[inds,:],orilist_adj[inds],step_size=5)
          
        discrim5[ll,sf,:] = np.squeeze(disc5)
        
    
    print('\nfor %s %s %s: ncomp needed is\n\n'%(model,training_str,dataset_dirs[dd]))
    print(n_comp)
    save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset_dirs[dd])
    if not os.path.exists(save_path):
      os.makedirs(save_path)
  
    # checkpoint number gets rounded here to make it easier to find later
    save_name =os.path.join(save_path,'Discrim_func_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)    
    np.save(save_name,discrim)

    save_name =os.path.join(save_path,'Discrim_func_5degsteps_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)    
    np.save(save_name,discrim5)
    
    save_name =os.path.join(save_path,'Discrim_func_binned_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)    
    np.save(save_name,discrim_binned)
