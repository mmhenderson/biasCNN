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

dataset_all = 'FiltImsCos'
#dataset_all = 'FiltImsAllSFCos'
#dataset_all = 'FiltNoiseCos'
#dataset_all = 'PhaseVaryingCosGratings'
#dataset_all='CosGratings'
#dataset_all = 'CircGratings'
nSets = 4;
loopSF = 1;
sf_vals = [0.01, 0.02, 0.04, 0.08, 0.14, 0.25]

model='vgg16'
param_str='params1'
training_str='scratch_imagenet_rot_0_cos'
#training_str='scratch_imagenet_rot_0_stop_early'
#training_str='pretrained'
#training_str = 'scratch_imagenet_rot_22_square'

#ckpt_str='0'
ckpt_str = '400000'


#%% use if all spatial frequencies are in one dataset
  
def get_discrim_func(dataset_all, model, param_str, training_str, ckpt_str):

  save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset_all)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # define folder corresponding to this data set
  dataset_dir = os.path.join(root, 'activations', model, training_str, param_str, dataset_all)
  
  # find the exact name of the checkpoint file of interest
  ckpt_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dataset_dir))
  nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in ckpt_dirs]
  
  # compare the first two characters    
  good2 = [jj for jj in range(np.size(ckpt_dirs)) if 'reduced' in ckpt_dirs[jj] and not 'sep_edges' in ckpt_dirs[jj] and ckpt_str[0:2] in nums[jj][0:2]]
  assert(np.size(good2)==1)
  ckpt_dir = ckpt_dirs[good2[0]]
  ckpt_num= ckpt_dir.split('_')[2][5:]

  # load activations 
  w, varexpl, info = load_activations.load_activ(model, dataset_dir, training_str, param_str, ckpt_num)
 
  # extract some things from info
  orilist = info['orilist']
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
  discrim = np.zeros([nLayers, nSF, 180])
  fisher5 = np.zeros([nLayers, nSF, 180])
  fisher2 = np.zeros([nLayers, nSF, 180])
  discrim5 = np.zeros([nLayers,nSF,180])
  discrim_binned = np.zeros([nLayers,nSF,180])
  
  for sf in sf2do:
    
   
    # loop over layers and get discriminability curve for each.
    for ll in range(nLayers):

        if np.size(w[ll][0])==0:
          print('missing data for layer %s\n'%info['layer_labels'][ll])          
          continue
        
        dat = w[ll][0]
        
        inds = np.where(sflist==sf)[0]
        
        ori_axis, fi = classifiers.get_fisher_info(dat[inds,:],orilist_adj[inds],delta=5)
        if nOri==360:
          fi = np.reshape(fi,[2,180])
          fi = np.mean(fi,axis=0)
        fisher5[ll,sf,:] = np.squeeze(fi)
        
        ori_axis, fi = classifiers.get_fisher_info(dat[inds,:],orilist_adj[inds],delta=2)
        if nOri==360:
          fi = np.reshape(fi,[2,180])
          fi = np.mean(fi,axis=0)
        fisher2[ll,sf,:] = np.squeeze(fi)
        
        ori_axis, disc = classifiers.get_discrim_func(dat[inds,:],orilist_adj[inds])
        if nOri==360:
          disc = np.reshape(disc,[2,180])
          disc = np.mean(disc,axis=0)
        discrim[ll,sf,:] = np.squeeze(disc)
        
        ori_axis, disc = classifiers.get_discrim_func_binned(dat[inds,:],orilist_adj[inds],bin_size=5)
        if nOri==360:
          disc = np.reshape(disc,[2,180])
          disc = np.mean(disc,axis=0)
        discrim_binned[ll,sf,:] = np.squeeze(disc)
        
        ori_axis, disc = classifiers.get_discrim_func(dat[inds,:],orilist_adj[inds],step_size=5)
        if nOri==360:
          disc = np.reshape(disc,[2,180])
          disc = np.mean(disc,axis=0)
        discrim5[ll,sf,:] = np.squeeze(disc)
       
  # checkpoint number gets rounded here to make it easier to find later
  save_name =os.path.join(save_path,'Fisher_info_delta5_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,fisher5)
  
  save_name =os.path.join(save_path,'Fisher_info_delta2_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,fisher2)
  
  save_name =os.path.join(save_path,'Discrim_func_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,discrim)
  
  # checkpoint number gets rounded here to make it easier to find later
  save_name =os.path.join(save_path,'Discrim_func_5degsteps_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,discrim5)
  
  save_name =os.path.join(save_path,'Discrim_func_binned_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,discrim_binned)


#%% use if spatial frequencies are in different datasets
  
def get_discrim_func_sfloop(sf_vals, dataset_all, model, param_str, training_str, ckpt_str):
    
  save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset_all)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  
  nSF = len(sf_vals)
  for sf in range(nSF):
    
    if 'FiltIms' in dataset_all:
      # define folder corresponding to this data set
      dataset_root = dataset_all.split('_')[0]
      ss = dataset_all[-1]
      dataset_dir = os.path.join(root, 'activations', model, training_str, param_str, '%s_SF_%.2f_rand%s'%(dataset_root, sf_vals[sf],ss))
    else:
      # define folder corresponding to this data set
      dataset_dir = os.path.join(root, 'activations', model, training_str, param_str, '%s_SF_%.2f'%(dataset_all, sf_vals[sf]))
      
    # find the exact name of the checkpoint file of interest
    ckpt_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dataset_dir))
    nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in ckpt_dirs]
    
    # compare the first two characters    
    good2 = [jj for jj in range(np.size(ckpt_dirs)) if 'reduced' in ckpt_dirs[jj] and not 'sep_edges' in ckpt_dirs[jj] and ckpt_str[0:2] in nums[jj][0:2]]
    assert(np.size(good2)==1)
    ckpt_dir = ckpt_dirs[good2[0]]
    ckpt_num= ckpt_dir.split('_')[2][5:]
   
    # load activations 
    w, varexpl, info = load_activations.load_activ(model, dataset_dir, training_str, param_str, ckpt_num)
    
    # extract some things from info
    nLayers = info['nLayers']
    orilist = info['orilist']
    assert(np.all(info['sflist']==0))
    if info['nPhase']==1:
      nOri=180
      orilist_adj = orilist
    else:
      nOri=360
      orilist_adj = deepcopy(orilist)
      phaselist = info['phaselist']
      orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
     
    # initialize my array here
    if sf==0:  
      fisher5 = np.zeros([nLayers, nSF, 180])
      fisher2 = np.zeros([nLayers, nSF, 180])
      discrim = np.zeros([nLayers, nSF, 180])
      discrim5 = np.zeros([nLayers,nSF,180])
      discrim_binned = np.zeros([nLayers,nSF,180])
    
    # loop over layers and get discriminability curve for each.
    for ll in range(nLayers):

        if np.size(w[ll][0])==0:
          print('missing data for layer %s\n'%info['layer_labels'][ll])          
          continue
        
        dat = w[ll][0]
      
        ori_axis, fi = classifiers.get_fisher_info(dat,orilist_adj,delta=5)
        if nOri==360:
          fi = np.reshape(fi,[2,180])
          fi = np.mean(fi,axis=0)
        fisher5[ll,sf,:] = np.squeeze(fi)
        
        ori_axis, fi = classifiers.get_fisher_info(dat,orilist_adj,delta=2)
        if nOri==360:
          fi = np.reshape(fi,[2,180])
          fi = np.mean(fi,axis=0)
        fisher2[ll,sf,:] = np.squeeze(fi)
      
        ori_axis, disc = classifiers.get_discrim_func(dat,orilist_adj)  
        if nOri==360:
          disc = np.reshape(disc,[2,180])
          disc = np.mean(disc,axis=0)        
        discrim[ll,sf,:] = np.squeeze(disc)
        
        ori_axis, disc = classifiers.get_discrim_func_binned(dat,orilist_adj,bin_size=5) 
        if nOri==360:
          disc = np.reshape(disc,[2,180])
          disc = np.mean(disc,axis=0)
        discrim_binned[ll,sf,:] = np.squeeze(disc)
        
        ori_axis, disc = classifiers.get_discrim_func(dat,orilist_adj,step_size=5)       
        if nOri==360:
          disc = np.reshape(disc,[2,180])
          disc = np.mean(disc,axis=0)
        discrim5[ll,sf,:] = np.squeeze(disc)
       
  # checkpoint number gets rounded here to make it easier to find later
  save_name =os.path.join(save_path,'Fisher_info_delta5_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,fisher5)
  
  save_name =os.path.join(save_path,'Fisher_info_delta2_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,fisher2)
   
  save_name =os.path.join(save_path,'Discrim_func_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,discrim)
  
  save_name =os.path.join(save_path,'Discrim_func_5degsteps_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,discrim5)
  
  save_name =os.path.join(save_path,'Discrim_func_binned_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,discrim_binned)

#%% main function to decide between subfunctions
  
if __name__=='__main__':
  
  if loopSF==1:
    if nSets==1:
      get_discrim_func_sfloop(sf_vals, dataset_all, model, param_str, training_str, ckpt_str)
    else:
      for ss in range(nSets):        
        if ss==0 and 'FiltIms' not in dataset_all:
          curr_dataset=dataset_all
        elif 'FiltIms' in dataset_all:
          curr_dataset='%s_rand%d'%(dataset_all,ss+1)
        else:
          curr_dataset='%s%d'%(dataset_all,ss)
        get_discrim_func_sfloop(sf_vals, curr_dataset, model, param_str, training_str, ckpt_str)
  else:    
    if nSets>1:
      for ss in range(nSets):
        
        if ss==0 and 'FiltIms' not in dataset_all:
          curr_dataset=dataset_all
        elif 'FiltIms' in dataset_all:
          curr_dataset='%s_rand%d'%(dataset_all,ss+1)
        else:
          curr_dataset='%s%d'%(dataset_all,ss)
        
        get_discrim_func(curr_dataset, model, param_str, training_str, ckpt_str)
    
    else:
      get_discrim_func(dataset_all, model, param_str, training_str, ckpt_str)

