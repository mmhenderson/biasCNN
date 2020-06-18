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

#dataset_all = 'FiltIms14AllSFCos'
dataset_all = 'FiltIms11Cos'
#dataset_all = 'CosGratings'
#training_str_list=['scratch_imagenet_rot_45_square']
training_str_list = ['scratch_imagenet_rot_45_stop_early']
#training_str_list = ['scratch_imagenet_rot_0_stop_early','pretrained','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos']
#training_str_list = ['pretrained']
nSets = 4;
loopSF = 1;
sf_vals = [0.01, 0.02, 0.04, 0.08, 0.14, 0.25]
#sf_vals = [0.01, 0.02, 0.04, 0.08, 0.14]


model='vgg16'
param_str='params1'


# values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)

#%% use if all spatial frequencies are in one dataset
  
def get_discrim_func(dataset, model, param_str, training_str, ckpt_str):

  save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  # define folder corresponding to this data set
  dataset_dir = os.path.join(root, 'activations', model, training_str, param_str, dataset)
  
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
  if 'AllSF' in dataset:
    sf2do = [0]
  else:
    sf2do = np.arange(0,nSF);

  fisher_info = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  deriv2 = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  varpooled = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  
  for sf in sf2do:
    
   
    # loop over layers and get discriminability curve for each.
    for ll in range(nLayers):

        if np.size(w[ll][0])==0:
          print('missing data for layer %s\n'%info['layer_labels'][ll])          
          continue
        
        dat = w[ll][0]
        
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
            
          fisher_info[ll,sf,:,dd] = np.squeeze(fi)
          deriv2[ll,sf,:,dd] = np.squeeze(d)
          varpooled[ll,sf,:,dd] = np.squeeze(v)

  save_name =os.path.join(save_path,'Fisher_info_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,fisher_info)
  
  save_name =os.path.join(save_path,'Deriv_sq_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,deriv2)
  
  save_name =os.path.join(save_path,'Pooled_var_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,varpooled)

#%% use if spatial frequencies are in different datasets
  
def get_discrim_func_sfloop(sf_vals, dataset, model, param_str, training_str, ckpt_str):
    
  save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset)
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  
  nSF = len(sf_vals)
  for sf in range(nSF):
    
    if 'FiltIms' in dataset:
      # define folder corresponding to this data set
      dataset_root = dataset.split('_')[0]
      ss = dataset[-1]
      dataset_dir = os.path.join(root, 'activations', model, training_str, param_str, '%s_SF_%.2f_rand%s'%(dataset_root, sf_vals[sf],ss))
    else:
      # define folder corresponding to this data set
      dataset_dir = os.path.join(root, 'activations', model, training_str, param_str, '%s_SF_%.2f'%(dataset, sf_vals[sf]))
      
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
      fisher_info = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
      deriv2 = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
      varpooled = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  
    # loop over layers and get discriminability curve for each.
    for ll in range(nLayers):

        if np.size(w[ll][0])==0:
          print('missing data for layer %s\n'%info['layer_labels'][ll])          
          continue
        
        dat = w[ll][0]
      
        for dd in range(np.size(delta_vals)):
          
          ori_axis, fi, d, v = classifiers.get_fisher_info(dat,orilist_adj,delta=delta_vals[dd])
          if nOri==360:
            fi = np.reshape(fi,[2,180])
            fi = np.mean(fi,axis=0)
            d = np.reshape(d,[2,180])
            d = np.mean(d,axis=0)
            v = np.reshape(v,[2,180])
            v = np.mean(v,axis=0)
            
          fisher_info[ll,sf,:,dd] = np.squeeze(fi)
          deriv2[ll,sf,:,dd] = np.squeeze(d)
          varpooled[ll,sf,:,dd] = np.squeeze(v)
       
  # checkpoint number gets rounded here to make it easier to find later
  save_name =os.path.join(save_path,'Fisher_info_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,fisher_info)

  save_name =os.path.join(save_path,'Deriv_sq_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,deriv2)
  
  save_name =os.path.join(save_path,'Pooled_var_eval_at_ckpt_%s0000_all_units.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,varpooled)
  
#%% main function to decide between subfunctions
  
if __name__=='__main__':
  
  for tr in range(np.size(training_str_list)):
    
    training_str = training_str_list[tr]
  
    if 'pretrained' in training_str or 'stop_early' in training_str:
      ckpt_str = '0'
    else:
      ckpt_str = '400000'
      
    if loopSF==1:
     
#      if nSets==1:
#        get_discrim_func_sfloop(sf_vals, dataset_all, model, param_str, training_str, ckpt_str)
#      else:
      for ss in range(nSets):        
        if ss==0 and 'FiltIms' not in dataset_all:
          curr_dataset=dataset_all
        elif 'FiltIms' in dataset_all:
          curr_dataset='%s_rand%d'%(dataset_all,ss+1)
        else:
          curr_dataset='%s%d'%(dataset_all,ss)
        get_discrim_func_sfloop(sf_vals, curr_dataset, model, param_str, training_str, ckpt_str)
    else:    
#      if nSets>1:
      for ss in range(nSets):
        
        if ss==0 and 'FiltIms' not in dataset_all:
          curr_dataset=dataset_all
        elif 'FiltIms' in dataset_all:
          curr_dataset='%s_rand%d'%(dataset_all,ss+1)
        else:
          curr_dataset='%s%d'%(dataset_all,ss)
        
       
        get_discrim_func(curr_dataset, model, param_str, training_str, ckpt_str)
        
#      else:
#        get_discrim_func(dataset_all, model, param_str, training_str, ckpt_str)
  
