#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import os
import numpy as np
import load_activations
import scipy.optimize

#%% get the data ready to go...then can run any below cells independently.
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))

nSamples =4
model='vgg16'
param_str='params1'

training_str_list = ['scratch_imagenet_rot_0_cos'];
dataset_all = 'FiltImsAllSFCos'
 
# collect all tuning curves, [nTrainingSchemes x nLayers]
all_units = []

debug=0

#%% define a circular tuning function shape (von mises) to use for fitting
def von_mises_deg180(xx,mu,k,b):
  # make a von mises function over the range in xx
  # assume the input is 0-180 deg space.
  xx_rad2pi = xx/180*2*np.pi
  mu_rad2pi = mu/180*2*np.pi
  yy = np.exp(k*(np.cos(xx_rad2pi-mu_rad2pi)-1)) + b;  
  return yy

#%% main anaylsis function
def analyze_orient_tuning(dataset_all, model,param_str,training_str,ckpt_num):
  
  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset_all)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    
  #%% load full activation patterns (all units, not reduced)
  
  # different versions of the evaluation image set (samples)
  all_units = [] 
  for kk in range(nSamples):
  
    if kk==0 and 'FiltIms' not in dataset_all:
      dataset = dataset_all
    elif 'FiltIms' in dataset_all:
      dataset = '%s_rand%d'%(dataset_all,kk+1)
    else:
      dataset = '%s%d'%(dataset_all,kk)
        
    if kk==0:
      info = load_activations.get_info(model,dataset)
      nLayers = info['nLayers']
      layers2load = info['layer_labels_full']     
      layer_labels = info['layer_labels']

    if debug==1:
      nLayers=2
      
    # find the exact name of the checkpoint file of interest
    ckpt_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dataset))
    nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in ckpt_dirs]
    
    # compare the first two characters    
    good2 = [jj for jj in range(np.size(ckpt_dirs)) if 'orient_tuning' in ckpt_dirs[jj] and ckpt_num[0:2] in nums[jj][0:2]]
    assert(np.size(good2)==1)
    ckpt_dir = ckpt_dirs[good2[0]]
    ckpt_num_actual= ckpt_dir.split('_')[2][5:]

      
    file_path = os.path.join(root,'activations',model,training_str,param_str,dataset,
                               'eval_at_ckpt-%s_orient_tuning'%(ckpt_num_actual))
    for ll in range(nLayers):

      file_name = os.path.join(file_path,'AllUnitsOrientTuning_%s.npy'%(layers2load[ll]))
      print('loading from %s\n'%file_name)
    
      # [nUnits x nSf x nOri]
    
      t = np.load(file_name)
    
      if kk==0:
        nUnits = np.shape(t)[0]
        nSF = np.shape(t)[1]
        nOri = np.shape(t)[2]
        ori_axis = np.arange(0.5, nOri,1) 
        all_units.append(np.zeros([nSamples, nUnits, nSF, nOri]))
#      print('nSF=%s'%nSF)
      all_units[ll][kk,:,:,:] = t
    
  #%% count units with zero response
  # going to get rid of the totally non responsive ones to reduce the size of this big matrix.  
  # also going to get rid of any units with no response variance (exactly same response for all stims)
   
  nTotalUnits = np.zeros([nLayers,1])
  propZeroUnits = np.zeros([nLayers,1])
  propConstUnits = np.zeros([nLayers,1])
  
  resp_units = []
  for ll in range(nLayers):
    nUnits = np.shape(all_units[ll])[1]
    is_zero = np.zeros([nUnits, nSamples,nSF])
    is_constant_nonzero = np.zeros([nUnits, nSamples, nSF])
    
    for kk in range(nSamples):
      print('identifying nonresponsive units in %s, sample %d'%(layer_labels[ll],kk))
      for sf in range(nSF):
        # take out data, [nUnits x nOri]
        
        vals = all_units[ll][kk,:,sf,:]
        
        # find units where signal is zero for all ori
        # add these to a running list of which units were zero for any sample and spatial frequency.
        is_zero[:,kk,sf] = np.all(vals==0,axis=1)
        
        # find units where signal is constant for all images (no variance)
        constval = all_units[ll][0,:,0,0]
        const = np.all(np.equal(vals, np.tile(np.expand_dims(constval, axis=1), [1,nOri])),axis=1)
        # don't count the zero units here so we can see how many of each...
        is_constant_nonzero[:,kk,sf] = np.logical_and(const, ~np.all(vals==0,axis=1))
        
    is_zero_any = np.any(np.any(is_zero,axis=2),axis=1)  
    propZeroUnits[ll] = np.sum(is_zero_any==True)/nUnits
    
    is_constant_any = np.any(np.any(is_constant_nonzero,axis=2),axis=1)
    propConstUnits[ll] = np.sum(is_constant_any==True)/nUnits

    nTotalUnits[ll] = nUnits
    
    # now put the good units only into a new matrix...
    units2use = np.logical_and(~is_zero_any, ~is_constant_any)
    resp_units.append(all_units[ll][:,units2use,:,:])
    
  #%% save the proportion of non-responsive units in each layer
  if debug==0:
    save_name =os.path.join(save_path,'PropZeroUnits_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,propZeroUnits)
    save_name =os.path.join(save_path,'PropConstUnits_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,propConstUnits)
    save_name =os.path.join(save_path,'TotalUnits_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,nTotalUnits)
    
  #%% Save the units at each layer, after removing non-responsive ones
  if debug==0:
    for ll in range(nLayers): 
      save_name =os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
      print('saving to %s\n'%save_name)
      np.save(save_name,resp_units[ll])
    
  #%% Fit von mises functions to each unit, save parameters
  
  nPars = 3;  #center,k,baseline
  fit_pars_all = []
  r2_all = []
  
  for ll in range(nLayers):
    
    nUnits = np.shape(resp_units[ll])[1]
    r2 = np.zeros((nUnits,nSF))
    fit_pars = np.zeros((nUnits,nSF,nPars))
    
    for sf in range(nSF):
      
      # get the data for this spatial frequency and this unit
      
      # dat is nSamples x nUnits x nOri
      dat = resp_units[ll][:,:,sf,:]
      # meandat is nUnits x nOri
      meandat = np.mean(dat,axis=0)
          
      if debug==0:
        units2do = np.arange(0,nUnits)
      else:
        units2do = np.arange(0,10)
        
      for uu in units2do:

        print('fitting %s, sf %d, unit %d \ %d'%(layer_labels[ll],sf,uu, nUnits))
    
        real_y = meandat[uu]
        # first, adjusting the baseline and height of this curve so that we can easily fit it with a von mises going from 0 to 1
        real_y = real_y - np.min(real_y)
        real_y = real_y/np.max(real_y)
        # initialize the mu parameter with the max of the curve
        init_mu = ori_axis[np.argmax(real_y)]
        init_k = 2
        init_b = 0
        params_start = (init_mu,init_k,init_b)
        # do the fitting with scipy curve fitting toolbox
        try:
          params, params_covar = scipy.optimize.curve_fit(von_mises_deg180, ori_axis, real_y, p0=params_start, bounds=((-1,0,-1),(180,100,1)))
        except:
          print('fitting failed for unit %d'%uu)
          r2[uu,sf] = np.nan
          fit_pars[uu,sf,:] = np.nan
          continue
          
        pred_y = von_mises_deg180(ori_axis,params[0],params[1],params[2])
        
        # calculate r2 for this fit.
        ssres = np.sum(np.power((pred_y - real_y),2));
        sstot = np.sum(np.power((real_y - np.mean(real_y)),2));
        r2[uu,sf] = 1-(ssres/sstot)
    
        # record the eventual fit parameters.
        fit_pars[uu,sf,:] = params
       
    r2_all.append(r2)
    fit_pars_all.append(fit_pars)  
    
  #%% save the result of all this analysis
  
  save_name =os.path.join(save_path,'Fit_r2_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,r2_all)
  save_name =os.path.join(save_path,'Fit_pars_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,fit_pars_all)

 #%% main function to decide between subfunctions
  
if __name__=='__main__':
  
  for tr in range(np.size(training_str_list)):
    
    training_str = training_str_list[tr]
  
    if 'pretrained' in training_str or 'stop_early' in training_str:
      ckpt_str = '0'
    else:
      ckpt_str = '400000'
    
    analyze_orient_tuning(dataset_all, model, param_str, training_str, ckpt_str)

   