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
import sys

debug=0

#%% define a circular tuning function shape (von mises) to use for fitting
def von_mises_deg(xx,mu,k,a,b):
  # make a von mises function over the range in xx
  # assume the input is 0-180 or 0-360 deg space.
  axis_size_deg = np.max(xx)+1
  if k<10**(-15):
    print('WARNING: k is too small, might get precision errors')
    
  xx_rad2pi = np.float128(xx/axis_size_deg*2*np.pi)
  mu_rad2pi = mu/axis_size_deg*2*np.pi
  yy = np.exp(k*(np.cos(xx_rad2pi-mu_rad2pi)-1))
  
  # first make the y values span from 0-1
  yy = yy-min(yy)
  yy = yy/max(yy)
  
  # then apply the entered amplitude and baseline.
  yy = a*yy+b
  
  return yy

#%% get full-width at half-max of a fitted tuning function
def get_fwhm(yy,xx):
  # assume xx is in degrees, 0-180 space or 0-360 space
  # assume the function has positive tuning here 
  axis_size_deg = np.max(xx)+1
  axis_half_deg = np.int64(axis_size_deg/2)
  # what is the half-max height of the tuning function
  halfmaxheight = np.min(yy)+(np.max(yy)-np.min(yy))/2
  
  # where is the peak of the tuning function?
  peak = xx[np.argmax(yy)]
  
  # want one value left of the peak, one val right of the peak.
  diff = xx-peak
  diff[diff>axis_half_deg] = diff[diff>axis_half_deg]-axis_size_deg
  diff[diff<-axis_half_deg] = diff[diff<-axis_half_deg]+axis_size_deg
  # now find the closest values to the desired height
  dist = np.abs(yy-halfmaxheight)
  val1 = np.min(dist[diff<0])
  val2 = np.min(dist[diff>0])
  ind1 = np.where(np.logical_and(dist==val1, diff<0))[0][0]
  ind2 = np.where(np.logical_and(dist==val2, diff>0))[0][0]
  
  # now subtract these values, taking into account circularity
  if ind1>ind2:
    fwhm = axis_size_deg - np.abs(xx[ind2]-xx[ind1])
  else:
    fwhm = np.abs(xx[ind2]-xx[ind1])
  
  return fwhm

#%% get r2 for each fitted curve.
def get_r2(actual,predicted):
  
    # calculate r2 for this fit.
    ssres = np.sum(np.power((predicted - actual),2));
    sstot = np.sum(np.power((actual - np.mean(actual)),2));
    r2 = 1-(ssres/sstot)
    
    return r2
  
#%% main analysis function
def analyze_orient_tuning(root, model, training_str, dataset_all, nSamples, param_str, ckpt_str, rand_seed):
  
  save_path = os.path.join(root,'biasCNN','code','unit_tuning',model,training_str,param_str,dataset_all)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    
  # get info about the model/dataset we're using here
  info = load_activations.get_info(model,dataset_all)
  nLayers = info['nLayers']
  layers2load = info['layer_labels_full']     
  layer_labels = info['layer_labels']
  
  if debug==1:
    nLayers=2
    
  # going to count units with zero response
  # going to get rid of the totally non responsive ones to reduce the size of this big matrix.  
  # also going to get rid of any units with no response variance (exactly same response for all stims)   
  nTotalUnits = np.zeros([nLayers,1])
  propZeroUnits = np.zeros([nLayers,1]) # how many units had zero resp for all stims?
  propConstUnits = np.zeros([nLayers,1])  # how many units had some constant, nonzero resp for all stims?
  
  #%% loop over layers and do processing for each
  for ll in range(nLayers):
    
    # check if this has been done yet, skip if so
    files_done = os.listdir(save_path)
    fit_files_this_layer = [ff for ff in files_done if layer_labels[ll] in ff and 'jitter_%d'%rand_seed in ff]
    n_done = np.size(fit_files_this_layer)
    if n_done>=3:
      continue
    
    #%% loop over different versions of the evaluation image set (samples)
    # load the full activation patterns
    for kk in range(nSamples):
    
      if kk==0 and 'FiltIms' not in dataset_all:
        dataset = dataset_all
      elif 'FiltIms' in dataset_all:
        dataset = '%s_rand%d'%(dataset_all,kk+1)
      else:
        if kk==0:
          dataset=dataset_all
        else:
          dataset = '%s%d'%(dataset_all,kk)
          
      file_path = os.path.join(root,'biasCNN','activations',model,training_str,param_str,dataset,
                               'eval_at_ckpt-%s_orient_tuning'%(ckpt_str))
      
      file_name = os.path.join(file_path,'AllUnitsOrientTuning_%s.npy'%(layers2load[ll]))
      print('loading from %s\n'%file_name)
    
      # [nUnits x nSf x nOri]    
      t = np.load(file_name)
            
      if kk==0:
        nUnits = np.shape(t)[0]
        nSF = np.shape(t)[1]
        nOri = np.shape(t)[2]
        ori_axis = np.arange(0, nOri,1) 
        all_units = np.zeros([nSamples, nUnits, nSF, nOri])
      
      print('analyzing tuning curves for %d spatial frequencies, orientations 0-%d deg\n'%(nSF,nOri))
      
      # [nSamples x nUnits x nSF x nOri]
      all_units[kk,:,:,:] = t
    
      #%% For each unit - label it according to its spatial position and channel.
      # full W matrix is in NHWC format: Number of images x Height (top to bottom) x Width (left ro right) x Channels.
      # First dim of t matrix is currently [H x W x C]  (reshaped from W)
      H = info['activ_dims'][ll]
      W = info['activ_dims'][ll]
      C = int(np.shape(t)[0]/H/W)        
      clabs = np.tile(np.expand_dims(np.arange(0,C),axis=1),[H*W,1])
      wlabs = np.expand_dims(np.repeat(np.tile(np.expand_dims(np.arange(0,W),axis=1),[H,1]), C),axis=1)
      hlabs = np.expand_dims(np.repeat(np.arange(0,H),W*C),axis=1)
      # the coords matrix goes [nUnits x 3] where columns are [H,W,C]
      coords = np.concatenate((hlabs,wlabs,clabs),axis=1)      
      assert np.array_equal(coords, np.unique(coords, axis=0))
  
    #%% now identify the non-responsive units in this layer.
    nUnits = np.shape(all_units)[1]
    is_zero = np.zeros([nUnits,nSamples,nSF])
    is_constant_nonzero = np.zeros([nUnits, nSamples, nSF])
    
    for kk in range(nSamples):
      print('identifying nonresponsive units in %s, sample %d'%(layer_labels[ll],kk))
      for sf in range(nSF):
        # take out data, [nUnits x nOri]        
        vals = all_units[kk,:,sf,:]
        
        # find units where signal is zero for all ori
        # add these to a running list of which units were zero for any sample and spatial frequency.
        is_zero[:,kk,sf] = np.all(vals==0,axis=1)
        
        # find units where signal is constant for all images (no variance)
        constval = all_units[0,:,0,0]
        const = np.all(np.equal(vals, np.tile(np.expand_dims(constval, axis=1), [1,nOri])),axis=1)
        # don't count the zero units here so we can see how many of each...
        is_constant_nonzero[:,kk,sf] = np.logical_and(const, ~np.all(vals==0,axis=1))
        
    is_zero_any = np.any(np.any(is_zero,axis=2),axis=1)  
    propZeroUnits[ll] = np.sum(is_zero_any==True)/nUnits
    
    is_constant_any = np.any(np.any(is_constant_nonzero,axis=2),axis=1)
    propConstUnits[ll] = np.sum(is_constant_any==True)/nUnits

    nTotalUnits[ll] = nUnits
    
    # make a new matrix with only the good units in it.
    units2use = np.logical_and(~is_zero_any, ~is_constant_any)
    resp_units = all_units[:,units2use,:,:]
    
    # record the spatial position and channel corresponding to each unit.
    coords_good = coords[units2use,:]
    
    #%% Save the responsive units   
    save_name =os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_str[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,resp_units)
  
    save_name =os.path.join(save_path,'%s_coordsHWC_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_str[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,coords_good)
    
    #%% Fit von mises functions to each unit, save parameters
    
    # random seed for generating random x-axis jitter
    np.random.seed(rand_seed+ll)  
    jitter_by = np.random.randint(0,nOri,size=[nSF,nUnits])
    
    nPars = 4;  #center,k,baseline
    nUnits = np.shape(resp_units)[1]  # note that nUnits is smaller now than it was above, counting only responsive ones.
    # record r2 and parameters of the final fit
    r2 = np.zeros((nUnits,nSF))
    fit_pars = np.zeros((nUnits,nSF,nPars+1))
    # as a check for tuning consistency - calculate r2 for the fit to each individual sample.
    r2_each_sample = np.zeros((nUnits, nSF, nSamples))
    
    for sf in range(nSF):
      
      # get the data for this spatial frequency and this unit
      
      # dat is nSamples x nUnits x nOri
      dat = resp_units[:,:,sf,:]
      # meandat is nUnits x nOri
      meandat = np.mean(dat,axis=0)
          
      if debug==0:
        units2do = np.arange(0,nUnits)
      else:
        units2do = np.arange(0,100)
        
      
      # loop over units and fit each one
      for uu in units2do:

        print('fitting %s, sf %d, unit %d \ %d, jitter by %d deg'%(layer_labels[ll],sf,uu, nUnits,jitter_by[sf,uu]))
    
        real_y = meandat[uu,:]
        # circularly shift the function over before fitting. Make sure that we don't end up with a bunch of units tuned to edges of orient space by accident.
        real_y = np.roll(real_y, jitter_by[sf,uu])
        # estimate amplitude and baseline from the max and min 
        init_b = np.min(real_y)
        init_a = np.max(real_y)-init_b
        # initialize the mu parameter with the max of the curve
        init_mu = ori_axis[np.argmax(real_y)]
        init_k = 1
        params_start = (init_mu,init_k,init_a,init_b)
       
        # do the fitting with scipy curve fitting toolbox
        try:
          # constrain the center to 0-179, constrain k to positive values that aren't too small, let a and b vary freely.
          params, params_covar = scipy.optimize.curve_fit(von_mises_deg, ori_axis, real_y, p0=params_start, bounds=((-0.0001,10**(-15),-np.inf,-np.inf),(nOri+0.0001,np.inf,np.inf,np.inf)))
        except:
          print('fitting failed for unit %d'%uu)
          r2[uu,sf] = np.nan
          fit_pars[uu,sf,:] = np.nan
          continue
        
       
        pred_y = von_mises_deg(ori_axis,params[0],params[1],params[2],params[3])
              
        # get r2 for this fit
        r2[uu,sf] = get_r2(real_y, pred_y)
        
        # get r2 for how well this fit captures individual samples
        for ss in range(nSamples):
          real_y_this_sample = dat[ss,uu,:]
          # circularly shift the function over to match the mean tuning function
          real_y_this_sample = np.roll(real_y_this_sample, jitter_by[sf,uu])
          r2_each_sample[uu,sf,ss] = get_r2(real_y_this_sample, pred_y)
          
        # record the final fit parameters.
        # finally, re-adjust the center to where it would have been before shifting over.
        params[0] = np.mod(params[0]-jitter_by[sf,uu],nOri)
        fit_pars[uu,sf,0:nPars] = params
        
        # also calculate fwhm for the von mises function - more interpretable than k.        
        if params[2]<0:
          # if the von mises had a negative amplitude, we actually want the width of the negative peak.
          fwhm = get_fwhm((-1)*pred_y,ori_axis)
        else:
          fwhm = get_fwhm(pred_y,ori_axis)
        fit_pars[uu,sf,nPars] = fwhm

    #%% save the results of the fitting, all units this layer
    save_name =os.path.join(save_path,'%s_fit_jitter_%d_r2_eval_at_ckpt_%s0000.npy'%(layer_labels[ll], rand_seed, ckpt_str[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,r2)
    save_name =os.path.join(save_path,'%s_fit_jitter_%d_r2_each_sample_eval_at_ckpt_%s0000.npy'%(layer_labels[ll], rand_seed, ckpt_str[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,r2_each_sample)
    save_name =os.path.join(save_path,'%s_fit_jitter_%d_pars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll], rand_seed, ckpt_str[0:2]))
    print('saving to %s\n'%save_name)
    np.save(save_name,fit_pars)


  #%% save the proportion of non-responsive units in each layer
 
  save_name =os.path.join(save_path,'PropZeroUnits_eval_at_ckpt_%s0000.npy'%(ckpt_str[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,propZeroUnits)
  save_name =os.path.join(save_path,'PropConstUnits_eval_at_ckpt_%s0000.npy'%(ckpt_str[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,propConstUnits)
  save_name =os.path.join(save_path,'TotalUnits_eval_at_ckpt_%s0000.npy'%(ckpt_str[0:2]))
  print('saving to %s\n'%save_name)
  np.save(save_name,nTotalUnits)
  
 #%% main function to decide between subfunctions
  
if __name__ == '__main__':
  
  root = sys.argv[1] #'The path where this project is located.'
  model = sys.argv[2] #  the name of the model
  training_str = sys.argv[3]
  dataset_all = sys.argv[4] # the name of the dataset
  nSamples = int(sys.argv[5]) # how many versions of the dataset are there?
  param_str = sys.argv[6] # which training params? always params1
  ckpt_str = sys.argv[7] # which checkpoint am i evaluating the trained model at?
  rand_seed = int(sys.argv[8]) # random seed to use for random jittering of circular axis when fitting tuning functions
  
  print('\nroot is %s'%root)
  print('model name is %s'%model)
  print('training str is %s'%training_str)
  print('dataset name is %s'%dataset_all)
  print('nSamples is %d'%nSamples)
  print('params are %s'%param_str)
  print('ckpt_num is %s'%ckpt_str)
  print('rand_seed is %d'%rand_seed)
#  ckpt_str=0

  analyze_orient_tuning(root, model, training_str, dataset_all, nSamples, param_str, ckpt_str, rand_seed)

   