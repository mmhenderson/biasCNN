#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:29:43 2020

@author: mmhender
"""

import scipy
import os
import numpy as np
import load_activations
from copy import deepcopy
import analyze_orient_tuning_jitter as analyze_orient_tuning


get_CI = 0
fit_centers = 0
get_mean_tf = 0
fit_centers_combine_inits = 1

root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))

nSamples =4
model='vgg16'
#model='vgg16avgpool'

param_str='params1'
inits=[0]
#param_str='params1_init2' 
#training_str='scratch_imagenet_rot_45_cos'
training_str='pretrained'
#training_str = 'scratch_imagenet_rot_0_stop_early';
#training_str = 'scratch_imagenet_rot_0_stop_early_init_ones';
#training_str = 'scratch_imagenet_rot_0_stop_early_weight_init_var_scaling'
#ckpt_str='400000'
ckpt_str='0'
#dataset_str='SpatFreqGratings'
dataset_str='FiltIms14AllSFCos'
#dataset_str='FiltIms11Cos_SF_0.01'

# when identifying well-fit units, what criteria to use?
r2_cutoff = 0.4;

#%% calculate mean response across each layer as a function of orientation (average all the TFs)
def get_mean_TF(root,model,training_str,dataset_str,nSamples,param_str,ckpt_str):
  """ calculate mean response across each layer as a function of orientation (average all the TFs)
  Save the result for all layers to one single file.
  """
  
  dataset = dataset_str
  
  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    
  info = load_activations.get_info(model,dataset)
  nLayers = info['nLayers'] 
  layer_labels = info['layer_labels']
#  layer_labels_full = info['layer_labels_full']

  mean_tf_all = []
  
  for ll in range(nLayers):
    
#    for kk in range(nSamples):
#    kk=0
#      load_path = os.path.join(root,'activations',model,training_str,param_str,dataset + '_rand%d'%(kk+1),'eval_at_ckpt-%s_orient_tuning'%ckpt_str)

               
#      file_name =os.path.join(load_path,'AllUnitsOrientTuning_%s.npy'%(layer_labels_full[ll]))
#    if ll>10:
    file_name =os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_str[0:2]))
    try:
      r = np.load(file_name)
      print('loading from %s\n'%file_name)
    except:
      print('%s not found\n'%file_name)
      continue
    r_all=r
#      if kk==0:
#        r_all = np.zeros([nSamples, np.shape(r)[0],np.shape(r)[2]])
#        r_all = np.zeros([1, np.shape(r)[0],np.shape(r)[1],np.shape(r)[2]])
#      # r_all goes [nSamples x nUnits x nSF x nOrient]
#      r_all[kk,:,:,:] = r
      
    # removing any inf responses
    is_inf = np.any(np.any(np.any(r_all==np.inf, axis=3),axis=2),axis=0)
    print('removing %d units with inf response'%np.sum(is_inf))
    if np.sum(is_inf)>0:
      r_all = r_all[:,~is_inf,:,:]
    # r goes [nSamples x nUnits x nSF x nOrient]
    if np.size(r_all)>0:
      mean_tf = np.mean(r_all, axis=1)
    else:
      mean_tf = []
    mean_tf_all.append(mean_tf)
    
  file_name = os.path.join(save_path, 'All_layers_mean_TF_eval_at_ckpt-%s0000.npy'%(ckpt_str[0:2]))
  np.save(file_name,mean_tf_all)
  
#%% estimate confidence intervals for each fit parameter
def get_conf_int(root, model,training_str,dataset_str,nSamples, param_str, ckpt_str):
  """ Estimate mean and confidence intervals for each fit parameter, binning as a function of fit center.
  """
  
  dataset = dataset_str
  
  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset)
  
  info = load_activations.get_info(model,dataset)
  sf=0
  nLayers = info['nLayers'] 
  layer_labels = info['layer_labels']
  if info['nPhase']==2:
    nOri=360
  else:
    nOri = info['nOri']
  
  fit_pars_all_jitt = []
  r2_each_sample_all_jitt = []
    
  # find the random seed(s) for the jitter that was used
  files=os.listdir(os.path.join(save_path))
  [jitt_file] = [ff for ff in files if '%s_fit_jitter'%layer_labels[0] in ff and 'pars' in ff];  
  rand_seed_str = jitt_file[jitt_file.find('jitter')+7:jitt_file.find('jitter')+13]
  
  #%% load all the single unit fits
  for ll in range(nLayers):
    
    file_name= os.path.join(save_path,'%s_fit_jitter_%s_r2_each_sample_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_str[0:2]))
    print('loading from %s\n'%file_name)
    r2_each_sample_all_jitt.append(np.load(file_name))
    
    file_name= os.path.join(save_path,'%s_fit_jitter_%s_pars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_str[0:2]))
    print('loading from %s\n'%file_name)
    fit_pars_all_jitt.append(np.load(file_name))
    
    #%% get confidence intervals for params at each layer
 
  # how many resampling iterations to do?
  niter_resamp = 10000
  # how many points to resample at a time?
  nvals_resamp = 100
  # going to bin the units by their centers, then get CI of each other parameter within each center bin.
  ori_bin_size=11.25
  # these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
  ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
  nCenterBins = np.size(ori_bins)-1
  
  # [k, ampltiude, baseline, fwhm]
  ppinds = [1,2,3,4]
  nPars=np.size(ppinds)
  
  CI_all = np.zeros((nLayers, nPars, nCenterBins, 3)) # third dim here is [mean, lower CI, upper CI]

  for ll in range(nLayers):
    
    print('estimating resampled confidence intervals for %s'%layer_labels[ll])
    
    # identify well-fit units
    rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all_jitt[ll][:,sf,:],axis=1)))      
    rvals[np.isnan(rvals)] = -1000
    inds2use = np.where(rvals>r2_cutoff)[0]
    # get values of center for these units 
    cvals = deepcopy(np.squeeze(fit_pars_all_jitt[ll][inds2use,sf,0]))
    
    for pp in range(nPars):
      # random seed for generating random resamples
      np.random.seed(int(rand_seed_str)+ll*nPars+pp)  
  
      # get values of other parameter whose relationship with the center is of interest
      parvals = deepcopy(np.squeeze(fit_pars_all_jitt[ll][inds2use,sf,ppinds[pp]]))
      
      # loop over bins and get CI for each
      for cc in range(nCenterBins):
        inds = np.logical_and(cvals>ori_bins[cc], cvals<=ori_bins[cc+1])          
        if np.sum(inds)>0:
#           get param values for centers at orientation of interest
          all_vals = parvals[inds]
#          if pp==0:
#            print('center=%d, %d units'%(ori_bins[cc]+ori_bin_size/2, np.size(all_vals)))
          # resample values from this set
          resamp_means = np.zeros((niter_resamp,1))
          for ii in range(niter_resamp):
            inds_samp = np.random.randint(0,np.size(all_vals), nvals_resamp)
            samp_vals = all_vals[inds_samp]
            resamp_means[ii] = np.mean(samp_vals)
          # take percentiles over the resampling iterations  
          resamp_mean = np.percentile(resamp_means,50)
          CI_lower = np.percentile(resamp_means,2.5)
          CI_upper = np.percentile(resamp_means,97.5)
          
        else:
          resamp_mean=np.nan
          CI_lower=np.nan
          CI_upper=np.nan

        CI_all[ll,pp,cc,0] = resamp_mean
        CI_all[ll,pp,cc,1] = CI_lower
        CI_all[ll,pp,cc,2] = CI_upper
      
  #%% save confidence interval calculations   
  file_name= os.path.join(save_path,'All_layers_fit_jitter_%s_CI_all_params_eval_at_ckpt_%s0000.npy'%(rand_seed_str,ckpt_str[0:2]))
  print('loading from %s\n'%file_name)
  np.save(file_name, CI_all)

#%% define a bimodal tuning shape to use for fitting
def double_von_mises_deg(xx,mu1,mu2,k1,k2,a1,a2,b):
  """Make a double von mises function over the range in xx
  assume the input is 0-180 or 0-360 deg space.
  """
  axis_size_deg = np.max(xx)+1
  if k1<10**(-15) or k2<10**(-15):
    print('WARNING: k is too small, might get precision errors')
    
  xx_rad2pi = np.float128(xx/axis_size_deg*2*np.pi)
  mu1_rad2pi = mu1/axis_size_deg*2*np.pi
  mu2_rad2pi = mu2/axis_size_deg*2*np.pi
  
  # get y values for each mu and k pair
  yy1 = np.exp(k1*(np.cos(xx_rad2pi-mu1_rad2pi)-1))
  yy2 = np.exp(k2*(np.cos(xx_rad2pi-mu2_rad2pi)-1))
  
  # make each curve span 0-1
  yy1 = yy1-min(yy1)
  yy1 = yy1/max(yy1)
  
  yy2 = yy2-min(yy2)
  yy2 = yy2/max(yy2)
      
  # finally, combine curves - apply both amplitudes and the baseline.
  yy = a1*yy1 + a2*yy2 +b
  
  return yy

#%% fit center distribution with double von-mises function  
def fit_center_dist(root, model,training_str,dataset_str,nSamples, param_str, ckpt_str):
  """ Fit a bimodal curve to the distribution of single unit tuning curve center at each layer.
  Save the result in a single file.
  """
  dataset = dataset_str
  
  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset)
  
  info = load_activations.get_info(model,dataset)
  sf=0
  nLayers = info['nLayers'] 
  layer_labels = info['layer_labels']
  if info['nPhase']==2:
    nOri=360
  else:
    nOri = info['nOri']
  
  fit_pars_all_jitt = []  
  r2_each_sample_all_jitt = []
    
  # find the random seed(s) for the jitter that was used
  files=os.listdir(os.path.join(save_path))
  [jitt_file] = [ff for ff in files if '%s_fit_jitter'%layer_labels[0] in ff and 'pars' in ff];  
  rand_seed_str = jitt_file[jitt_file.find('jitter')+7:jitt_file.find('jitter')+13]
  
  #%% load all the single unit fits
  for ll in range(nLayers):
    
    file_name= os.path.join(save_path,'%s_fit_jitter_%s_r2_each_sample_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_str[0:2]))
    print('loading from %s\n'%file_name)
    r2_each_sample_all_jitt.append(np.load(file_name))
    
    file_name= os.path.join(save_path,'%s_fit_jitter_%s_pars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_str[0:2]))
    print('loading from %s\n'%file_name)
    fit_pars_all_jitt.append(np.load(file_name))
   
  #%% fit a bimodal curve to the distribution of fit centers in each layer.
  
  mu_pairs = [[0, 90], [22.5, 112.5], [45, 135], [67.5, 157.5]]
  nParams=7 # [mu1, mu2, k1, k2, amp1, amp2, baseline]  
  center_dist_pars_all = np.zeros([nLayers, np.shape(mu_pairs)[0], nParams+1]) # final column is r2
  
  # define bins to use the for the distribution
  ori_bin_size=1
  # these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
  ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
  bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2
  
  # loop over locations for the two centers to be at
  for pp in range(np.shape(mu_pairs)[0]):
    
   
    # define where the centers are for this bimodal curve shape
    mu_pair = mu_pairs[pp]
    print('fitting curves with centers at %d and %d'%(mu_pair[0],mu_pair[1]))
    # constrain the two mu parameters, create a new function which only takes other parameters as input.
    fixed_mu_function = lambda xx, k1, k2, a1, a2, b : double_von_mises_deg(xx,mu_pair[0],mu_pair[1],k1,k2,a1,a2,b)
    
    for ll in range(nLayers):
      
#      print('fitting double von-mises function for %s'%layer_labels[ll])
      
      # identify well-fit units
      rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all_jitt[ll][:,sf,:],axis=1)))      
      rvals[np.isnan(rvals)] = -1000
      inds2use = np.where(rvals>r2_cutoff)[0]
      # get values of center for these units 
      cvals = deepcopy(np.squeeze(fit_pars_all_jitt[ll][inds2use,sf,0]))
      
      # get the actual curve that describes the distribution of centers
      h = np.histogram(cvals, ori_bins) # h[0] is the number per bin, h[1] is the bin edges
      # divide by total to get a proportion.
      real_y = h[0]/np.sum(h[0])
      
      # set up some initial parameters:
      # estimate baseline from the min of the curve
      init_b = np.min(real_y)
      # estimate amplitude from the peak height at each mu
      ind1 = np.argmin(np.abs(bin_centers-mu_pair[0]))
      ind2 = np.argmin(np.abs(bin_centers-mu_pair[1]))
      init_a1 = np.max([real_y[ind1]- init_b, 10**(-15)]) # make sure amp isn't too small, fitting will fail if zero
      init_a2 = np.max([real_y[ind2]- init_b, 10**(-15)]) 
      init_k = 10
      
      params_start = (init_k,init_k,init_a1,init_a2,init_b)
     
      # do the fitting with scipy curve fitting toolbox
      try:
        # constrain k and amplitude to positive values that aren't too small, let b vary freely.
        params, params_covar = scipy.optimize.curve_fit(fixed_mu_function, bin_centers, real_y, p0=params_start, bounds=((1,1 ,10**(-15),10**(-15),-np.inf),(np.inf,np.inf,np.inf,np.inf,np.inf)))
#        params, params_covar = scipy.optimize.curve_fit(fixed_mu_function, bin_centers, real_y, p0=params_start, bounds=((10**(-15),10**(-15),10**(-15),10**(-15),-np.inf),(np.inf,np.inf,np.inf,np.inf,np.inf)))
      except:
        print('fitting failed for layer %d'%ll) 
        center_dist_pars_all[ll,pp,0:2] = np.nan
        center_dist_pars_all[ll,pp,2:nParams] = np.nan
        center_dist_pars_all[ll,pp,nParams] = np.nan            
        continue
      
      # calculate the best fit curve based on these pars
      pred_y = fixed_mu_function(bin_centers, params[0],params[1],params[2],params[3],params[4])
      
      # make sure this fitting worked properly - there should be local maxima at both of the expected locations.
      missing_peaks=0
      for mm in range(np.size(mu_pair)):
        if mu_pair[mm]!=0 and np.mod(mu_pair[mm],1)==0:          
          ind_center = np.argmin(np.abs(bin_centers-mu_pair[mm]))       
          ind_left = ind_center-1
          ind_right = ind_center+1          
          max_here = pred_y[ind_center]>pred_y[ind_left] and pred_y[ind_center]>pred_y[ind_right]
        elif np.mod(mu_pair[mm],1)>0:
          ind_center1 = np.argmin(np.abs(bin_centers-np.floor(mu_pair[mm])))
          ind_center2 = np.argmin(np.abs(bin_centers-np.ceil(mu_pair[mm])))
          ind_left = ind_center1-1
          ind_right = ind_center2+1          
          max_here = pred_y[ind_center1]>pred_y[ind_left] and pred_y[ind_center2]>pred_y[ind_right]
        else:
          ind_center1 = np.argmin(np.abs(bin_centers-mu_pair[mm]))
          ind_center2 = np.argmin(np.abs(bin_centers-(mu_pair[mm]+180)))
          ind_left = ind_center2-1
          ind_right = ind_center1+1          
          max_here = pred_y[ind_center2]>pred_y[ind_left] and pred_y[ind_center1]>pred_y[ind_right]
        if not max_here:
          missing_peaks = missing_peaks +1
          
      if missing_peaks>0:
        # don't use this fit
        print('bad fit for layer %d'%(ll))
        center_dist_pars_all[ll,pp,0:2] = np.nan
        center_dist_pars_all[ll,pp,2:nParams] = np.nan
        center_dist_pars_all[ll,pp,nParams] = np.nan
      else:
          
        # get r2 for this fit
        r2 = analyze_orient_tuning.get_r2(real_y, pred_y)
        # save the final parameters
        center_dist_pars_all[ll,pp,0:2] = mu_pair
        center_dist_pars_all[ll,pp,2:nParams] = params
        center_dist_pars_all[ll,pp,nParams] = r2
      
  #%% save confidence interval calculations   
  file_name= os.path.join(save_path,'All_layers_fit_jitter_%s_center_dist_pars_eval_at_ckpt_%s0000.npy'%(rand_seed_str,ckpt_str[0:2]))
  print('loading from %s\n'%file_name)
  np.save(file_name, center_dist_pars_all)
  
#%% fit center distribution with double von-mises function  
def fit_center_dist_combine_inits(root, model,training_str,inits,dataset_str,nSamples, param_str, ckpt_str):
  """ Fit a bimodal curve to the distribution of single unit tuning curve center at each layer.
  Save the result in a single file.
  """
  dataset = dataset_str
    
  info = load_activations.get_info(model,dataset)
  sf=0
  nLayers = info['nLayers'] 
  layer_labels = info['layer_labels']
  if info['nPhase']==2:
    nOri=360
  else:
    nOri = info['nOri']

  # where the final analysis across all initializations gets saved  
  final_save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset)    
  
  fit_pars_all_jitt = []  
  r2_each_sample_all_jitt = []
  
  for ii in inits:  
    
    if ii==0:
      param_str_full=param_str
    else:
      param_str_full = param_str +'_init%d'%ii
      
    save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str_full,dataset)
    
    # find the random seed(s) for the jitter that was used
    files=os.listdir(os.path.join(save_path))
    [jitt_file] = [ff for ff in files if '%s_fit_jitter'%layer_labels[0] in ff and 'pars' in ff];  
    rand_seed_str = jitt_file[jitt_file.find('jitter')+7:jitt_file.find('jitter')+13]
    
    #%% load all the single unit fits
    r=[]
    f=[]
    for ll in range(nLayers):
      
      file_name= os.path.join(save_path,'%s_fit_jitter_%s_r2_each_sample_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_str[0:2]))
      print('loading from %s\n'%file_name)
      r.append(np.load(file_name))
      
      file_name= os.path.join(save_path,'%s_fit_jitter_%s_pars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_str[0:2]))
      print('loading from %s\n'%file_name)
      f.append(np.load(file_name))
      
    r2_each_sample_all_jitt.append(r) 
    fit_pars_all_jitt.append(f)
    
  #%% fit a bimodal curve to the distribution of fit centers in each layer.
  
  mu_pairs = [[0, 90], [22.5, 112.5], [45, 135], [67.5, 157.5]]
  nParams=7 # [mu1, mu2, k1, k2, amp1, amp2, baseline]  
  center_dist_pars_all = np.zeros([nLayers, np.shape(mu_pairs)[0], nParams+1]) # final column is r2
  
  # define bins to use the for the distribution
  ori_bin_size=1
  # these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
  ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
  bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2
  
  # loop over locations for the two centers to be at
  for pp in range(np.shape(mu_pairs)[0]):
    
   
    # define where the centers are for this bimodal curve shape
    mu_pair = mu_pairs[pp]
    print('fitting curves with centers at %d and %d'%(mu_pair[0],mu_pair[1]))
    # constrain the two mu parameters, create a new function which only takes other parameters as input.
    fixed_mu_function = lambda xx, k1, k2, a1, a2, b : double_von_mises_deg(xx,mu_pair[0],mu_pair[1],k1,k2,a1,a2,b)
    
    for ll in range(nLayers):
      
      # concatenate units from all initializations      
      for ii in range(np.size(inits)):
        if ii==0:
          rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all_jitt[ii][ll][:,sf,:],axis=1)))
          cvals = deepcopy(np.squeeze(fit_pars_all_jitt[ii][ll][:,sf,0]))
        else:
          rvals = np.concatenate((rvals,deepcopy(np.squeeze(np.mean(r2_each_sample_all_jitt[ii][ll][:,sf,:],axis=1)))),axis=0)
          cvals = np.concatenate((cvals,deepcopy(np.squeeze(fit_pars_all_jitt[ii][ll][:,sf,0]))),axis=0)
          
      # identify well-fit units          
      rvals[np.isnan(rvals)] = -1000
      inds2use = np.where(rvals>r2_cutoff)[0]
      # get values of center for these units 
      cvals = cvals[inds2use]
      
      # get the actual curve that describes the distribution of centers
      h = np.histogram(cvals, ori_bins) # h[0] is the number per bin, h[1] is the bin edges
      # divide by total to get a proportion.
      real_y = h[0]/np.sum(h[0])
      
      # set up some initial parameters:
      # estimate baseline from the min of the curve
      init_b = np.min(real_y)
      # estimate amplitude from the peak height at each mu
      ind1 = np.argmin(np.abs(bin_centers-mu_pair[0]))
      ind2 = np.argmin(np.abs(bin_centers-mu_pair[1]))
      init_a1 = np.max([real_y[ind1]- init_b, 10**(-15)]) # make sure amp isn't too small, fitting will fail if zero
      init_a2 = np.max([real_y[ind2]- init_b, 10**(-15)]) 
      init_k = 10
      
      params_start = (init_k,init_k,init_a1,init_a2,init_b)
     
      # do the fitting with scipy curve fitting toolbox
      try:
        # constrain k and amplitude to positive values that aren't too small, let b vary freely.
        params, params_covar = scipy.optimize.curve_fit(fixed_mu_function, bin_centers, real_y, p0=params_start, bounds=((1,1 ,10**(-15),10**(-15),-np.inf),(np.inf,np.inf,np.inf,np.inf,np.inf)))
#        params, params_covar = scipy.optimize.curve_fit(fixed_mu_function, bin_centers, real_y, p0=params_start, bounds=((10**(-15),10**(-15),10**(-15),10**(-15),-np.inf),(np.inf,np.inf,np.inf,np.inf,np.inf)))
      except:
        print('fitting failed for layer %d'%ll) 
        center_dist_pars_all[ll,pp,0:2] = np.nan
        center_dist_pars_all[ll,pp,2:nParams] = np.nan
        center_dist_pars_all[ll,pp,nParams] = np.nan            
        continue
      
      # calculate the best fit curve based on these pars
      pred_y = fixed_mu_function(bin_centers, params[0],params[1],params[2],params[3],params[4])
      
      # make sure this fitting worked properly - there should be local maxima at both of the expected locations.
      missing_peaks=0
      for mm in range(np.size(mu_pair)):
        if mu_pair[mm]!=0 and np.mod(mu_pair[mm],1)==0:          
          ind_center = np.argmin(np.abs(bin_centers-mu_pair[mm]))       
          ind_left = ind_center-1
          ind_right = ind_center+1          
          max_here = pred_y[ind_center]>pred_y[ind_left] and pred_y[ind_center]>pred_y[ind_right]
        elif np.mod(mu_pair[mm],1)>0:
          ind_center1 = np.argmin(np.abs(bin_centers-np.floor(mu_pair[mm])))
          ind_center2 = np.argmin(np.abs(bin_centers-np.ceil(mu_pair[mm])))
          ind_left = ind_center1-1
          ind_right = ind_center2+1          
          max_here = pred_y[ind_center1]>pred_y[ind_left] and pred_y[ind_center2]>pred_y[ind_right]
        else:
          ind_center1 = np.argmin(np.abs(bin_centers-mu_pair[mm]))
          ind_center2 = np.argmin(np.abs(bin_centers-(mu_pair[mm]+180)))
          ind_left = ind_center2-1
          ind_right = ind_center1+1          
          max_here = pred_y[ind_center2]>pred_y[ind_left] and pred_y[ind_center1]>pred_y[ind_right]
        if not max_here:
          missing_peaks = missing_peaks +1
          
      if missing_peaks>0:
        # don't use this fit
        print('bad fit for layer %d'%(ll))
        center_dist_pars_all[ll,pp,0:2] = np.nan
        center_dist_pars_all[ll,pp,2:nParams] = np.nan
        center_dist_pars_all[ll,pp,nParams] = np.nan
      else:
          
        # get r2 for this fit
        r2 = analyze_orient_tuning.get_r2(real_y, pred_y)
        # save the final parameters
        center_dist_pars_all[ll,pp,0:2] = mu_pair
        center_dist_pars_all[ll,pp,2:nParams] = params
        center_dist_pars_all[ll,pp,nParams] = r2
      
  #%% save confidence interval calculations   
  file_name= os.path.join(final_save_path,'All_layers_fit_jitter_center_dist_pars_eval_at_ckpt_%s0000_all_inits.npy'%(ckpt_str[0:2]))
  print('loading from %s\n'%file_name)
  np.save(file_name, center_dist_pars_all)
  
#%% main function 
if __name__ == '__main__':
  
   if get_CI==1:
     get_conf_int(root, model,training_str,dataset_str,nSamples, param_str, ckpt_str)
     
   if fit_centers==1:
     fit_center_dist(root, model,training_str,dataset_str,nSamples, param_str, ckpt_str)
    
   if fit_centers_combine_inits==1:
     fit_center_dist_combine_inits(root, model,training_str,inits,dataset_str,nSamples, param_str, ckpt_str)
     
   if get_mean_tf==1:
     get_mean_TF(root, model,training_str,dataset_str,nSamples, param_str, ckpt_str)