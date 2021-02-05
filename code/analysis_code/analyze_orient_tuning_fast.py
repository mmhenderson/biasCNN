#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate some basic parameters (max etc) of orient tuning functions, without fitting.
Before running this, need to run get_orient_tuning.py
Plot the output of this code with plot_unit_tuning.py

"""

import os
import numpy as np
import load_activations
#import scipy.optimize
import sys

debug=0

#%% main analysis function
def analyze_orient_tuning(root, model, training_str, dataset_all, nSamples, param_str, ckpt_str):
  
  if ckpt_str=='0':
    ckpt_str_print='00000'
  else:
    ckpt_str_print = '%s'%(np.round(int(ckpt_str),-4))

  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset_all)
  assert(os.path.exists(save_path))

  # get info about the model/dataset we're using here
  info = load_activations.get_info(model,dataset_all)
  nLayers = info['nLayers']
  layer_labels = info['layer_labels']
  
  if debug==1:
    nLayers=2
  
  #%% loop over layers and do processing for each
  for ll in range(nLayers):
    
    # check if this has been done yet, skip if so
    files_done = os.listdir(save_path)
    fit_files_this_layer = [ff for ff in files_done if layer_labels[ll] in ff and 'fast' in ff]
    n_done = np.size(fit_files_this_layer)
    if n_done>=1:
      continue

   
    # already have raw tuning curves for all the responsive units
    # load them here and then will proceed to fit
    save_name =os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s.npy'%(layer_labels[ll],ckpt_str_print))
    print('loading from %s\n'%save_name)
    resp_units = np.load(save_name)

    nOri = 180
    if np.shape(resp_units)[3]>nOri:
      resp_units = resp_units[:,:,:,0:nOri]
    
    # some parameters that can be estimated quickly - no fitting needed
    maxori = np.argmax(resp_units,axis=3)
    minori = np.argmin(resp_units,axis=3)
    sqslopevals = np.diff(resp_units,axis=3)**2  # finding most slopey region of each unit's orientaton tuning function
    maxsqslopeori = np.argmax(sqslopevals, axis=3)
    maxsqslopeval = np.max(sqslopevals, axis=3)
    meanresp = np.mean(resp_units,axis=3)
    maxresp = np.max(resp_units,axis=3)
    minresp = np.min(resp_units,axis=3)
    
    
    fastpars = dict()
    fastpars['maxori'] = maxori
    fastpars['minori'] = minori
    fastpars['maxsqslopeori'] = maxsqslopeori
    fastpars['maxsqslopeval'] = maxsqslopeval
    fastpars['meanresp'] = meanresp
    fastpars['maxresp'] = maxresp
    fastpars['minresp'] = minresp
    
    resp_units=None
   
    #%% save the results of the fitting, all units this layer
    save_name =os.path.join(save_path,'%s_fastpars_eval_at_ckpt_%s.npy'%(layer_labels[ll],ckpt_str_print))
    print('saving to %s\n'%save_name)
    np.save(save_name,fastpars)
  
    fastpars = None
  
 #%% main function to decide between subfunctions
  
if __name__ == '__main__':
  
  root = sys.argv[1] #'The path where this project is located.'
  model = sys.argv[2] #  the name of the model
  training_str = sys.argv[3]
  dataset_all = sys.argv[4] # the name of the dataset
  nSamples = int(sys.argv[5]) # how many versions of the dataset are there?
  param_str = sys.argv[6] # which training params? always params1
  ckpt_str = sys.argv[7] # which checkpoint am i evaluating the trained model at?
  
  print('\nroot is %s'%root)
  print('model name is %s'%model)
  print('training str is %s'%training_str)
  print('dataset name is %s'%dataset_all)
  print('nSamples is %d'%nSamples)
  print('params are %s'%param_str)
  print('ckpt_num is %s'%ckpt_str)
  
#  ckpt_str=0

  analyze_orient_tuning(root, model, training_str, dataset_all, nSamples, param_str, ckpt_str)

   