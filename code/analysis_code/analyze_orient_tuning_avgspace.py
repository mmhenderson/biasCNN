#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform quick analyses (no curve fitting) of spatially-averaged TFs.
Before running this, need to run get_orient_tuning_avgspace.py

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
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    
  # get info about the model/dataset we're using here
  info = load_activations.get_info(model,dataset_all)
  nLayers = info['nLayers']
  layers2load = info['layer_labels_full']     
  layer_labels = info['layer_labels']
  
  if debug==1:
    nLayers=2

  #%% loop over layers and do processing for each
  for ll in range(nLayers):
    
    # check if this has been done yet, skip if so
    files_done = os.listdir(save_path)
    fit_files_this_layer = [ff for ff in files_done if layer_labels[ll] in ff and 'fastpars_avgspace' in ff]
    n_done = np.size(fit_files_this_layer)
    if n_done>=1:
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
          
      file_path = os.path.join(root,'activations',model,training_str,param_str,dataset,
                               'eval_at_ckpt-%s_orient_tuning'%(ckpt_str))
      
      file_name = os.path.join(file_path,'AllUnitsOrientTuningAvgSpace_%s.npy'%(layers2load[ll]))
      print('loading from %s\n'%file_name)
    
      # [nUnits x nSf x nOri]    
      t = np.load(file_name)
            
      if kk==0:
        nUnits = np.shape(t)[0]
        nSF = np.shape(t)[1]
        nOri = np.shape(t)[2]
#        ori_axis = np.arange(0, nOri,1) 
        all_units = np.zeros([nSamples, nUnits, nSF, nOri])
      
      print('analyzing tuning curves for %d units, %d spatial frequencies, orientations 0-%d deg\n'%(nUnits, nSF,nOri))
      
      # [nSamples x nUnits x nSF x nOri]
      all_units[kk,:,:,:] = t
    
    #%% now identify the non-responsive units in this layer.
    nUnits = np.shape(all_units)[1]
    
    resp_units = all_units
   
    mean_TF = np.mean(resp_units,1)
    
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
    
    
    #%% Save the responsive units   
    save_name =os.path.join(save_path,'%s_mean_resp_avgspace_eval_at_ckpt_%s.npy'%(layer_labels[ll],ckpt_str_print))
    print('saving to %s\n'%save_name)
    np.save(save_name,mean_TF)
 
    save_name =os.path.join(save_path,'%s_fastpars_avgspace_eval_at_ckpt_%s.npy'%(layer_labels[ll],ckpt_str_print))
    print('saving to %s\n'%save_name)
    np.save(save_name,fastpars)
       
    resp_units=None
    all_units=None
    
 #%% main function to decide between subfunctions
  
if __name__ == '__main__':
  
  root = sys.argv[1] #'The path where this project is located.
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
 
  analyze_orient_tuning(root, model, training_str, dataset_all, nSamples, param_str, ckpt_str)

   