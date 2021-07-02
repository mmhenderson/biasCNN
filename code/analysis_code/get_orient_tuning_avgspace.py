#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate the orientation tuning of units at each layer of a trained network, 
by loading activation patterns from an image set and averaging trials with same orientation.
Also generate spatially-averaged TFs by averaging over the spatial dimension of units 
(i.e. getting a single value for each channel/filter)

"""

import os
import numpy as np
import shutil
import sys
import load_activations
from copy import deepcopy

def get_orient_tuning_fn(load_path, save_path, model_name, dataset_name, num_batches):
  
  #%% get info for the netwok
  info = load_activations.get_info(model_name, dataset_name)
  
  # extract some things from info  
  sflist = info['sflist']  
  nSF = np.size(np.unique(sflist))

  # get the correct orientation vals to use
  if info['nPhase']==2:
    # make a full 360 deg space
    orilist = deepcopy(info['orilist'])
    phaselist=deepcopy(info['phaselist'])
    orilist[phaselist==1] = orilist[phaselist==1]+180
  else: 
    # use the regular 180 deg space
    orilist = info['orilist'] 

  sflist = info['sflist']
  
  ori_axis = np.unique(orilist);
  nOri = np.size(ori_axis)
  
  layers2load = info['layer_labels_full']
  nLayers = np.size(layers2load)
  
  print('dataset %s has %d spatial freqs, orients from 0-%d deg, %d unique phases\n'%(dataset_name,nSF,nOri, info['nPhase']))
  
  # if the save directory already exists, clear it before beginning.
  if os.path.isdir(save_path):
      print('deleting contents of %s' % save_path)
      shutil.rmtree(save_path, ignore_errors = True)
      os.mkdir(save_path)    
  else:
      os.mkdir(save_path)   
    
  for ll in range(nLayers):
      allw = None
      allw_avgspace = None
      
      for bb in np.arange(0,num_batches,1):
  
          file = os.path.join(load_path, 'batch' + str(bb) +'_' + layers2load[ll] +'.npy')
          print('loading from %s\n' % file)
          w = np.load(file)
          # full W here is NHWC format: Number of images x Height (top to bottom) x Width (left ro right) x Channels.
          
          # AVERAGE OVER THE SPATIAL DIMENSIONS - GET JUST NCHANNELS UNITS TO USE
          if 'fc' in layers2load[ll]:
            w_avg = np.squeeze(w) # if fully connected, no spatial dims to average
          else:            
            w_avg = np.mean(np.mean(w, axis=2),axis=1)            
            assert(np.shape(w_avg)[0]==np.shape(w)[0] and np.shape(w_avg)[1]==np.shape(w)[3])
         
          # RESHAPE SO ALL FEATURES ARE IN A SINGLE DIMENSION
          w = np.reshape(w, [np.shape(w)[0], np.prod(np.shape(w)[1:])])
          
          if bb==0:
              allw_avgspace = w_avg
              allw = w
          else:
              allw = np.concatenate((allw, w), axis=0)
              allw_avgspace = np.concatenate((allw_avgspace, w_avg), axis=0)
      
      #%% get the tuning curves for every unit in this layer
      nUnits = np.shape(allw)[1]
      nUnits_avgspace = np.shape(allw_avgspace)[1]
      print('\nprocessing tuning curves for %d total units and %d spatially-averaged "units"'%(nUnits, nUnits_avgspace))
      tuning_curves = np.zeros([nUnits, nSF, nOri])
      tuning_curves_avgspace = np.zeros([nUnits_avgspace, nSF, nOri])
      
      for sf in range(nSF):
        for ii in range(nOri):
          
          # find all stims that have this orientation and spatial frequency
          inds2use = np.where(np.logical_and(sflist==sf, orilist==ori_axis[ii]))[0]
          
          # take mean over all stims with this feature value
          # [nUnits long]
          meandat = np.mean(allw[inds2use,:],axis=0)
          meandat_avgspace =np.mean(allw_avgspace[inds2use,:], axis=0)
          
          # put this into the big tuning curve matrix
          #[nUnits x nSF x nOri]
          tuning_curves[:,sf,ii] = meandat
          tuning_curves_avgspace[:,sf,ii] = meandat_avgspace
        
      #%% Save the result as a single file
      
      fn2save = os.path.join(save_path, 'AllUnitsOrientTuning_' + layers2load[ll] +'.npy') 
      print('saving to %s\n' % (fn2save))  
      np.save(fn2save, tuning_curves)         
      
      fn2save = os.path.join(save_path, 'AllUnitsOrientTuningAvgSpace_' + layers2load[ll] +'.npy')  
      print('saving to %s\n' % (fn2save))
      np.save(fn2save, tuning_curves_avgspace)      
      
      fn2save = os.path.join(save_path, 'AllActivAvgSpace_' + layers2load[ll] +'.npy')  
      print('saving to %s\n' % (fn2save))
      np.save(fn2save, allw_avgspace)      
      
        
if __name__ == '__main__':
  
  load_path = sys.argv[1] #'The path to load the big activation files from.'
  save_path = sys.argv[2] # The path to save the tuning curves files to.
  model_name = sys.argv[3] #  the name of the model
  dataset_name = sys.argv[4] # the name of the dataset
  num_batches = int(sys.argv[5])  # how many batches are there in total?
  
  print('\nload_path is %s'%load_path)
  print('save_path is %s'%save_path)
  print('model name is %s'%model_name)
  print('dataset name is %s'%dataset_name)
  print('num batches is %d'%num_batches)

  get_orient_tuning_fn(load_path,save_path,model_name,dataset_name,num_batches)
