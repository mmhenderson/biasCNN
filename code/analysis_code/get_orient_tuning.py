#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate the orientation tuning of units at each layer of a trained network, 
by loading activation patterns from an image set and averaging trials with same orientation.
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
#  sf_vals = info['sf_vals'][0:nSF]

  # get the correct orientation vals to use
  if info['nPhase']==2:
    # make a full 360 deg space
    orilist = deepcopy(info['orilist'])
#    orilist_orig=deepcopy(info['orilist'])
    phaselist=deepcopy(info['phaselist'])
    orilist[phaselist==1] = orilist[phaselist==1]+180
  else: 
    # use the regular 180 deg space
    orilist = info['orilist'] 
#    orilist_orig=orilist
    
#  exlist=info['exlist']
  sflist = info['sflist']
  
  ori_axis = np.unique(orilist);
  nOri = np.size(ori_axis)
  
  layers2load = info['layer_labels_full']
  nLayers = np.size(layers2load)
  
  print('dataset %s has %d spatial freqs, orients from 0-%d deg\n'%(dataset_name,nSF,nOri))
  
  # if the save directory already exists, clear it before beginning.
  if os.path.isdir(save_path):
      print('deleting contents of %s' % save_path)
      shutil.rmtree(save_path, ignore_errors = True)
      os.mkdir(save_path)    
  else:
      os.mkdir(save_path)   
    
  for ll in range(nLayers):
      allw = None
      
      for bb in np.arange(0,num_batches,1):
  
          file = os.path.join(load_path, 'batch' + str(bb) +'_' + layers2load[ll] +'.npy')
          print('loading from %s\n' % file)
          w = np.squeeze(np.load(file))
#          # full W here is NHWC format: Number of images x Height (top to bottom) x Width (left ro right) x Channels.
#          # new w will be nIms x nFeatures
#          N = np.shape(w)[0]
#          H = np.shape(w)[1]
#          W = np.shape(w)[2]
#          C = np.shape(w)[3]            
#          clabs = np.tile(np.expand_dims(np.arange(0,C),axis=1),[H*W,1])
#          wlabs = np.expand_dims(np.repeat(np.tile(np.expand_dims(np.arange(0,W),axis=1),[H,1]), C),axis=1)
#          hlabs = np.expand_dims(np.repeat(np.arange(0,H),W*C),axis=1)
#          coords = np.concatenate((hlabs,wlabs,clabs),axis=1)      
#          assert np.array_equal(coords, np.unique(coords, axis=0))
          w = np.reshape(w, [np.shape(w)[0], np.prod(np.shape(w)[1:])])
          
          if bb==0:
              allw = w
          else:
              allw = np.concatenate((allw, w), axis=0)
      
      #%% get the tuning curves for every unit in this layer
      nUnits = np.shape(allw)[1]
      tuning_curves = np.zeros([nUnits, nSF, nOri])
      for sf in range(nSF):
        for ii in range(nOri):
          
          # find all stims that have this orientation and spatial frequency
          inds2use = np.where(np.logical_and(sflist==sf, orilist==ori_axis[ii]))[0]
          
          # take mean over all stims with this feature value
          # [nUnits long]
          meandat = np.mean(allw[inds2use,:],axis=0)
          
          # put this into the big tuning curve matrix
          #[nUnits x nSF x nOri]
          tuning_curves[:,sf,ii] = meandat
          
      #%% Save the result as a single file
      
      fn2save = os.path.join(save_path, 'AllUnitsOrientTuning_' + layers2load[ll] +'.npy')
      
      np.save(fn2save, tuning_curves)
      print('saving to %s\n' % (fn2save))
      
        
if __name__ == '__main__':
  
  load_path = sys.argv[1] #'The path to load the big activation files from.'
  save_path = sys.argv[2] # The path to save the reduced activation files to.
  model_name = sys.argv[3] #  the name of the model
  dataset_name = sys.argv[4] # the name of the dataset
  num_batches = int(sys.argv[5])  # how many batches are there in total?
  
  print('\nload_path is %s'%load_path)
  print('save_path is %s'%save_path)
  print('model name is %s'%model_name)
  print('dataset name is %s'%dataset_name)
  print('num batches is %d'%num_batches)

  get_orient_tuning_fn(load_path,save_path,model_name,dataset_name,num_batches)
