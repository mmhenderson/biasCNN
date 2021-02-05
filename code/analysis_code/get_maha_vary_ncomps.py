#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate distance bw responses in multidimensional PC space - normalize by the variability of distances

"""
import os
import numpy as np
import classifiers_custom as classifiers
import sys
import load_activations
from copy import deepcopy

# values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)

ncomp2do = np.arange(2,48,1)

def get_dist(activ_path,save_path,model_name,dataset_name):
  
  #%% get information for this model and dataset
  info=load_activations.get_info(model_name, dataset_name)

  # extract some things from info
  layers2load=info['layer_labels_full']  
  nLayers = info['nLayers']
 
  if not os.path.exists(save_path):
    os.makedirs(save_path)
     
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
     
  nSF = np.size(np.unique(sflist))

  mdist_all = np.zeros([nLayers, nSF, 180, len(ncomp2do), np.size(delta_vals)])
 
  #%% loop over layers
  for ll in range(nLayers):
    
      #%% load all activations
      
      fn = os.path.join(activ_path,'allStimsReducedWts_%s.npy'%layers2load[ll])
      print('loading reduced activations from %s\n'%fn)
      allw = np.load(fn)
      # allw is nIms x nFeatures
   
      #%% make sure no bad units that will mess up calculations
      constant_inds = np.all(np.equal(allw, np.tile(np.expand_dims(allw[0,:], axis=0), [np.shape(allw)[0], 1])),axis=0)      
      assert(np.sum(constant_inds)==0)
      is_inf = np.any(allw==np.inf, axis=0)
      assert(np.sum(is_inf)==0)
      #%% loop over all spatial frequencies (usually just one) and delta values and number of component to include...
      
      print('calculating maha distances across all remaining units [%d x %d]...\n'%(np.shape(allw)[0],np.shape(allw)[1]))
      for sf in range(nSF):
         
        inds = np.where(sflist==sf)[0]
        
        for dd in range(np.size(delta_vals)):
          
          for nn in range(len(ncomp2do)):
            
            axis, mdist = classifiers.get_mahal_dist_loop(allw[inds,0:ncomp2do[nn]],orilist_adj[inds],delta=delta_vals[dd])
            if nOri==360:
              mdist = np.reshape(mdist,[180,2],order='F')
              mdist = np.mean(mdist,axis=1)

            if np.any(np.isnan(mdist)):
              print('warning: there are some nan elements in matrix for %s'%layers2load[ll])

            mdist_all[ll,sf,:,nn,dd] = np.squeeze(mdist)

  #%% save everything into one big file
  save_name =os.path.join(save_path,'Mahal_dist_vary_ncomps.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,mdist_all)
  
#%% main function to decide between subfunctions
  
if __name__ == '__main__':
  
  activ_path = sys.argv[1] #The path to load the activation files from.'
  save_path = sys.argv[2] #The path to save the FI calculation to
  model_name = sys.argv[3] # The name of the current model.
  dataset_name = sys.argv[4] #The name of the dataset.
 
  print('\nactiv_path is %s'%activ_path)
  print('save_path is %s'%save_path)
  print('model name is %s'%model_name)  
  print('dataset name is %s'%dataset_name)
 
  get_dist(activ_path, save_path, model_name, dataset_name)
