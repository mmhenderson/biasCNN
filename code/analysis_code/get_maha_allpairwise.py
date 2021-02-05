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

def get_dist(activ_path,save_path,model_name,dataset_name,ncomp2use=10):
  
  #%% get information for this model and dataset
  info=load_activations.get_info(model_name, dataset_name)

  # extract some things from info
  layers2load=info['layer_labels_full']  
  nLayers = info['nLayers']
 
  if not os.path.exists(save_path):
    os.makedirs(save_path)
     
  orilist = info['orilist']
  sflist = info['sflist']

  nOri=180
  orilist_adj = orilist

  nSF = np.size(np.unique(sflist))

  mdist_all = np.zeros([nLayers, nSF, nOri, nOri])
 
  #%% loop over layers
  for ll in range(nLayers):
    
      #%% load all activations
      
      fn = os.path.join(activ_path,'allStimsReducedWts_%s.npy'%layers2load[ll])
      print('loading reduced activations from %s\n'%fn)
      allw = np.load(fn)
      # allw is nIms x nFeatures
   
      # taking a specified number of PCs here
      allw = allw[:,0:ncomp2use]
      
      #%% make sure no bad units that will mess up calculations
      constant_inds = np.all(np.equal(allw, np.tile(np.expand_dims(allw[0,:], axis=0), [np.shape(allw)[0], 1])),axis=0)      
      assert(np.sum(constant_inds)==0)
      is_inf = np.any(allw==np.inf, axis=0)
      assert(np.sum(is_inf)==0)
      #%% loop over all spatial frequencies (usually just one) and delta values and number of component to include...
      
      print('calculating all pairwise maha distances using first %d PCs [%d x %d]...\n'%(ncomp2use,np.shape(allw)[0],np.shape(allw)[1]))
      for sf in range(nSF):
                
        for oo1 in range(nOri):
          
          for oo2 in np.arange(oo1+1,nOri):
            
            dat1 = allw[np.where(np.logical_and(orilist_adj==oo1, sflist==sf))[0],:]
            dat2 = allw[np.where(np.logical_and(orilist_adj==oo2, sflist==sf))[0],:]
            
            mdist = classifiers.get_mahal_dist(dat1,dat2)
            
            if np.any(np.isnan(mdist)):
              print('warning: there are some nan elements in matrix for %s'%layers2load[ll])

            mdist_all[ll,sf,oo1,oo2] = np.squeeze(mdist)

  #%% save everything into one big file
  save_name =os.path.join(save_path,'Mahal_dist_all_pairwise.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,mdist_all)
  
#%% main function to decide between subfunctions
  
if __name__ == '__main__':
  
  activ_path = sys.argv[1] #The path to load the activation files from.'
  save_path = sys.argv[2] #The path to save the FI calculation to
  model_name = sys.argv[3] # The name of the current model.
  dataset_name = sys.argv[4] #The name of the dataset.
  ncomp2use = int(sys.argv[5])
  
  print('\nactiv_path is %s'%activ_path)
  print('save_path is %s'%save_path)
  print('model name is %s'%model_name)  
  print('dataset name is %s'%dataset_name)
  print('ncomp2use is %d'%ncomp2use)
  
  get_dist(activ_path, save_path, model_name, dataset_name, ncomp2use)
