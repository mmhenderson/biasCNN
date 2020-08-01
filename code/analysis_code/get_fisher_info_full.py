#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Fisher information at each layer of a trained network, by loading 
full activation patterns for each stimulus and computing derivative of each 
unit's orientation response profile. Save the result to desired path.

"""
import os
import numpy as np
import classifiers_custom as classifiers
import sys
import load_activations
from copy import deepcopy

# values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)

def get_fisher_info(activ_path,save_path,model_name,dataset_name,num_batches):
  
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

  fisher_info = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  deriv2 = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  varpooled = np.zeros([nLayers, nSF, 180, np.size(delta_vals)])
  
 
  #%% loop over layers
  for ll in range(nLayers):
    
      #%% load all activations
      allw = None
      
      for bb in np.arange(0,num_batches,1):
  
#          file = os.path.join(activ_path, 'batch' + str(0) +'_' + layers2load[ll] +'.npy')
          file = os.path.join(activ_path, 'batch' + str(bb) +'_' + layers2load[ll] +'.npy')
          print('loading from %s\n' % file)
          w = np.squeeze(np.load(file))
          # w will be nIms x nFeatures
          w = np.reshape(w, [np.shape(w)[0], np.prod(np.shape(w)[1:])])
          
          if bb==0:
              allw = w
#              allw = w[:,0:3]
          else:
              allw = np.concatenate((allw, w), axis=0)
#              allw = np.concatenate((allw, w[:,0:3]), axis=0)
              
      #%% remove any bad units that will mess up Fisher information calculation  
      # first take out all the constant units, leaving only units with variance over images
      constant_inds = np.all(np.equal(allw, np.tile(np.expand_dims(allw[0,:], axis=0), [np.shape(allw)[0], 1])),axis=0)      
      print('removing units with no variance (%d out of %d)...\n'%(np.sum(constant_inds),np.size(constant_inds)))
      allw = allw[:,~constant_inds]
      # also remove any units with infinite responses (usually none)
      is_inf = np.any(allw==np.inf, axis=0)
      print('removing units with inf response (%d out of %d)...\n'%(np.sum(is_inf),np.size(is_inf)))
      allw = allw[:,~is_inf]
      if np.shape(allw)[1]>0:
      
        #%% get fisher information for this layer, each spatial frequency
        print('calculating fisher information across all remaining units [%d x %d]...\n'%(np.shape(allw)[0],np.shape(allw)[1]))
        for sf in range(nSF):
           
          inds = np.where(sflist==sf)[0]
          
          for dd in range(np.size(delta_vals)):
              
            ori_axis, fi, d, v = classifiers.get_fisher_info(allw[inds,:],orilist_adj[inds],delta=delta_vals[dd])
            if nOri==360:
              fi = np.reshape(fi,[2,180])
              fi = np.mean(fi,axis=0)
              d = np.reshape(d,[2,180])
              d = np.mean(d,axis=0)
              v = np.reshape(v,[2,180])
              v = np.mean(v,axis=0)
              
            if np.any(np.isnan(fi)):
              print('warning: there are some nan elements in fisher information matrix for %s'%layers2load[ll])
  #          assert(not np.any(np.isnan(fi)))
  #          assert(not np.any(np.isnan(d)))
  #          assert(not np.any(np.isnan(v)))
            fisher_info[ll,sf,:,dd] = np.squeeze(fi)
            deriv2[ll,sf,:,dd] = np.squeeze(d)
            varpooled[ll,sf,:,dd] = np.squeeze(v) 
      else:
        print('skipping %s, no units left\n'%layers2load[ll])
  #%% save everything into one big file
  save_name =os.path.join(save_path,'Fisher_info_all_units.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,fisher_info)
  
  # also saving these intermediate values (numerator and denominator of FI expressions)
  save_name =os.path.join(save_path,'Deriv_sq_all_units.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,deriv2)
  
  save_name =os.path.join(save_path,'Pooled_var_all_units.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,varpooled)
  
#%% main function to decide between subfunctions
  
if __name__ == '__main__':
  
  activ_path = sys.argv[1] #The path to load the big activation files from.'
  save_path = sys.argv[2] #The path to save the reduced activation files to.  
  model_name = sys.argv[3] # The name of the current model.
  dataset_name = sys.argv[4] #The name of the dataset.
  num_batches = int(sys.argv[5]) # The number of batches in the full set of images
  
  print('\nactiv_path is %s'%activ_path)
  print('save_path is %s'%save_path)
  print('model name is %s'%model_name)  
  print('dataset name is %s'%dataset_name)
  print('num_batches is %d'%num_batches)
 
  get_fisher_info(activ_path, save_path, model_name, dataset_name, num_batches)
