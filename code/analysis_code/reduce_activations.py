#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reduce dimensionality of activations for each layer of the network, so that 
they're small enough to do further analyses (i.e. multivariate Fisher info)

"""
import os
import numpy as np
import sys
import load_activations
from sklearn import decomposition

def reduce_activ(raw_path,reduced_path,model_name,dataset_name,num_batches,min_var_expl, max_comp_keep=1000, min_comp_keep=10):
  
  #%% get information for this model and dataset
  info=load_activations.get_info(model_name, dataset_name)

  # extract some things from info
  layers2load=info['layer_labels_full']  
  nLayers = info['nLayers']
 
  if not os.path.exists(reduced_path):
    os.makedirs(reduced_path)
 
  #%% loop over layers
  for ll in range(nLayers):
    
      #%% first loading all activations, raw (large) format
      allw = None
      
      for bb in np.arange(0,num_batches,1):
  
          file = os.path.join(raw_path, 'batch' + str(bb) +'_' + layers2load[ll] +'.npy')
          print('loading from %s\n' % file)
          w = np.squeeze(np.load(file))
          # full W here is NHWC format: Number of images x Height (top to bottom) x Width (left ro right) x Channels.
          # new w will be nIms x nFeatures
          w = np.reshape(w, [np.shape(w)[0], np.prod(np.shape(w)[1:])])
          
          if bb==0:
              allw = w
          else:
              allw = np.concatenate((allw, w), axis=0)
           
      #%% now use PCA to reduce dimensionality
      
      pca = decomposition.PCA()
      print('size of allw before reducing is %d by %d'%(np.shape(allw)[0],np.shape(allw)[1]))
      print('\n STARTING PCA\n')      
      weights_reduced = pca.fit_transform(allw)   
      
      # decide how many components needed 
      var_expl = pca.explained_variance_ratio_     
      ncomp2keep = np.where(np.cumsum(var_expl)>min_var_expl/100)
      
      if np.size(ncomp2keep)==0:
        ncomp2keep = max_comp_keep
        print('need all the data to capture %d percent of variance, but max components is set to %d so keeping that many only!\n'%(min_var_expl, max_comp_keep))
      elif ncomp2keep[0][0]>max_comp_keep:
        ncomp2keep = max_comp_keep
        print('need >%d components to capture %d percent of variance, but max components is set to %d so keeping that many only!' % (max_comp_keep, min_var_expl, max_comp_keep))
      else:
        ncomp2keep = ncomp2keep[0][0]
        print('need %d components to capture %d percent of variance' % (ncomp2keep, min_var_expl))
        
      if ncomp2keep<min_comp_keep:
          ncomp2keep = min_comp_keep
          print('need only %d components to capture %d percent of variance, but keeping first %d!' % (ncomp2keep, min_var_expl, min_comp_keep))
    
      weights_reduced = weights_reduced[:,0:ncomp2keep]
    
      print('saving %d components\n'%np.shape(weights_reduced)[1])
      #%% Save the result as a single file
      
      fn2save = os.path.join(reduced_path, 'allStimsReducedWts_' + layers2load[ll] +'.npy')
      print('saving to %s\n' % (fn2save))
      np.save(fn2save, weights_reduced)
            
      fn2save = os.path.join(reduced_path, 'allStimsVarExpl_' + layers2load[ll] +'.npy')
      print('saving to %s\n' % (fn2save))
      np.save(fn2save, var_expl)      
 
#%% main function to decide between subfunctions
  
if __name__ == '__main__':
  
  raw_path = sys.argv[1] #The path to load the big activ files from
  reduced_path = sys.argv[2] #The path to save the reduced activs to.
  model_name = sys.argv[3] # The name of the current model.
  dataset_name = sys.argv[4] #The name of the dataset.
  num_batches = int(sys.argv[5]) # The number of batches in the full set of images
  min_var_expl = int(sys.argv[6]) # The amount of variance to explain w pca.
  max_comp_keep = int(sys.argv[7]) # The max components to save after pca.
  min_comp_keep = int(sys.argv[8]) # The min components to save after pca.
  
  print('\nraw_path is %s'%raw_path)
  print('reduced_path is %s'%reduced_path)
  print('model name is %s'%model_name)  
  print('dataset name is %s'%dataset_name)
  print('num_batches is %d'%num_batches)
  print('min_var_expl is %d'%min_var_expl)
  print('max_comp_keep is %d'%max_comp_keep)
  print('min_comp_keep is %d'%min_comp_keep)
 
  reduce_activ(raw_path, reduced_path, model_name, dataset_name, num_batches,min_var_expl, max_comp_keep, min_comp_keep)
