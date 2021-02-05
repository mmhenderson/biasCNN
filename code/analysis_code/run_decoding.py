#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate pattern-level info at each layer of a trained network, by loading 
activation patterns for each stimulus, and performing SVM decoding between 
activations for different orientation stimuli.
Save the result to desired path.

"""
import os
import numpy as np
import sys
import load_activations
from copy import deepcopy
from sklearn import svm


def get_decoding(activ_path,save_path,model_name,dataset_name,num_batches,min_var_expl=99):
  
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

  dec_acc = np.zeros([nLayers, nSF, nOri, nOri])

  #%% loop over layers
  for ll in range(nLayers):
    
      #%% load all activations

      fn = os.path.join(activ_path,'allStimsReducedWts_%s.npy'%layers2load[ll])
      w = np.load(fn)
      # w is nIms x nFeatures
      
      # take the specified number of components to do decoding with
      fn2 = os.path.join(activ_path,'allStimsVarExpl_%s.npy'%layers2load[ll])
      var_expl = np.load(fn2)
      ncomp2keep = np.where(np.cumsum(var_expl)>min_var_expl/100)
      if np.size(ncomp2keep)==0:
        ncomp2keep=np.shape(w)[1]
      else:
        ncomp2keep = ncomp2keep[0][0]
        
      w = w[:,0:ncomp2keep]
      
      #%% run decoding
      print('decoding analysis for %s (after reducing w pca), size is [%d x %d]...\n'%(layers2load[ll],np.shape(w)[0],np.shape(w)[1]))
      for sf in range(nSF):
      
        for oo1 in range(nOri):
          
          for oo2 in np.arange(oo1+1,nOri):
            
            dat1 = w[np.where(np.logical_and(orilist_adj==oo1, sflist==sf))[0],:]
            dat2 = w[np.where(np.logical_and(orilist_adj==oo2, sflist==sf))[0],:]

            # run binary decoder w cross-validation
            clf=svm.LinearSVC(max_iter=10000000)
            
            assert(np.shape(dat1)[0]==np.shape(dat2)[0])
            nCrossVal=np.shape(dat1)[0]
           
            alldat = np.concatenate((dat1,dat2),axis=0)
            allreallabs = np.repeat([1,2],np.shape(dat1)[0])
            allcvlabs = np.tile(np.arange(0,nCrossVal),2)
            allpredlabs = np.zeros(np.shape(allreallabs))
                             
            print('%s, decoding %d versus %d\n'%(layers2load[ll],oo1,oo2))
            for cv in range(nCrossVal):
              
              trninds = np.where(allcvlabs!=cv)[0]
              tstinds = np.where(allcvlabs==cv)[0]
              
              clf.fit(alldat[trninds,:],allreallabs[trninds])
                           
              predlabs = clf.predict(alldat[tstinds,:])
              
              allpredlabs[tstinds] = predlabs
              
            acc = np.mean(allpredlabs==allreallabs)
            
            dec_acc[ll,sf,oo1,oo2] = acc
      
  #%% save everything into one big file
  
  save_name =os.path.join(save_path,'Dec_acc_all.npy')
  print('saving to %s\n'%save_name)
  np.save(save_name,dec_acc)
 
#%% main function to decide between subfunctions
  
if __name__ == '__main__':
  
  activ_path = sys.argv[1] #The path to load the reduced activation files from.'
  save_path = sys.argv[2] #The path to save the distance estimates to. 
  model_name = sys.argv[3] # The name of the current model.
  dataset_name = sys.argv[4] #The name of the dataset.
  num_batches = int(sys.argv[5]) # The number of batches in the full set of images
  min_var_expl = int(sys.argv[6]) # The number of batches in the full set of images
  
  print('\nactiv_path is %s'%activ_path)
  print('save_path is %s'%save_path)
  print('model name is %s'%model_name)  
  print('dataset name is %s'%dataset_name)
  print('num_batches is %d'%num_batches)
  print('min_var_expl is %d'%min_var_expl)
 
  get_decoding(activ_path, save_path, model_name, dataset_name, num_batches, min_var_expl)
