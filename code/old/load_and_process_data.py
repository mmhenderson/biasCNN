#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:56:54 2019

@author: mmhender
"""

#%% Set up paths and general parameters

root = '/usr/local/serenceslab/maggie/biasCNN/';
import os
os.chdir(os.path.join(root, 'code'))

import numpy as np



def load_data_nasnet_oriTst0():
      
    # load in all my data for this version of the experiment, to be used for analysis
    
    info = dict()
       
    weight_path_before = os.path.join(root, 'activations', 'nasnet_activations_short_reduced')
    weight_path_after = os.path.join(root, 'activations', 'nasnet_activations_long_reduced')

    info['timepoint_labels'] = ['before retraining','after retraining']
    
    layer_labels = []
    for cc in range(17):
        layer_labels.append('Cell_%d' % (cc+1))
    layer_labels.append('global_pool')
    layer_labels.append('logits')
    
    info['layer_labels'] = layer_labels
    
    #%% information about the stimuli. 
         
    info['noise_levels'] = [0, 0.2, 0.4, 0.6, 0.8]    
    info['sf_vals'] = np.logspace(np.log10(0.2), np.log10(2),5)
    info['stim_types'] = ['Full','Gaus']
    
    nOri=180
    nPhase=4
    nSF=np.size(info['sf_vals'])   
    nType = np.size(info['stim_types'])
    nNoiseLevels = np.size(info['noise_levels'])
    
    # list all the image features in a big matrix, where every row is unique.
    info['noiselist'] = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
    info['typelist'] = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
    info['orilist']=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
    info['sflist']=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
    info['phaselist']=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))
    
    featureMat = np.concatenate((info['noiselist'],info['typelist'],info['orilist'],info['sflist'],info['phaselist']),axis=1)
    
    assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
    
      
    info['xx']=np.arange(0,180,1)
    
    
    #%% load the data (already in reduced/PCA-d format)
    
    allw = []   
    
    for ll in range(np.size(layer_labels)):
               
        for nn in range(np.size(info['noise_levels'])):

            file = os.path.join(weight_path_before, 'noise%.2f' % info['noise_levels'][nn], 'allStimsReducedWts_%s.npy' % layer_labels[ll])
            w1 = np.load(file)
            
            file = os.path.join(weight_path_after, 'noise%.2f' % info['noise_levels'][nn], 'allStimsReducedWts_%s.npy' % layer_labels[ll])
            w2 = np.load(file)
            
            if nn==0:
                w1_all = w1
                w2_all = w2
            else:
                w1_all = np.concatenate((w1_all, w1), axis=0)
                w2_all = np.concatenate((w2_all, w2), axis=0)
                
        allw.append([w1_all,w2_all])
       
    #%% load the predicted orientation labels from the last layer
       
    all_labs = []
    
    for nn in range(np.size(info['noise_levels'])):
    
        file = os.path.join(weight_path_before, 'noise%.2f' %(info['noise_levels'][nn]), 'allStimsLabsPredicted_Cell_1.npy')    
        labs1 = np.load(file)
         
        file = os.path.join(weight_path_after, 'noise%.2f' %(info['noise_levels'][nn]), 'allStimsLabsPredicted_Cell_1.npy')    
        labs2 = np.load(file)
        
        if nn==0:
            labs1_all = labs1
            labs2_all = labs2
        else:
            labs1_all = np.concatenate((labs1_all, labs1), axis=0)
            labs2_all = np.concatenate((labs2_all, labs2), axis=0)
            
        
    all_labs.append([labs1_all,labs2_all])
        
    return allw, all_labs, info
#%%

def load_data_nasnet_oriTst1():
    
    # load in all my data for this version of the experiment, to be used for analysis
    info = dict()
    
    weight_path_before = os.path.join(root, 'activations', 'nasnet_oriTst1_short_reduced')
    weight_path_after = os.path.join(root, 'activations', 'nasnet_oriTst1_long_reduced')
    
    info['timepoint_labels'] = ['before retraining','after retraining']
    
    layer_labels = []
    for cc in range(17):
        layer_labels.append('Cell_%d' % (cc+1))
    layer_labels.append('global_pool')
    layer_labels.append('logits')
    
    info['layer_labels'] = layer_labels
    
    #%% information about the stimuli. 
         
    info['noise_levels'] = [0, 0.2, 0.4, 0.6, 0.8]    
    info['sf_vals'] = np.logspace(np.log10(0.2), np.log10(2),5)
    info['stim_types'] = ['Full','Gaus']
    
    nOri=180
    nPhase=4
    nSF=np.size(info['sf_vals'])   
    nType = np.size(info['stim_types'])
    nNoiseLevels = np.size(info['noise_levels'])
    
    # list all the image features in a big matrix, where every row is unique.
    info['noiselist'] = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
    info['typelist'] = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
    info['orilist']=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
    info['sflist']=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
    info['phaselist']=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))
    
    featureMat = np.concatenate((info['noiselist'],info['typelist'],info['orilist'],info['sflist'],info['phaselist']),axis=1)
    
    assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
          
    info['xx']=np.arange(0,180,1)
        
    #%% load the data (already in reduced/PCA-d format)
    
    allw = []
    
    for ll in range(np.size(layer_labels)):
        
        tmp = []

        file = os.path.join(weight_path_before, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w1 = np.load(file)
        
        file = os.path.join(weight_path_after, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w2 = np.load(file)
        
        tmp.append([w1,w2])
            
        allw.append(tmp)

    #%% load the predicted orientation labels from the last layer
  
    all_labs = []
    
    file = os.path.join(weight_path_before, 'allStimsLabsPredicted_Cell_1.npy')    
    labs1 = np.load(file)
     
    file = os.path.join(weight_path_after,'allStimsLabsPredicted_Cell_1.npy')    
    labs2 = np.load(file)
    
    all_labs.append([labs1,labs2])
     
    
    return allw, all_labs, info