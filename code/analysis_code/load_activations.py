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


def load_activ(model, dataset, training_str, param_str, ckpt_num):
    """ load the activations for the model specified"""
    
    info = dict()
    
    #%% define where all the activations should be found
      
    weight_path = os.path.join(root, 'activations', model, training_str, param_str,  dataset, 'eval_at_ckpt-' + ckpt_num + '_reduced')
   
    #%% list all the layers of this network
    if 'nasnet'==model:
        
        layer_labels = []
        for cc in range(17):
            layer_labels.append('Cell_%d' % (cc+1))
        layer_labels.append('global_pool')
        layer_labels.append('logits')
        info['layer_labels'] = layer_labels
     
    elif 'inception'==model:
        
        # list of all the endpoints in this network.
        layer_labels = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3',
         'MaxPool_3a_3x3','Conv2d_3b_1x1','Conv2d_4a_3x3', 'MaxPool_5a_3x3',
         'Mixed_5b','Mixed_5c','Mixed_5d','Mixed_6a','Mixed_6b','Mixed_6c',
         'Mixed_6d','Mixed_6e','Mixed_7a', 'Mixed_7b','Mixed_7c',
         'AuxLogits','AvgPool_1a','PreLogits','Logits','Predictions']       
        info['layer_labels'] = layer_labels
     
    elif 'vgg16'==model:
        
        # list of all the endpoints in this network (these are the exact strings we need to use to load files)
        layer_labels = ['conv1_conv1_1','conv1_conv1_2','pool1',
         'conv2_conv2_1','conv2_conv2_2','pool2',
         'conv3_conv3_1','conv3_conv3_2','conv3_conv3_3','pool3',
         'conv4_conv4_1','conv4_conv4_2','conv4_conv4_3','pool4',
         'conv5_conv5_1','conv5_conv5_2','conv5_conv5_3','pool5',
         'fc6','fc7', 'fc8','logits']  
        
        nLayers = len(layer_labels)
        for nn in range(nLayers-1):
             layer_labels[nn] = 'vgg_16_' + layer_labels[nn]
            
        # returning a slightly different version of these names, shorter, for use in plots
        layer_labels_plot = ['conv1_1','conv1_2','pool1',
         'conv2_1','conv2_2','pool2',
         'conv3_1','conv3_2','conv3_3','pool3',
         'conv4_1','conv4_2','conv4_3','pool4',
         'conv5_1','conv5_2','conv5_3','pool5',
         'fc6','fc7','fc8','logits']
    
        info['layer_labels'] = layer_labels_plot

    else:
        raise ValueError('model string not recognized')
            
    #%% load a list of the features used in this dataset
    if 'ori' in dataset:
        raise ValueError('dataset string not recognized')
    else:
        feat_path = os.path.join(root, 'datasets','gratings',dataset,'featureMat.npy')
    
    featureMat_orig = np.load(feat_path)
    #%% list all the parameters of the image set that was used for this round of evaluation

    if dataset=='SpatFreqGratings':
        [u,noiselist] = np.unique(featureMat_orig[:,0],return_inverse=True)
        noiselist = np.expand_dims(noiselist,1)      
        [u,exlist] = np.unique(featureMat_orig[:,1],return_inverse=True)
        exlist=  np.expand_dims(exlist,1)
        [u,typelist] = np.unique(featureMat_orig[:,2],return_inverse=True)
        typelist = np.expand_dims(typelist,1)
        [u,orilist] = np.unique(featureMat_orig[:,3],return_inverse=True)
        orilist = np.expand_dims(orilist,1)
        [u,sflist] = np.unique(featureMat_orig[:,4],return_inverse=True)
        sflist = np.expand_dims(sflist,1)
        [u,phaselist] = np.unique(featureMat_orig[:,5],return_inverse=True)
        phaselist = np.expand_dims(phaselist,1)
        contrastlist = np.zeros(np.shape(phaselist))

        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.01]
        info['sf_vals'] = np.logspace(np.log10(0.02),np.log10(.4),6)   
        info['contrast_levels'] = [0.8]
        info['phase_vals'] = [0,180]
        nPhase=2
        nEx = 4
     
    else:
        raise ValueError('model string not recognized')

    #%% list out all the stimuli
    nOri=180
    info['xx']=np.arange(0,nOri,1)
    info['stim_types'] = ['Gaus'] 
  
    nSF=np.size(info['sf_vals'])   
    nType = np.size(info['stim_types'])
    nNoiseLevels = np.size(info['noise_levels'])
    nContrastLevels = np.size(info['contrast_levels'])
    
    info['nEx'] = nEx
    info['nContrastLevels'] = nContrastLevels    
    info['nPhase'] = nPhase
    info['nSF'] = nSF
    info['nType'] = nType
    info['nNoiseLevels'] = nNoiseLevels
    info['nLayers'] = np.size(layer_labels)
    info['nTimePts'] = np.size(info['timepoint_labels'])
    
    assert(nNoiseLevels==1 or nContrastLevels==1)
    
    # list all the image features in a big matrix, where every row is unique.
    info['noiselist'] = noiselist
    info['contrastlist'] = contrastlist
    info['exlist'] = exlist
    info['typelist'] = typelist
    info['orilist']=orilist
    info['sflist']=sflist
    info['phaselist']=phaselist
    
    featureMat = np.concatenate((info['noiselist'],info['contrastlist'],info['exlist'],info['typelist'],info['orilist'],info['sflist'],info['phaselist']),axis=1)
    
    assert np.shape(featureMat)==np.shape(np.unique(featureMat, axis=0))
          
   
    #%% load the data (already in reduced/PCA-d format)
    
    allw = []   
    
    for ll in range(np.size(layer_labels)):

        file = os.path.join(weight_path, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w1 = np.load(file)
        w2 = []
                      
        allw.append([w1,w2])
       
    #% load the predicted orientation labels from the last layer
       
    all_labs = []
    
    if 'inception' in model:   
        file = os.path.join(weight_path, 'allStimsLabsPredicted_Logits.npy')   
    else:             
        file = os.path.join(weight_path, 'allStimsLabsPredicted_logits.npy')  
    labs1 = np.load(file)
    labs2 = []                
         
    all_labs.append([labs1,labs2])
        
    return allw, all_labs, info

