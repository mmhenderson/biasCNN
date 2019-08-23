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


def load_activ(model_str):
    """ load the activations for the model specified by model_str"""
    
    info = dict()
      
    #%% list all the layers of this network
    
    underscore = np.char.find(model_str,'_')
    dataset = model_str[underscore+1:]
    
    if 'nasnet' in model_str:
        
        layer_labels = []
        for cc in range(17):
            layer_labels.append('Cell_%d' % (cc+1))
        layer_labels.append('global_pool')
        layer_labels.append('logits')
        info['layer_labels'] = layer_labels
     
    elif 'inception' in model_str:
        
        # list of all the endpoints in this network.
        layer_labels = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3',
         'MaxPool_3a_3x3','Conv2d_3b_1x1','Conv2d_4a_3x3', 'MaxPool_5a_3x3',
         'Mixed_5b','Mixed_5c','Mixed_5d','Mixed_6a','Mixed_6b','Mixed_6c',
         'Mixed_6d','Mixed_6e','Mixed_7a', 'Mixed_7b','Mixed_7c',
         'AuxLogits','AvgPool_1a','PreLogits','Logits','Predictions']       
        info['layer_labels'] = layer_labels
     
    elif 'vgg16' in model_str:
        
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
        
    
     
    
    featureMat_orig = np.load(os.path.join(root, 'datasets',dataset,'featureMat.npy'))
    #%% list all the parameters of the image set that was used for this round of evaluation
   
    weight_path_before = os.path.join(root, 'activations', model_str + '_short_reduced')
    weight_path_after = os.path.join(root, 'activations', model_str + '_long_reduced')
    
    group1 = ['oriTst1','oriTst2','oriTst3','oriTst4','oriTst4a',
              'oriTst5','oriTst5a','oriTst6','oriTst6a','oriTst7','oriTst7a',
              'oriTst8','oriTst8a','oriTst8b','oriTst9',
              'oriTst10','oriTst10a'];
    group2 = ['oriTst9a']
    group3 = ['oriTst11','oriTst12']
    
    if dataset in group1:    
        [u,noiselist] = np.unique(featureMat_orig[:,0],return_inverse=True)
        noiselist = np.expand_dims(noiselist,1)       
        [u,typelist] = np.unique(featureMat_orig[:,1],return_inverse=True)
        typelist = np.expand_dims(typelist,1)
        [u,orilist] = np.unique(featureMat_orig[:,2],return_inverse=True)
        orilist = np.expand_dims(orilist,1)
        [u,sflist] = np.unique(featureMat_orig[:,3],return_inverse=True)
        sflist = np.expand_dims(sflist,1)
        [u,phaselist] = np.unique(featureMat_orig[:,4],return_inverse=True)
        phaselist = np.expand_dims(phaselist,1)
        contrastlist = np.zeros(np.shape(phaselist))
        exlist = np.zeros(np.shape(phaselist))
        
    elif dataset in group2:
        [u,contrastlist] = np.unique(featureMat_orig[:,0],return_inverse=True)
        contrastlist = np.expand_dims(contrastlist,1)      
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
        noiselist = np.zeros(np.shape(phaselist))
        
    elif dataset in group3:
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
        
    if 'oriTst1'==dataset:
       
        info['timepoint_labels'] = ['before retraining','after retraining']  
        info['noise_levels'] = [0, 0.25, 0.5]   
        info['sf_vals'] = np.logspace(np.log10(0.4), np.log10(2.2),4)
        nPhase=4   
        info['phase_vals'] = ['r%d'%pp for pp in range(nPhase)]
        nEx = 1
        info['contrast_levels'] = [0.5]
              
    elif 'oriTst2'==dataset or 'oriTst2a'==dataset:
        
        info['timepoint_labels'] = ['before retraining','after retraining']  
        info['noise_levels'] = [0]   
        info['sf_vals'] = np.logspace(np.log10(0.7061), np.log10(2.2),3)
        nPhase = 16
        info['phase_vals'] = ['r%d'%pp for pp in range(nPhase)]
        nEx = 1
        info['contrast_levels'] = [0.5]
 
    elif 'oriTst3'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.20]   
        info['sf_vals'] = np.logspace(np.log10(0.7061), np.log10(2.2),3)
        info['contrast_levels'] = [0.5]
        nPhase=1    # phase is fixed here  
        info['phase_vals'] = [0]
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
        
    elif 'oriTst4'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.20]   
        info['sf_vals'] = np.logspace(np.log10(1.25), np.log10(3.88),3)
        info['contrast_levels'] = [0.5]
        nPhase=16   # phase is random here
        info['phase_vals'] = ['r%d'%pp for pp in range(nPhase)]
        nEx = 1
        
    elif 'oriTst4a'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.20]   
        info['sf_vals'] = np.logspace(np.log10(1.25), np.log10(3.88),3)
        info['contrast_levels'] = [0.5]
        nPhase=1    # phase is fixed here  
        info['phase_vals'] = [0]
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
   
    elif 'oriTst5'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.20, 0.40, 0.60]   
        info['sf_vals'] = [2.20]         
        info['contrast_levels'] = [0.5]
        nPhase=16   # phase is random
        info['phase_vals'] = ['r%d'%pp for pp in range(nPhase)]
        nEx = 1
        
    elif 'oriTst5a'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.20, 0.40, 0.60]   
        info['sf_vals'] = [2.20]         
        info['contrast_levels'] = [0.5]
        nPhase=1    # phase is fixed here
        info['phase_vals'] = [0]
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
        
    elif 'oriTst6'==dataset:   
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0, 0.10, 0.20]   
        info['sf_vals'] = [2.20]         
        info['contrast_levels'] = [0.5]
        nPhase=1    # phase is fixed here 
        info['phase_vals'] = [0]
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
        
    elif 'oriTst6a'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.01, 0.10, 0.20]   
        info['sf_vals'] = [2.20]       
        info['contrast_levels'] = [0.5]
        nPhase=1    # phase is fixed here 
        info['phase_vals'] = [0]
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
               
    elif 'oriTst7'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.01]   
        info['sf_vals'] = [2.20] 
        info['contrast_levels'] = [0.5]
        nPhase=48
        nEx = 1
        info['phase_vals'] = np.linspace(0,180,nPhase+1)
        info['phase_vals'] = info['phase_vals'][0:nPhase]  
   
    elif 'oriTst7a'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.01]   
        info['sf_vals'] = [2.20] 
        info['contrast_levels'] = [0.5]
        nPhase=48
        nEx = 1
        info['phase_vals'] = np.linspace(0,360,nPhase+1)
        info['phase_vals'] = info['phase_vals'][0:nPhase]     
            
    elif 'oriTst8'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [1, 10, 20]   
        info['sf_vals'] = [2.20] 
        info['contrast_levels'] = [1]
        nPhase=1    # phase is not a variable here
        info['phase_vals'] = ['none']
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
    
    elif 'oriTst8a'==dataset:
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [500, 50, 5]
        info['sf_vals'] = [2.20] 
        info['contrast_levels'] = [1]
        nPhase=1    # phase is not a variable here
        info['phase_vals'] = ['none']
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
        
    elif 'oriTst8b'==dataset:             
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.04, 1, 2]
        info['sf_vals'] = [1.21]       
        info['contrast_levels'] = [1]
        nPhase=1    # phase is not a variable here
        info['phase_vals'] = ['none']
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
                    
    elif 'oriTst9'==dataset:       
        info['timepoint_labels'] = ['before retraining']        
        info['noise_levels'] = [0.01]
        info['sf_vals'] = [2.20] 
        info['contrast_levels'] = [0.2, 0.4, 0.8]
        contrastlist=noiselist # noise is fixed here
        noiselist = np.zeros(np.shape(noiselist))
        nPhase=1
        info['phase_vals'] = [0]
        nEx = 16
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
        
    elif 'oriTst9a'==dataset:       
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0]
        info['sf_vals'] = [1.21] 
        info['contrast_levels'] = np.logspace(np.log10(0.01),np.log10(1),24)
        nPhase=2
        info['phase_vals'] = [0,180]
        nEx = 1
        
    elif 'oriTst10'==dataset:
                
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.01]
        info['sf_vals'] = np.logspace(np.log10(0.2),np.log10(4),6)
        info['contrast_levels'] = [0.8]
        nPhase=1    # phase is fixed here 
        info['phase_vals'] = [0]
        nEx = 8
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
        
    elif 'oriTst10a'==dataset:
                
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.01]
        info['sf_vals'] = np.logspace(np.log10(0.2),np.log10(4),6)
        info['contrast_levels'] = [0.8]
        nPhase=1    # phase is fixed here 
        info['phase_vals'] = ['r0']
        nEx = 8
        exlist = phaselist
        phaselist = np.zeros(np.shape(phaselist))
        
    elif 'oriTst11'==dataset:             
    
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.01]
        info['sf_vals'] = np.logspace(np.log10(0.2),np.log10(4),6)   
        info['contrast_levels'] = [0.8]
        info['phase_vals'] = [0,180]
        nPhase=2
        nEx = 4
        
    elif 'oriTst12'==dataset:             
    
        info['timepoint_labels'] = ['before retraining']
        info['noise_levels'] = [0.01, 0.10, 0.25]
        info['sf_vals'] = [0.20, 1.21]  
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
               
        for nn in range(np.size(info['noise_levels'])):

            file = os.path.join(weight_path_before, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
            w1 = np.load(file)
            
            if info['nTimePts']>1:
                file = os.path.join(weight_path_after, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
                w2 = np.load(file)
            else:
                w2 = []
                
            if nn==0:
                w1_all = w1
                w2_all = w2
            else:
                w1_all = np.concatenate((w1_all, w1), axis=0)
                w2_all = np.concatenate((w2_all, w2), axis=0)
                
        allw.append([w1_all,w2_all])
       
    #% load the predicted orientation labels from the last layer
       
    all_labs = []
    
    for nn in range(np.size(info['noise_levels'])):
    
        if 'inception' in model_str:   
            file = os.path.join(weight_path_before, 'allStimsLabsPredicted_Logits.npy')   
        else:             
            file = os.path.join(weight_path_before, 'allStimsLabsPredicted_logits.npy')  
        labs1 = np.load(file)
          
        if info['nTimePts']>1:
            file = os.path.join(weight_path_after, 'allStimsLabsPredicted_logits.npy')    
            labs2 = np.load(file)
        else:
            labs2 = []
            
        if nn==0:
            labs1_all = labs1
            labs2_all = labs2
        else:
            labs1_all = np.concatenate((labs1_all, labs1), axis=0)
            labs2_all = np.concatenate((labs2_all, labs2), axis=0)
            
        
    all_labs.append([labs1_all,labs2_all])
        
    return allw, all_labs, info

