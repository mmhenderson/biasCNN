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
    
    if model_str == 'nasnet_oriTst0':
        allw, all_labs, info = load_activ_nasnet_oriTst0()
    elif model_str == 'nasnet_oriTst1':
        allw, all_labs, info = load_activ_nasnet_oriTst1()
    elif model_str == 'inception_oriTst0':
        allw, all_labs, info = load_activ_inception_oriTst0()
    elif model_str == 'inception_oriTst1':
        allw, all_labs, info = load_activ_inception_oriTst1()
    elif model_str == 'vgg16_oriTst1':
        allw, all_labs, info = load_activ_vgg16_oriTst1()
    else:
        raise ValueError('model string not recognized')
        
    
    return allw, all_labs, info

def load_activ_nasnet_oriTst0():
    """ load in all my data for this version of the experiment """
    
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
    
    #% information about the stimuli. 
         
    info['noise_levels'] = [0, 0.2, 0.4, 0.6, 0.8]    
    info['sf_vals'] = np.logspace(np.log10(0.2), np.log10(2),5)
    info['stim_types'] = ['Full','Gaus']
    
    nOri=180
    nPhase=4
    nSF=np.size(info['sf_vals'])   
    nType = np.size(info['stim_types'])
    nNoiseLevels = np.size(info['noise_levels'])
    
    info['nPhase'] = nPhase
    info['nSF'] = nSF
    info['nType'] = nType
    info['nNoiseLevels'] = nNoiseLevels
    info['nLayers'] = np.size(layer_labels)
    info['nTimePts'] = np.size(info['timepoint_labels'])
    
    # list all the image features in a big matrix, where every row is unique.
    info['noiselist'] = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
    info['typelist'] = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
    info['orilist']=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
    info['sflist']=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
    info['phaselist']=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))
    
    featureMat = np.concatenate((info['noiselist'],info['typelist'],info['orilist'],info['sflist'],info['phaselist']),axis=1)
    
    assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
          
    info['xx']=np.arange(0,180,1)
        
    #% load the data (already in reduced/PCA-d format)
    
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
       
    #% load the predicted orientation labels from the last layer
       
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

def load_activ_nasnet_oriTst1():
    """ load in all my data for this version of the experiment """
    
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
    
    #% information about the stimuli. 
         
    info['noise_levels'] = [0, 0.25, 0.5]   
    info['sf_vals'] = np.logspace(np.log10(0.4), np.log10(2.2),4)
    info['stim_types'] = ['Gaus']
    
    nOri=180
    nPhase=4
    nSF=np.size(info['sf_vals'])   
    nType = np.size(info['stim_types'])
    nNoiseLevels = np.size(info['noise_levels'])
    
    info['nPhase'] = nPhase
    info['nSF'] = nSF
    info['nType'] = nType
    info['nNoiseLevels'] = nNoiseLevels
    info['nLayers'] = np.size(layer_labels)
    info['nTimePts'] = np.size(info['timepoint_labels'])
    
    # list all the image features in a big matrix, where every row is unique.
    info['noiselist'] = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
    info['typelist'] = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
    info['orilist']=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
    info['sflist']=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
    info['phaselist']=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))
    
    featureMat = np.concatenate((info['noiselist'],info['typelist'],info['orilist'],info['sflist'],info['phaselist']),axis=1)
    
    assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
          
    info['xx']=np.arange(0,180,1)
        
    #% load the data (already in reduced/PCA-d format)
    
    allw = []
    
    for ll in range(np.size(layer_labels)):
        
        file = os.path.join(weight_path_before, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w1 = np.load(file)
        
        file = os.path.join(weight_path_after, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w2 = np.load(file)
                   
        allw.append([w1,w2])

    #% load the predicted orientation labels from the last layer
  
    all_labs = []
    
    file = os.path.join(weight_path_before, 'allStimsLabsPredicted_Cell_1.npy')    
    labs1 = np.load(file)
     
    file = os.path.join(weight_path_after,'allStimsLabsPredicted_Cell_1.npy')    
    labs2 = np.load(file)
    
    all_labs.append([labs1,labs2])
     
    
    return allw, all_labs, info

#%%

def load_activ_inception_oriTst0():
    """ load in all my data for this version of the experiment """
    
    info = dict()
   
    info['timepoint_labels'] = ['before retraining','after retraining']
    
    weight_path_before = os.path.join(root, 'activations', 'inception_activations_short_reduced')
    weight_path_after = os.path.join(root, 'activations', 'inception_activations_long_reduced')
   
    # list of all the endpoints in this network.
    info['layer_labels'] = ['Conv2d_1a_3x3',
     'Conv2d_2a_3x3',
     'Conv2d_2b_3x3',
     'MaxPool_3a_3x3',
     'Conv2d_3b_1x1',
     'Conv2d_4a_3x3',
     'MaxPool_5a_3x3',
     'Mixed_5b',
     'Mixed_5c',
     'Mixed_5d',
     'Mixed_6a',
     'Mixed_6b',
     'Mixed_6c',
     'Mixed_6d',
     'Mixed_6e',
     'Mixed_7a',
     'Mixed_7b',
     'Mixed_7c',
     'AuxLogits',
     'AvgPool_1a',
     'PreLogits',
     'Logits',
     'Predictions']
    
    #% information about the stimuli. 
         
    info['noise_levels'] = [0, 0.2, 0.4, 0.6, 0.8]    
    info['sf_vals'] = np.logspace(np.log10(0.2), np.log10(2),5)
    info['stim_types'] = ['Full','Gaus']
    
    nOri=180
    nPhase=4
    nSF=np.size(info['sf_vals'])   
    nType = np.size(info['stim_types'])
    nNoiseLevels = np.size(info['noise_levels'])
    
    info['nPhase'] = nPhase
    info['nSF'] = nSF
    info['nType'] = nType
    info['nNoiseLevels'] = nNoiseLevels
    info['nLayers'] = np.size(info['layer_labels'])
    info['nTimePts'] = np.size(info['timepoint_labels'])
    
    # list all the image features in a big matrix, where every row is unique.
    info['noiselist'] = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
    info['typelist'] = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
    info['orilist']=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
    info['sflist']=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
    info['phaselist']=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))
    
    featureMat = np.concatenate((info['noiselist'],info['typelist'],info['orilist'],info['sflist'],info['phaselist']),axis=1)
    
    assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
         
    info['xx']=np.arange(0,180,1)
    
    #% load the data (already in reduced/PCA-d format)
    
    allw = []
    
    for ll in range(np.size(info['layer_labels'])):
       
        for nn in range(np.size(info['noise_levels'])):
    #        print(nn)
    #        print(noise_levels[nn])
            file = os.path.join(weight_path_before, 'noise%.2f' % info['noise_levels'][nn], 'allStimsReducedWts_%s.npy' % info['layer_labels'][ll])
            w1 = np.load(file)
            
            file = os.path.join(weight_path_after, 'noise%.2f' % info['noise_levels'][nn], 'allStimsReducedWts_%s.npy' % info['layer_labels'][ll])
            w2 = np.load(file)
                      
            if nn==0:
                w1_all = w1
                w2_all = w2
            else:
                w1_all = np.concatenate((w1_all, w1), axis=0)
                w2_all = np.concatenate((w2_all, w2), axis=0)
                
        allw.append([w1_all,w2_all])
  
    #% load the predicted orientation labels from the re-trained network
    
    all_labs = []
    
    for nn in range(np.size(info['noise_levels'])):
    
        file = os.path.join(weight_path_before, 'noise%.2f' %(info['noise_levels'][nn]), 'allStimsLabsPredicted_Logits.npy')    
        labs1 = np.load(file)
         
        file = os.path.join(weight_path_after, 'noise%.2f' %(info['noise_levels'][nn]), 'allStimsLabsPredicted_Logits.npy')    
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
def load_activ_inception_oriTst1():
    """ load in all my data for this version of the experiment """
    
    info = dict()
   
    info['timepoint_labels'] = ['before retraining','after retraining']
    
    weight_path_before = os.path.join(root, 'activations', 'inception_oriTst1_short_reduced')
    weight_path_after = os.path.join(root, 'activations', 'inception_oriTst1_long_reduced')
   
    # list of all the endpoints in this network.
    info['layer_labels'] = ['Conv2d_1a_3x3',
     'Conv2d_2a_3x3',
     'Conv2d_2b_3x3',
     'MaxPool_3a_3x3',
     'Conv2d_3b_1x1',
     'Conv2d_4a_3x3',
     'MaxPool_5a_3x3',
     'Mixed_5b',
     'Mixed_5c',
     'Mixed_5d',
     'Mixed_6a',
     'Mixed_6b',
     'Mixed_6c',
     'Mixed_6d',
     'Mixed_6e',
     'Mixed_7a',
     'Mixed_7b',
     'Mixed_7c',
     'AuxLogits',
     'AvgPool_1a',
     'PreLogits',
     'Logits',
     'Predictions']
    layer_labels = info['layer_labels']
    
    #% information about the stimuli. 
         
    info['noise_levels'] = [0, 0.25, 0.5]   
    info['sf_vals'] = np.logspace(np.log10(0.4), np.log10(2.2),4)
    info['stim_types'] = ['Gaus']
    
    nOri=180
    nPhase=4
    nSF=np.size(info['sf_vals'])   
    nType = np.size(info['stim_types'])
    nNoiseLevels = np.size(info['noise_levels'])
    
    info['nPhase'] = nPhase
    info['nSF'] = nSF
    info['nType'] = nType
    info['nNoiseLevels'] = nNoiseLevels
    info['nLayers'] = np.size(layer_labels)
    info['nTimePts'] = np.size(info['timepoint_labels'])
    
    # list all the image features in a big matrix, where every row is unique.
    info['noiselist'] = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
    info['typelist'] = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
    info['orilist']=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
    info['sflist']=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
    info['phaselist']=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))
    
    featureMat = np.concatenate((info['noiselist'],info['typelist'],info['orilist'],info['sflist'],info['phaselist']),axis=1)
    
    assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
          
    info['xx']=np.arange(0,180,1)
        
    #% load the data (already in reduced/PCA-d format)
   
    allw = []
    
    for ll in range(np.size(layer_labels)):
        
        file = os.path.join(weight_path_before, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w1 = np.load(file)
        
        file = os.path.join(weight_path_after, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w2 = np.load(file)
                   
        allw.append([w1,w2])

    #% load the predicted orientation labels from the last layer
  
    all_labs = []
    
    file = os.path.join(weight_path_before, 'allStimsLabsPredicted_Logits.npy')    
    labs1 = np.load(file)
     
    file = os.path.join(weight_path_after,'allStimsLabsPredicted_Logits.npy')    
    labs2 = np.load(file)
    
    all_labs.append([labs1,labs2])
     
    
    return allw, all_labs, info

#%%
    
def load_activ_vgg16_oriTst1():
    """ load in all my data for this version of the experiment """
    
    info = dict()
    
    weight_path_before = os.path.join(root, 'activations', 'vgg16_oriTst1_short_reduced')
    weight_path_after = os.path.join(root, 'activations', 'vgg16_oriTst1_long_reduced')
    
    info['timepoint_labels'] = ['before retraining','after retraining']
    
    
    # list of all the endpoints in this network (these are the exact strings we need to use to load files)
    layer_labels = ['conv1_conv1_1','conv1_conv1_2','pool1',
     'conv2_conv2_1','conv2_conv2_2','pool2',
     'conv3_conv3_1','conv3_conv3_2','conv3_conv3_3','pool3',
     'conv4_conv4_1','conv4_conv4_2','conv4_conv4_3','pool4',
     'conv5_conv5_1','conv5_conv5_2','conv5_conv5_3','pool5',
     'fc6',
     'fc7',
     'fc8',
     'logits']
    
    nLayers = len(layer_labels)
    for nn in range(nLayers-1):
        layer_labels[nn] = 'vgg_16_' + layer_labels[nn]
    
    # returning a slightly different version of these names, shorter, for use in plots
    layer_labels_plot = ['conv1_1','conv1_2','pool1',
     'conv2_1','conv2_2','pool2',
     'conv3_1','conv3_2','conv3_3','pool3',
     'conv4_1','conv4_2','conv4_3','pool4',
     'conv5_1','conv5_2','conv5_3','pool5',
     'fc6',
     'fc7',
     'fc8',
     'logits']

    info['layer_labels'] = layer_labels_plot
    
    #% information about the stimuli. 
         
    info['noise_levels'] = [0, 0.25, 0.5]   
    info['sf_vals'] = np.logspace(np.log10(0.4), np.log10(2.2),4)
    info['stim_types'] = ['Gaus']
    
    nOri=180
    nPhase=4
    nSF=np.size(info['sf_vals'])   
    nType = np.size(info['stim_types'])
    nNoiseLevels = np.size(info['noise_levels'])
    
    info['nPhase'] = nPhase
    info['nSF'] = nSF
    info['nType'] = nType
    info['nNoiseLevels'] = nNoiseLevels
    info['nLayers'] = np.size(layer_labels)
    info['nTimePts'] = np.size(info['timepoint_labels'])
    
    # list all the image features in a big matrix, where every row is unique.
    info['noiselist'] = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
    info['typelist'] = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
    info['orilist']=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
    info['sflist']=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
    info['phaselist']=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))
    
    featureMat = np.concatenate((info['noiselist'],info['typelist'],info['orilist'],info['sflist'],info['phaselist']),axis=1)
    
    assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
          
    info['xx']=np.arange(0,180,1)
        
    #% load the data (already in reduced/PCA-d format)
    
    allw = []
    
    for ll in range(np.size(layer_labels)):
        
        file = os.path.join(weight_path_before, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w1 = np.load(file)
        
        file = os.path.join(weight_path_after, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w2 = np.load(file)
                   
        allw.append([w1,w2])

    #% load the predicted orientation labels from the last layer
  
    all_labs = []
    
    file = os.path.join(weight_path_before, 'allStimsLabsPredicted_logits.npy')    
    labs1 = np.load(file)
     
    file = os.path.join(weight_path_after,'allStimsLabsPredicted_logits.npy')    
    labs2 = np.load(file)
    
    all_labs.append([labs1,labs2])
   
    
    return allw, all_labs, info
