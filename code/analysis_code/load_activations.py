#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get parameters for a network and information about the dataset used to evaluate.
Note this does not actually load activations, it is incorrectly named.

"""

#%% Set up paths and general parameters

import numpy as np

def get_info(model, dataset):
    """ get information about this model and the dataset 
    """
  
    info = dict()  
  
    #%% list all the layers of this network
    if 'vgg16' in model and 'simul' not in model:
        
        # list of all the endpoints in this network (these are the exact strings we need to use to load files)
        layer_labels_full = ['vgg_16_conv1_conv1_1','vgg_16_conv1_conv1_2','vgg_16_pool1',
         'vgg_16_conv2_conv2_1','vgg_16_conv2_conv2_2','vgg_16_pool2',
         'vgg_16_conv3_conv3_1','vgg_16_conv3_conv3_2','vgg_16_conv3_conv3_3','vgg_16_pool3',
         'vgg_16_conv4_conv4_1','vgg_16_conv4_conv4_2','vgg_16_conv4_conv4_3','vgg_16_pool4',
         'vgg_16_conv5_conv5_1','vgg_16_conv5_conv5_2','vgg_16_conv5_conv5_3','vgg_16_pool5',
         'vgg_16_fc6','vgg_16_fc7', 'vgg_16_fc8']  
        info['layer_labels_full'] = layer_labels_full
    
        # returning a slightly different version of these names, shorter, for use in plots
        layer_labels = ['conv1_1','conv1_2','pool1',
         'conv2_1','conv2_2','pool2',
         'conv3_1','conv3_2','conv3_3','pool3',
         'conv4_1','conv4_2','conv4_3','pool4',
         'conv5_1','conv5_2','conv5_3','pool5',
         'fc6','fc7','fc8']    
        info['layer_labels'] = layer_labels

        info['activ_dims'] = [224,224,112,112,112,56,56,56,56,28,28,28,28,14,14,14,14,7,1,1,1]
        info['output_chans'] = [64,64,64,128,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096,4096,1000]
        
    
    else:
        raise ValueError('model string not recognized')
        
    #%% list information about the images it was evaluated on       
    nSF = 6  
    nOri=180
    
    if 'FiltImsAllSF' in dataset:
      
      nPhase=1
      nEx=48
      nSF_here=1
      phaselist =np.expand_dims(np.repeat(np.arange(nPhase),nOri*nEx*nSF_here),axis=1)
      sflist =np.expand_dims(np.repeat(np.arange(nSF_here),nOri*nEx*nPhase),axis=1)
      exlist = np.expand_dims(np.repeat(np.arange(nEx), nPhase*nOri*nSF_here), axis=1)     
      orilist=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF_here*nPhase), [1,nEx]))
      # make sure we have listed only unique things.
      featureMat = np.concatenate((exlist,orilist),axis=1)      
      assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
    elif 'FiltIms' in dataset:
      nPhase=1
      if 'SF' in dataset:
        # the 6 spatial frequencies are spread across 6 datasets
        nEx=48   
        nSF_here=1
      else:
        # these 6 spatial frequencies are included in one dataset
        nEx=8
        nSF_here=nSF
       # list all the image features here. 
      exlist = np.expand_dims(np.repeat(np.arange(nEx), nPhase*nOri*nSF_here), axis=1)     
      orilist=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF_here*nPhase), [1,nEx]))
      sflist=np.transpose(np.tile(np.repeat(np.arange(nSF_here),nPhase),[1,nOri*nEx]))
      phaselist=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF_here*nEx]))
      # make sure we have listed only unique things.
      featureMat = np.concatenate((exlist,orilist,sflist,phaselist),axis=1)      
      assert np.array_equal(featureMat, np.unique(featureMat, axis=0))

    elif 'GratingsMultiPhase' in dataset:
       
      nPhase = 24;
      nEx = 2     
      # the 6 spatial frequencies are spread across 6 datasets    
      nSF_here=1
      # list all the image features here. 
      orilist=np.transpose(np.tile(np.repeat(np.arange(nOri),nPhase*nEx),[1,1]))
      phaselist=np.transpose(np.tile(np.repeat(np.arange(nPhase),nEx),[1,nOri]))
      exlist = np.transpose(np.tile(np.arange(nEx), [1,nPhase*nOri]))
      
      featureMat = np.concatenate((orilist,phaselist,exlist),axis=1)
      
      assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
      
      # just list of ones
      sflist=np.transpose(np.tile(np.arange(nSF_here),[1,nOri*nEx*nPhase]))
      
      
    else:
        raise ValueError('dataset string not recognized')

    info['nLayers'] = len(info['layer_labels'])
    info['sf_vals'] = np.logspace(np.log10(0.0125),np.log10(0.250),6)  
    info['nOri'] = nOri
    info['nEx'] = nEx
    info['nPhase'] = nPhase
    info['nSF'] = nSF
   
    info['exlist'] = exlist
    info['orilist']=orilist
    info['sflist']=sflist
    info['phaselist']=phaselist

    return info      
