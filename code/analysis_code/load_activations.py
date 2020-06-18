#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load activations (post-pca) from a network presented with grating images.
"""

#%% Set up paths and general parameters


import os
import numpy as np

def get_info(model, dataset):
    """ get information about this model and the dataset 
    """
  
    info = dict()  
  
    #%% list all the layers of this network
    if 'nasnet'==model:
        
        layer_labels = []
        for cc in range(17):
            layer_labels.append('Cell_%d' % (cc+1))
        layer_labels.append('global_pool')
        layer_labels.append('logits')
        info['layer_labels'] = layer_labels
        info['layer_labels_full'] = layer_labels
        
    elif 'inception'==model:
        
        # list of all the endpoints in this network.
        layer_labels = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3',
         'MaxPool_3a_3x3','Conv2d_3b_1x1','Conv2d_4a_3x3', 'MaxPool_5a_3x3',
         'Mixed_5b','Mixed_5c','Mixed_5d','Mixed_6a','Mixed_6b','Mixed_6c',
         'Mixed_6d','Mixed_6e','Mixed_7a', 'Mixed_7b','Mixed_7c',
         'AuxLogits','AvgPool_1a','PreLogits','Logits','Predictions']       
        info['layer_labels'] = layer_labels
        info['layer_labels_full'] = layer_labels
        
    elif 'vgg16' in model and 'simul' not in model:
        
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
        
    elif 'vgg16_simul'==model:
        
        # list of all the endpoints in this network (these are the exact strings we need to use to load files)
         
        
        # returning a slightly different version of these names, shorter, for use in plots
        layer_labels = ['conv1_1','conv1_2','pool1',
         'conv2_1','conv2_2','pool2',
         'conv3_1','conv3_2','conv3_3','pool3',
         'conv4_1','conv4_2','conv4_3','pool4',
         'conv5_1','conv5_2','conv5_3','pool5',
         'fc6']    
        info['layer_labels'] = layer_labels
        info['layer_labels_full'] = layer_labels
    
        info['activ_dims'] = [224,224,112,112,112,56,56,56,56,28,28,28,28,14,14,14,14,7,1]
        info['output_chans'] = [64,64,64,128,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096]
    elif 'pixel'==model:
        layer_labels = ['pixel']  
        info['layer_labels'] = layer_labels   
        info['layer_labels_full'] = layer_labels
        info['activ_dims'] = [224]
        info['output_chans'] = [1]
    else:
        raise ValueError('model string not recognized')
        
    #%% list information about the images it was evaluated on       
    nSF = 6  
    nOri=180
    
    if 'Gratings' in dataset and 'PhaseVarying' not in dataset:

      nPhase=2;
      nEx = 4;      
      # list all the image features here. 
      exlist = np.expand_dims(np.repeat(np.arange(nEx), nPhase*nOri*nSF),axis=1)
      orilist=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nEx]))
      sflist=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nEx]))
      phaselist=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nEx]))      
      # make sure we have listed only unique things.
      featureMat = np.concatenate((exlist,orilist,sflist,phaselist),axis=1)
      assert np.array_equal(featureMat, np.unique(featureMat, axis=0))
           
    elif 'PhaseVarying' in dataset:
    
      nEx=1
      # this is the number of phase "pairs" - each of these pairs will be  a complementary pair, 
      # such that together it makes a complete 360 degree rotation about the center.
      nPhasePairs = 24
      info['nPhasePairs'] = nPhasePairs
      nPhase = 2
      nSF_here=1
      # list all the image features here. 
      exlist = np.expand_dims(np.repeat(np.arange(nEx), nPhasePairs*nPhase*nOri), axis=1)  
      orilist=np.expand_dims(np.transpose(np.repeat(np.arange(nOri),nPhasePairs*nPhase)),axis=1)
      phasejitterlist = np.transpose(np.tile(np.repeat(np.arange(nPhasePairs), nPhase),[1,nOri]))
      sflist=np.transpose(np.tile(np.repeat(np.arange(nSF_here),nPhase),[1,nOri*nEx*nPhasePairs]))
      info['phasejitterlist'] = phasejitterlist
      phaselist=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nPhasePairs]))
      # make sure we have listed only unique things.
      featureMat = np.concatenate((orilist,sflist,phasejitterlist,phaselist),axis=1)
      assert np.array_equal(featureMat, np.unique(featureMat, axis=0))   
        
    elif 'FiltNoise' in dataset:
      
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
    elif 'FiltImsAllSF' in dataset:
      
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

def load_fisher5(model, dataset, training_str=None, param_str=None, ckpt_num=None, part_str=None,root = '/usr/local/serenceslab/maggie/biasCNN/'):
    """ load the discriminability curve (discrim versus orientation) for the model specified.
    """
    info = get_info(model,dataset)
    
    if model=='pixel':
      save_path = os.path.join(root,'code','discrim_func',model,dataset,'Fisher_info_delta5_pixels.npy')
    else:    
      save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset,'Fisher_info_delta5_eval_at_ckpt_%s_%s.npy'%(ckpt_num,part_str))
    
    print('loading from %s\n'%save_path)
    
    discrim = np.load(save_path)
    
    return discrim, info
  
def load_fisher2(model, dataset, training_str=None, param_str=None, ckpt_num=None, part_str=None,root = '/usr/local/serenceslab/maggie/biasCNN/'):
    """ load the discriminability curve (discrim versus orientation) for the model specified.
    """
    info = get_info(model,dataset)
    
    if model=='pixel':
      save_path = os.path.join(root,'code','discrim_func',model,dataset,'Fisher_info_delta2_pixels.npy')
    else:    
      save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset,'Fisher_info_delta2_eval_at_ckpt_%s_%s.npy'%(ckpt_num,part_str))
    
    print('loading from %s\n'%save_path)
    
    discrim = np.load(save_path)
    
    return discrim, info

def load_discrim(model, dataset, training_str=None, param_str=None, ckpt_num=None, part_str=None,root = '/usr/local/serenceslab/maggie/biasCNN/'):
    """ load the discriminability curve (discrim versus orientation) for the model specified.
    """
    info = get_info(model,dataset)
    
    if model=='pixel':
      save_path = os.path.join(root,'code','discrim_func',model,dataset,'Discrim_func_pixels.npy')
    else:    
      save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset,'Discrim_func_eval_at_ckpt_%s_%s.npy'%(ckpt_num,part_str))
    
    print('loading from %s\n'%save_path)
    
    discrim = np.load(save_path)
    
    return discrim, info
  
def load_discrim_binned(model, dataset, training_str=None, param_str=None, ckpt_num=None, part_str=None,root = '/usr/local/serenceslab/maggie/biasCNN/'):
    """ load the discriminability curve (discrim versus orientation) for the model specified.
    """
    info = get_info(model,dataset)
    
    if model=='pixel':
      save_path = os.path.join(root,'code','discrim_func',model,dataset,'Discrim_func_binned_pixels.npy')
    else:    
      save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset,'Discrim_func_binned_eval_at_ckpt_%s_%s.npy'%(ckpt_num,part_str))
    
    print('loading from %s\n'%save_path)
    
    discrim = np.load(save_path)
    
    return discrim, info
  
def load_discrim_5degsteps(model, dataset, training_str=None, param_str=None, ckpt_num=None, part_str=None,root = '/usr/local/serenceslab/maggie/biasCNN/'):
    """ load the discriminability curve (discrim versus orientation) for the model specified.
    """
    info = get_info(model,dataset)
    
    if model=='pixel':
      save_path = os.path.join(root,'code','discrim_func',model,dataset,'Discrim_func_5degsteps_pixels.npy')
    else:    
      save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset,'Discrim_func_5degsteps_eval_at_ckpt_%s_%s.npy'%(ckpt_num,part_str))
    
    print('loading from %s\n'%save_path)
    
    discrim5 = np.load(save_path)
    
    return discrim5, info

def load_activ(model, dataset, training_str, param_str, ckpt_num,root = '/usr/local/serenceslab/maggie/biasCNN/'):
    """ load the activations for the model specified"""
    
    info = get_info(model,dataset)
    layer_labels = info['layer_labels_full']
    #%% define where all the activations should be found      
    weight_path = os.path.join(root, 'activations', model, training_str, param_str,  dataset, 'eval_at_ckpt-' + ckpt_num + '_reduced')
    print('loading activations from %s\n'%weight_path)

    #%% load the data (already in reduced/PCA-d format)
    
    allw = []   
    allvarexpl = []
    for ll in range(np.size(layer_labels)):

        file = os.path.join(weight_path, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        if os.path.isfile(file):
          w1 = np.load(file)
        else:
          print('missing file at %s\n'%file)
          w1 = []
        w2 = []

        allw.append([w1,w2])
        
        file = os.path.join(weight_path, 'allStimsVarExpl_%s.npy' % layer_labels[ll])
        if os.path.isfile(file):
          v1 = np.load(file)
        else:
          print('missing file at %s\n'%file)
          v1 = []
        v2 = []
        
        allvarexpl.append([v1,v2])

    return allw, allvarexpl, info


def load_activ_sep_edges(model, dataset, training_str, param_str, ckpt_num,root = '/usr/local/serenceslab/maggie/biasCNN/'):
    """ load the activations for the model specified"""
    
    info = get_info(model,dataset)
    layer_labels = info['layer_labels_full']
    layer_labels_plot = info['layer_labels']
    
    #%% define where all the activations should be found      
    weight_path = os.path.join(root, 'activations', model, training_str, param_str,  dataset, 'eval_at_ckpt-' + ckpt_num + '_reduced_sep_edges')
    print('loading activations from %s\n'%weight_path)

    #%% load the data (already in reduced/PCA-d format)
    
    allw = []   
    allvarexpl = []
    
    # list of just the layers that the center/edge units were separately analyzed for
    layer_labels_tmp = []
    
    # looping over all layers, some won't exist and we will skip them
    for ll in range(np.size(layer_labels)):

        try:
          
          file = os.path.join(weight_path, 'allStimsReducedWts_%s_center_units.npy' % layer_labels[ll])        
          w1 = np.load(file)
          file = os.path.join(weight_path, 'allStimsReducedWts_%s_edge_units.npy' % layer_labels[ll])
          w2 = np.load(file)
          w3 = []
          allw.append([w1,w2,w3])
  
          file = os.path.join(weight_path, 'allStimsVarExpl_%s_center_units.npy' % layer_labels[ll])
          v1 = np.load(file)
          file = os.path.join(weight_path, 'allStimsVarExpl_%s_edge_units.npy' % layer_labels[ll])
          v2 = np.load(file)
          v3 = []
          allvarexpl.append([v1,v2,v3])
          
          layer_labels_tmp.append(layer_labels_plot[ll])
          
        except:
#          print('file at %s is missing'%file)
          continue
       
        
    info['layer_labels'] = layer_labels_tmp
    info['nLayers'] = np.size(layer_labels_tmp)
    
    return allw, allvarexpl, info