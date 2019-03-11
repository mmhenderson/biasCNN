#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:33:38 2019

@author: mmhender
"""

# load in all my data for this version of the experiment, to be used for analysis

root = '/usr/local/serenceslab/maggie/biasCNN/';
import os

os.chdir(os.path.join(root, 'code'))

import numpy as np
import matplotlib.pyplot as plt

import scipy

from sklearn import decomposition
 
from sklearn import manifold
 
from sklearn import discriminant_analysis

import IEM

import sklearn
 
import classifiers

#import pycircstat

import scipy

noise_levels = [0, 0.2, 0.4, 0.6, 0.8]

weight_path_before = os.path.join(root, 'activations', 'nasnet_activations_short_reduced')
weight_path_after = os.path.join(root, 'activations', 'nasnet_activations_long_reduced')
#weight_path_before = os.path.join(root, 'weights', 'inception_v3_grating_orient_short')
#weight_path_after = os.path.join(root, 'weights', 'inception_v3_grating_orient_long')
#dataset_path = os.path.join(root, 'datasets', 'datasets_Grating_Orient_SF')

#layer_labels = ['Conv2d_1a_3x3', 'Conv2d_4a_3x3','Mixed_7c','logits']
timepoint_labels = ['before retraining','after retraining']

layer_labels = []
for cc in range(17):
    layer_labels.append('Cell_%d' % (cc+1))
layer_labels.append('global_pool')
layer_labels.append('logits')
#%% information about the stimuli. There are two types - first is a full field 
# sinusoidal grating (e.g. a rectangular image with the whole thing a grating)
# second is a gaussian windowed grating.
sf_vals = np.logspace(np.log10(0.2), np.log10(2),5)
stim_types = ['Full','Gaus']
nOri=180
nSF=5
nPhase=4
nType = 2

# list all the image features in a big matrix, where every row is unique.
typelist = np.expand_dims(np.repeat(np.arange(nType), nPhase*nOri*nSF), 1)
orilist=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType]))
sflist=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType]))
phaselist=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType]))

featureMat = np.concatenate((typelist,orilist,sflist,phaselist),axis=1)

assert np.array_equal(featureMat, np.unique(featureMat, axis=0))

actual_labels = orilist

xx=np.arange(0,180,1)

#%% load the data (already in reduced/PCA-d format)

allw = []

for ll in range(np.size(layer_labels)):
    
    tmp = []
    
    for nn in range(np.size(noise_levels)):
#        print(nn)
#        print(noise_levels[nn])
        file = os.path.join(weight_path_before, 'noise%.2f' % noise_levels[nn], 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w1 = np.load(file)
        
        file = os.path.join(weight_path_after, 'noise%.2f' % noise_levels[nn], 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w2 = np.load(file)
        
        tmp.append([w1,w2])
        
    allw.append(tmp)
#    allw.append([w1])
    
nLayers = len(allw)
nTimepts = len(allw[0][0])
nNoiseLevels = len(allw[0])
# can change these if you want just a subset of plots made at a time
layers2plot = np.arange(0,nLayers,1)
#timepts2plot = np.arange(0,nTimepts,1)
timepts2plot = np.arange(0,1)

#%% load the predicted orientation labels from the re-trained network

num_batches = 80

all_labs = []

for nn in range(np.size(noise_levels)):

    file = os.path.join(weight_path_before, 'noise%.2f' %(noise_levels[nn]), 'allStimsLabsPredicted_Cell_1.npy')    
    labs1 = np.load(file)
     
    file = os.path.join(weight_path_after, 'noise%.2f' %(noise_levels[nn]), 'allStimsLabsPredicted_Cell_1.npy')    
    labs2 = np.load(file)
    
    all_labs.append([labs1,labs2])
 