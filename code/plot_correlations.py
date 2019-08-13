#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt
import scipy
import scipy.stats
   
import classifiers    

import numpy as np
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
#%% get the data ready to go...then can run any below cells independently.

#model_str = 'inception_oriTst1'
#model_name_2plot = 'Inception-V3'

#model_str = 'nasnet_oriTst1'
#model_name_2plot = 'NASnet'

model_str = 'vgg16_oriTst2'
model_name_2plot = 'VGG-16'

root = '/usr/local/serenceslab/maggie/biasCNN/';

import os
os.chdir(os.path.join(root, 'code'))
figfolder = os.path.join(root, 'figures')

import load_activations
#allw, all_labs, info = load_activations.load_activ_nasnet_oriTst0()
allw, all_labs, info = load_activations.load_activ(model_str)

# extract some fields that will help us process the data
orilist = info['orilist']
phaselist=  info['phaselist']
sflist = info['sflist']
typelist = info['typelist']
noiselist = info['noiselist']

layer_labels = info['layer_labels']
sf_vals = info['sf_vals']
noise_levels = info['noise_levels']
timepoint_labels = info['timepoint_labels']
stim_types = info['stim_types']

nLayers = info['nLayers']
nPhase = info['nPhase']
nSF = info['nSF']
nType = info['nType']
nTimePts = info['nTimePts']
nNoiseLevels = info['nNoiseLevels']

actual_labels = orilist

#%% Plot the standardized euclidean distance, within spatial frequency and noise level
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 2 # spat freq
tt = 0

plt.figure()
xx=1

steps = np.arange(0,5,1)
#steps = [0]
for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
#    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros([np.shape(un)[0],np.size(steps)])
    sd_disc = np.zeros([np.shape(un)[0],np.size(steps)])
 
    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
        
        for ss in steps:
                
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-(ss+1), 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+(ss+1), 180), myinds_bool))[0]
            
            dist_left = classifiers.get_norm_euc_dist(w[inds,:],w[inds_left,:])
            dist_right = classifiers.get_norm_euc_dist(w[inds,:],w[inds_right,:])
            
            disc[ii,ss] = np.mean([dist_left,dist_right])
            sd_disc[ii,ss] = np.std([dist_left,dist_right])
       
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    n_feat = np.shape(w)[1]
    corr = 1 - (disc/(2*n_feat))
    for ss in steps:
        plt.plot(un,corr[:,ss])
#        plt.errorbar(un,disc[:,ss], sd_disc[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('correlation coefficient')
        plt.legend(['%d deg apart'%(steps[ss]+1) for ss in range(np.size(steps))])
    else:
        plt.xticks([])
   
    plt.suptitle('Correlation between pairs of orientations\n%s, SF=%.2f, noise=%.2f' % (stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1
    
