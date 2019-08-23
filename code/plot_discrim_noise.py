#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt

import classifiers    

import numpy as np

from copy import deepcopy

#%% get the data ready to go...then can run any below cells independently.


model_str = 'vgg16_oriTst5a'
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
exlist = info['exlist']
contrastlist = info['contrastlist']

layer_labels = info['layer_labels']
sf_vals = info['sf_vals']
noise_levels = info['noise_levels']
timepoint_labels = info['timepoint_labels']
stim_types = info['stim_types']
phase_vals = info['phase_vals']
contrast_levels = info['contrast_levels']

nLayers = info['nLayers']
nPhase = info['nPhase']
nSF = info['nSF']
nType = info['nType']
nTimePts = info['nTimePts']
nNoiseLevels = info['nNoiseLevels']
nEx = info['nEx']
nContrastLevels = info['nContrastLevels']

#%%  Plot the standardized euclidean distance, in 0-180 space
# Overlay noise levels
plt.close('all')
plt.figure()
xx=1

layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

for ww1 in layers2plot:

    w = allw[ww1][0]
    
    for nn in range(np.size(noise2plot)):
        
        inds = np.where(noiselist==noise2plot[nn])[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist[inds])
       
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(ori_axis,disc)

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,360,45))
    else:
        plt.xticks([])
   
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations' % (model_str))
             
    xx=xx+1



#%% plot Cardinal anisotropy, overlay Noise levels
      
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

b = np.arange(22.5,180,45)
a = np.arange(0,180,90)
bin_size = 4

for nn in noise2plot:
    
    inds1 = np.where(noiselist==noise2plot[nn])[0]
           
    aniso_vals = np.zeros([2,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist[inds1])
       
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(180+b[ii]))<bin_size/2))[0] 
#            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(180+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(legendlabs)
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

#%% plot Mean discrim overlay noise levels
      
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

      
for nn in noise2plot:
    
    inds1 = np.where(noiselist==noise2plot[nn])[0]
             
    vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in range(np.size(layers2plot)):
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist[inds1])
           
        vals[ww1] = np.mean(disc)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(legendlabs)
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nMean Orientation Discriminability' % (model_str))
plt.ylabel('Normalized Euclidean Distance')
plt.xlabel('Layer number')
