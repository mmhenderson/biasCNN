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

model_str = 'vgg16_oriTst9a'
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

#%% plot discriminability across 0-360 space, overlay contrast levels
    
assert 'oriTst9a' in model_str
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')
plt.figure()
xx=1

layers2plot = np.arange(0,nLayers,1)
contrast2plot = np.arange(0,24,1)
legendlabs = ['contrast=%.2f'%(contrast_levels[cc]) for cc in contrast2plot]

for ww1 in layers2plot:

    w = allw[ww1][0]
    
    for cc in range(np.size(contrast2plot)):
        
        inds = np.where(contrastlist==contrast2plot[cc])[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
       
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
 
#%% plot Cardinal anisotropy in 0-360 space, overlay contrast
assert 'oriTst9a' in model_str
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
contrast2plot = np.arange(0,24,1)
legendlabs = ['contrast=%.2f'%(contrast_levels[cc]) for cc in contrast2plot]

b = np.arange(22.5,360,45)
a = np.arange(0,360,90)
bin_size = 6
      
for cc in contrast2plot:
    
    inds1 = np.where(contrastlist==contrast2plot[cc])[0]
        
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
       
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0] 
#            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(360+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(['contrast=%.2f'%contrast_levels[cc] for cc in contrast2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

#%% plot Mean Discrim in 0-360 space, overlay contrast
assert 'oriTst9a' in model_str
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
contrast2plot = np.arange(0,24,1)
legendlabs = ['contrast=%.2f'%(contrast_levels[cc]) for cc in contrast2plot]

for cc in contrast2plot:
    
    inds1 = np.where(contrastlist==contrast2plot[cc])[0]
        
    vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in range(np.size(layers2plot)):
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
     
        vals[ww1] = np.mean(disc)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

#ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(['contrast=%.2f'%contrast_levels[cc] for cc in contrast2plot])
plt.title('%s\nMean Discriminability' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

#%% plot contrast response function for several layers
      
assert 'oriTst9a' in model_str

plt.close('all')

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

ww2 = 0;
sf = 0
#layers2plot = np.arange(0,nLayers,1)
layers2plot = np.arange(0, nLayers, 4)
tt  = 0
nn=0
contrast2plot =np.arange(0,24,1)

plt.figure()

for ww1 in range (np.size(layers2plot)):
    
    w = allw[layers2plot[ww1]][ww2]  

    cr_func = np.zeros([np.size(contrast2plot),1])      
    
    for cc in contrast2plot:
       
        myinds_bool = np.all([contrastlist==cc, sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        cr_func[cc] = np.mean(w[np.where(myinds_bool)[0],:])
        
#        un,ia = np.unique(orilist_adj, return_inverse=True)
#        assert np.all(np.expand_dims(ia,1)==orilist_adj)
                 
#        disc = np.zeros([np.shape(un)[0],1])
#    
#        for ii in np.arange(0,np.size(un)):
##            for ss in range(np.size(steps)):
#                    
#            # find all gratings at the positions of interest
#            inds_left = np.where(np.logical_and(orilist_adj==np.mod(un[ii], 360), myinds_bool))[0]        
#            inds_right = np.where(np.logical_and(orilist_adj==np.mod(un[ii]+1, 360), myinds_bool))[0]
#            
#            dist = classifiers.get_euc_dist(w[inds_right,:],w[inds_left,:])
# 
#            disc[ii] = dist
      
#        cr_func[cc] = np.mean(disc)
        
   
    plt.plot(contrast_levels[contrast2plot],cr_func)

#ylims = [-1,1]
#xlims = [-1, np.size(layers2plot)+2]

plt.legend(['layer %d'%ww1 for ww1 in layers2plot])
#plt.plot(xlims, [0,0], 'k')
#plt.xlim(xlims)
#plt.ylim(ylims)
#plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nContrast response function (mean response)' % (model_str))
plt.ylabel('Average response')
plt.xlabel('Stim. contrast')
