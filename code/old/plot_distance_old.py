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

from copy import deepcopy
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
#%% get the data ready to go...then can run any below cells independently.

#model_str = 'inception_oriTst5a'
#model_name_2plot = 'Inception-V3'


#model_str = 'nasnet_oriTst5a'
#model_name_2plot = 'NASnet'

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

#%% plot discriminability across 0-360 space, overlay spatial frequencies
    
assert 'oriTst11' in model_str
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf2plot = [1] # spat freq
tt = 0

plt.figure()
xx=1
pp=1

legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    for sf in range(np.size(sf2plot)):
            
        myinds_bool = np.all([sflist==sf2plot[sf],   typelist==tt, noiselist==nn], axis=0)  
        un,ia = np.unique(orilist_adj, return_inverse=True)
#        assert np.all(np.expand_dims(ia,1)==orilist_adj)
 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
        
            # find gratings at the orientations of interest
            inds_left = np.where(np.logical_and(orilist_adj==np.mod(un[ii], 360), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist_adj==np.mod(un[ii]+1, 360), myinds_bool))[0]
            assert(np.shape(inds_left)==np.shape(inds_right))
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])

            disc[ii] = dist
    
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(np.mod(un+0.5, 360),disc)
#        for ll in np.arange(0,360,45):
#            plt.plot([ll,ll],plt.gca().get_ylim(),'k')


    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,360,45))
    else:
        plt.xticks([])
   
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\n%s, noise=%.2f' % (model_str,stim_types[tt],noise_levels[nn]))
             
    xx=xx+1
    
 
#%%  Plot the standardized euclidean distance, within spatial freq and noise level
# Overlay noise levels
 
plt.close('all')
noise2plot= [0,1,2]
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 0 # spat freq
tt = 0

plt.figure()
xx=1

legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    for nn in range(np.size(noise2plot)):
            
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==noise2plot[nn]], axis=0)  
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
        
            # find gratings at the orientations of interest
            inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])

            disc[ii] = dist
    
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(np.mod(un+0.5, 180),disc)


    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\n%s, SF=%.2f' % (model_str,stim_types[tt],sf_vals[sf]))
             
    xx=xx+1

#%%  Plot the non-standardized euclidean distance, within spatial freq
# Overlay contrast levels
assert 'oriTst9a' in model_str
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')
contrast2plot = np.arange(0,24,1)
#contrast2plot = [20]
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 0 # spat freq
tt = 0

plt.figure()
xx=1

legendlabs = ['contrast=%.2f'%(contrast_levels[cc]) for cc in contrast2plot]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    for cc in range(np.size(contrast2plot)):
            
        myinds_bool = np.all([contrastlist==contrast2plot[cc]], axis=0)  
        un,ia = np.unique(orilist_adj, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist_adj)
 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
        
            # find gratings at the orientations of interest
            inds_left = np.where(np.logical_and(orilist_adj==np.mod(un[ii], 360), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist_adj==np.mod(un[ii]+1, 360), myinds_bool))[0]
            
            dist = classifiers.get_euc_dist(w[inds_right,:],w[inds_left,:])

            disc[ii] = dist
    
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(np.mod(un+0.5, 360),disc)


    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\n%s, SF=%.2f' % (model_str,stim_types[tt],sf_vals[sf]))
             
    xx=xx+1
    
#%%  Plot the non-standardized euclidean distance, within spatial freq and noise level
# Overlay noise levels
 
plt.close('all')
noise2plot=[0,1,2]
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 0 # spat freq
tt = 0

plt.figure()
xx=1

legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    for nn in range(np.size(noise2plot)):
            
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==noise2plot[nn]], axis=0)  
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
        
            # find gratings at the orientations of interest
            inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            dist = classifiers.get_euc_dist(w[inds_right,:],w[inds_left,:])

            disc[ii] = dist
    
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(np.mod(un+0.5, 180),disc)


    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
   
    plt.suptitle('%s\nDiscriminability (euc distance) between pairs of orientations\n%s, SF=%.2f' % (model_str,stim_types[tt],sf_vals[sf]))
             
    xx=xx+1

#%%  Plot the standardized euclidean distance, within spatial freq and noise level
# Overlay spatial frequencies 
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf2plot = [0,1,2,3,4,5] # spat freq
tt = 0

plt.figure()
xx=1
pp=1

legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    for sf in range(np.size(sf2plot)):
            
        myinds_bool = np.all([sflist==sf2plot[sf],   typelist==tt, noiselist==nn], axis=0)  
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
        
            # find gratings at the orientations of interest
            inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
            assert(np.shape(inds_left)==np.shape(inds_right))
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])

            disc[ii] = dist
    
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(np.mod(un+0.5, 180),disc)


    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\n%s, noise=%.2f' % (model_str,stim_types[tt],noise_levels[nn]))
             
    xx=xx+1


#%% Plot the standardized euclidean distance across all SF
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

tt = 0

plt.figure()
xx=1
steps = [1]
legendlabs = ['%d deg apart'%(steps[ss]) for ss in range(np.size(steps))]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([typelist==tt, noiselist==nn], axis=0)
    
    un,ia = np.unique(orilist, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==orilist)
    
     
    disc = np.zeros([np.shape(un)[0],np.size(steps)])

    for ii in np.arange(0,np.size(un)):
        for ss in range(np.size(steps)):
                
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(orilist==np.mod(un[ii]-(np.floor(steps[ss]/2)), 180), myinds_bool))[0]   
            inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+(np.ceil(steps[ss]/2)), 180), myinds_bool))[0]
            
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii,ss] = dist

    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for ss in range(np.size(steps)):
        plt.plot(np.mod(un+0.5, 180),disc[:,ss])


    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
   
    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, all SF, noise=%.2f' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1


#%% plot anisotropy versus layers: plot each spatial frequency and noise level
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
tt  = 0

#sf2plot = [0]
#noise2plot = [0,1,2]
sf2plot = [4]
noise2plot = [0]

for sf in sf2plot:
        
    for nn in noise2plot:
            
        aniso_vals = np.zeros([4,np.size(layers2plot)])
        
        for ww1 in layers2plot:
            
           
            w = allw[ww1][ww2]
            
            myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
            
            un,ia = np.unique(orilist, return_inverse=True)
            assert np.all(np.expand_dims(ia,1)==orilist)
            
             
            disc = np.zeros([np.shape(un)[0],1])
        
            for ii in np.arange(0,np.size(un)):
                for ss in range(np.size(steps)):
                        
                    # find all gratings at the positions of interest
                    inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                    inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                    
                    dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
         
                    disc[ii] = dist
          
            # take the bins of interest to get amplitude
            b = np.arange(22.5,180,45)
            baseline_discrim = [];
            for ii in range(np.size(b)):        
                inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
                baseline_discrim.append(disc[inds])
                
            a = np.arange(0,180,45)
            
            for ii in range(np.size(a)):       
                
                inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(180+a[ii]))<1))[0]
                assert(np.size(inds)==2)
                aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
          
            
        plt.figure()
        ylims = [-1,1]
        xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]
        h = []
        for aa in range(4):
            h.append(plt.plot(layers2plot,aniso_vals[aa,]))
        
        plt.legend(['0','45','90','135'])
        plt.plot(xlims, [0,0], 'k')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
        plt.title('%s\nDiscriminability at each orientation rel. to baseline\nsf=%.2f, noise=%.2f' % (model_str, sf_vals[sf], noise_levels[nn]))
        plt.ylabel('Normalized Euclidean Distance difference')
        plt.xlabel('Layer number')


#%% plot Cardinal anisotropy, overlay SF
      
plt.close('all')

ww2 = 0;

#layers2plot = [0,1,3,4,6,7,8,10,11,12,14,15,16,18,19,20,21]
layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf2plot = [0,1,2] # spat freq
tt  = 0
nn=0


plt.figure()

for sf in sf2plot:
       
    aniso_vals = np.zeros([2,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
        
       
        w = allw[layers2plot[ww1]][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
#            for ss in range(np.size(steps)):
                    
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
            assert(np.shape(inds_left)==np.shape(inds_right))
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii] = dist
      
        # take the bins of interest to get amplitude
        b = np.arange(22.5,180,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,180,90)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(180+a[ii]))<1))[0]
            assert(np.size(inds)==2)
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy\nnoise=%.2f' % (model_str, noise_levels[nn]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

#%% plot Cardinal anisotropy in 0-360 space, overlay Contrast levels
      
assert 'oriTst9a' in model_str

plt.close('all')

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

ww2 = 0;
sf = 0
layers2plot = np.arange(0,nLayers,1)
tt  = 0
nn=0
contrast2plot =np.arange(3, 24, 4)

plt.figure()

for cc in contrast2plot:
       
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
        
       
        w = allw[layers2plot[ww1]][ww2]
        
        myinds_bool = np.all([contrastlist==cc, sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist_adj, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist_adj)
                 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
#            for ss in range(np.size(steps)):
                    
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(orilist_adj==np.mod(un[ii], 360), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist_adj==np.mod(un[ii]+1, 360), myinds_bool))[0]
            
            dist = classifiers.get_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii] = dist
      
        # take the bins of interest to get amplitude
        b = np.arange(22.5,360,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,360,90)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(360+a[ii]))<1))[0]
            assert(np.size(inds)==2)
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(['contrast=%.3f'%contrast_levels[cc] for cc in contrast2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy\nnoise=%.2f' % (model_str, noise_levels[nn]))
plt.ylabel('Euclidean Distance difference')
plt.xlabel('Layer number')
#%% plot Mean Discriminability in 0-360 space, overlay Contrast levels
      
assert 'oriTst9a' in model_str

plt.close('all')

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

ww2 = 0;
sf = 0
layers2plot = np.arange(0,nLayers,1)
tt  = 0
nn=0
contrast2plot =np.arange(3, 24, 4)

plt.figure()

for cc in contrast2plot:
       
    mean_disc_vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in range(np.size(layers2plot)):
        
       
        w = allw[layers2plot[ww1]][ww2]
        
        myinds_bool = np.all([contrastlist==cc, sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist_adj, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist_adj)
                 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
#            for ss in range(np.size(steps)):
                    
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(orilist_adj==np.mod(un[ii], 360), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist_adj==np.mod(un[ii]+1, 360), myinds_bool))[0]
            
            dist = classifiers.get_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii] = dist
      
        mean_disc_vals[ww1] = np.mean(disc)
        
   
    plt.plot(np.arange(0,np.size(layers2plot),1),mean_disc_vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(['contrast=%.3f'%contrast_levels[cc] for cc in contrast2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nMean Discriminability\nnoise=%.2f' % (model_str, noise_levels[nn]))
plt.ylabel('Euc. Distance.')
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

#%% plot Cardinal anisotropy in 0-360 space, overlay SF
      
assert 'oriTst11' in model_str

plt.close('all')

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

ww2 = 0;

#layers2plot = [0,1,3,4,6,7,8,10,11,12,14,15,16,18,19,20,21]
layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf2plot = [0,1,2,3,4,5] # spat freq
tt  = 0
nn=0


plt.figure()

for sf in sf2plot:
       
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
        
       
        w = allw[layers2plot[ww1]][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist_adj, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist_adj)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
#            for ss in range(np.size(steps)):
                    
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(orilist_adj==np.mod(un[ii], 360), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist_adj==np.mod(un[ii]+1, 360), myinds_bool))[0]
            
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii] = dist
      
        # take the bins of interest to get amplitude
        b = np.arange(22.5,360,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,360,90)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(360+a[ii]))<1))[0]
            assert(np.size(inds)==2)
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy\nnoise=%.2f' % (model_str, noise_levels[nn]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

#%% plot Mean Discrim in 0-360 space, overlay SF
      
assert 'oriTst11' in model_str

plt.close('all')

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

ww2 = 0;

#layers2plot = [0,1,3,4,6,7,8,10,11,12,14,15,16,18,19,20,21]
layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf2plot = [0,1,2,3,4,5] # spat freq
tt  = 0
nn=0


plt.figure()

for sf in sf2plot:
       
    mean_disc_vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in range(np.size(layers2plot)):
        
       
        w = allw[layers2plot[ww1]][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist_adj, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist_adj)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
#            for ss in range(np.size(steps)):
                    
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(orilist_adj==np.mod(un[ii], 360), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist_adj==np.mod(un[ii]+1, 360), myinds_bool))[0]
            
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii] = dist
      
        mean_disc_vals[ww1] = np.mean(disc)
        
#    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),mean_disc_vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nMean Discriminability \nnoise=%.2f' % (model_str, noise_levels[nn]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')


#%% plot Mean discriminability, overlay SF
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf2plot = [0,1,2,3,4,5] # spat freq
tt  = 0
nn=0


plt.figure()

for sf in sf2plot:
       
    mean_disc_vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in layers2plot:
        
       
        w = allw[ww1][ww2]
        
        myinds_bool = np.all([phaselist<8, sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
            for ss in range(np.size(steps)):
                    
                # find all gratings at the positions of interest
                inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
     
                disc[ii] = dist
                
        mean_disc_vals[ww1,0] = np.mean(disc)
   
    plt.plot(layers2plot,mean_disc_vals)

ylims = [-1,1]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nMean Orientation Discriminability\nnoise=%.2f' % (model_str, noise_levels[nn]))
plt.ylabel('Normalized Euclidean Distance')
plt.xlabel('Layer number')


#%% plot Cardinal anisotropy based on the non-standardized euc distance, overlay SF
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf2plot = [0,1,2] # spat freq
tt  = 0
nn=0


plt.figure()

for sf in sf2plot:
       
    aniso_vals = np.zeros([2,np.size(layers2plot)])
    
    for ww1 in layers2plot:
        
       
        w = allw[ww1][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
            for ss in range(np.size(steps)):
                    
                # find all gratings at the positions of interest
                inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                dist = classifiers.get_euc_dist(w[inds_right,:],w[inds_left,:])
     
                disc[ii] = dist
      
        # take the bins of interest to get amplitude
        b = np.arange(22.5,180,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,180,90)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(180+a[ii]))<1))[0]
            assert(np.size(inds)==2)
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(layers2plot,vals)

ylims = [-1,1]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy\nnoise=%.2f' % (model_str, noise_levels[nn]))
plt.ylabel('Euclidean Distance difference')
plt.xlabel('Layer number')

#%% plot Cardinal anisotropy, overlay noise levels
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = 0
tt  = 0
noise2plot = [0,1,2]


plt.figure()

for nn in noise2plot:
       
    aniso_vals = np.zeros([2,np.size(layers2plot)])
    
    for ww1 in layers2plot:
        
       
        w = allw[ww1][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
            for ss in range(np.size(steps)):
                    
                # find all gratings at the positions of interest
                inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
     
                disc[ii] = dist
      
        # take the bins of interest to get amplitude
        b = np.arange(22.5,180,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,180,90)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(180+a[ii]))<1))[0]
#            assert(np.size(inds)==2)
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(layers2plot,vals)

ylims = [-1,1]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]

plt.legend(['noise=%.2f'%noise_levels[nn] for nn in noise2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy\nsf=%.2f' % (model_str, sf_vals[sf]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')


#%% plot Mean discriminability, overlay noise levels
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = sf
tt  = 0
noise2plot = [0,1,2]


plt.figure()

for nn in noise2plot:
       
    mean_discrim_vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in layers2plot:
        
       
        w = allw[ww1][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
            for ss in range(np.size(steps)):
                    
                # find all gratings at the positions of interest
                inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
     
                disc[ii] = dist
      
        mean_discrim_vals[ww1] = np.mean(disc)
       
   
    plt.plot(layers2plot,mean_discrim_vals)

ylims = [-1,1]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]

plt.legend(['noise=%.2f'%noise_levels[nn] for nn in noise2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nMean Orientation Discriminability\nsf=%.2f' % (model_str, sf_vals[sf]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

#%% plot non-standardized discriminability, overlay noise levels
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = sf
tt  = 0
noise2plot = [0,1,2]


plt.figure()

for nn in noise2plot:
       
    mean_discrim_vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in layers2plot:
        
       
        w = allw[ww1][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
            for ss in range(np.size(steps)):
                    
                # find all gratings at the positions of interest
                inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                dist = classifiers.get_euc_dist(w[inds_right,:],w[inds_left,:])
     
                disc[ii] = dist
      
        mean_discrim_vals[ww1] = np.mean(disc)
       
   
    plt.plot(layers2plot,mean_discrim_vals)

ylims = [-1,1]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]

plt.legend(['noise=%.2f'%noise_levels[nn] for nn in noise2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nMean Orientation Discriminability\nsf=%.2f' % (model_str, sf_vals[sf]))
plt.ylabel('Non-standardized Euclidean Distance difference')
plt.xlabel('Layer number')


#%% plot Cardinal anisotropy based on the NON-standardized euc distance, overlay noise levels
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = sf
tt  = 0
noise2plot = [0,1,2]


plt.figure()

for nn in noise2plot:
       
    aniso_vals = np.zeros([2,np.size(layers2plot)])
    
    for ww1 in layers2plot:
        
       
        w = allw[ww1][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
            for ss in range(np.size(steps)):
                    
                # find all gratings at the positions of interest
                inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                dist = classifiers.get_euc_dist(w[inds_right,:],w[inds_left,:])
     
                disc[ii] = dist
      
        # take the bins of interest to get amplitude
        b = np.arange(22.5,180,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,180,90)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(180+a[ii]))<1))[0]
#            assert(np.size(inds)==2)
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(layers2plot,vals)

ylims = [-1,1]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]

plt.legend(['noise=%.2f'%noise_levels[nn] for nn in noise2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy\nsf=%.2f' % (model_str, sf_vals[sf]))
plt.ylabel('Non-standardized Euclidean Distance difference')
plt.xlabel('Layer number')


#%% plot Cardinal - Oblique anisotropy, overlay SF
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf2plot = [0,1,2] # spat freq
tt  = 0
nn=0


plt.figure()

for sf in sf2plot:
       
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in layers2plot:
        
       
        w = allw[ww1][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
            for ss in range(np.size(steps)):
                    
                # find all gratings at the positions of interest
                inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
     
                disc[ii] = dist
      
        # take the bins of interest to get amplitude
        b = np.arange(22.5,180,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,180,45)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(180+a[ii]))<1))[0]
            assert(np.size(inds)==2)
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals[[0,2],:],0) - np.mean(aniso_vals[[1,3],:],0)
 
    plt.plot(layers2plot,vals)

ylims = [-1,1]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy  - oblique anisotropy\nnoise=%.2f' % (model_str, noise_levels[nn]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')


#%% plot Cardinal - Oblique anisotropy, overlay noise
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = 0
tt  = 0
noise2plot = [0,1,2]

plt.figure()

for nn in noise2plot:
       
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in layers2plot:
        
       
        w = allw[ww1][ww2]
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
        un,ia = np.unique(orilist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==orilist)
        
         
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
            for ss in range(np.size(steps)):
                    
                # find all gratings at the positions of interest
                inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
     
                disc[ii] = dist
      
        # take the bins of interest to get amplitude
        b = np.arange(22.5,180,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,180,45)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(180+a[ii]))<1))[0]
            assert(np.size(inds)==2)
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals[[0,2],:],0) - np.mean(aniso_vals[[1,3],:],0)
 
    plt.plot(layers2plot,vals)

ylims = [-1,1]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]

plt.legend(['noise=%.2f'%noise_levels[nn] for nn in noise2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy  - oblique anisotropy\nsf=%.2f' % (model_str, sf_vals[sf]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')


#%% plot anisotropy versus layers: all SF, plot each noise level
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
#sf = 0 # spat freq
tt = 0
nn = 0

aniso_vals = np.zeros([4,np.size(layers2plot)])

for ww1 in layers2plot:
    
   
    w = allw[ww1][ww2]
    
    myinds_bool = np.all([typelist==tt, noiselist==nn], axis=0)
    
    un,ia = np.unique(orilist, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==orilist)
    
     
    disc = np.zeros([np.shape(un)[0],1])

    for ii in np.arange(0,np.size(un)):
        for ss in range(np.size(steps)):
                
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(orilist==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(orilist==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii] = dist
  
    # take the bins of interest to get amplitude
    b = np.arange(22.5,180,45)
    baseline_discrim = [];
    for ii in range(np.size(b)):        
        inds = np.where(np.abs(un+0.5-b[ii])<4)[0]        
        baseline_discrim.append(disc[inds])
        
    a = np.arange(0,180,45)
    
    for ii in range(np.size(a)):       
        inds = np.where(np.logical_or(np.abs(un+0.5-a[ii])<1, np.abs(un+0.5-(180+a[ii]))<1))[0]
        assert(np.size(inds)==2)
        aniso_vals[ii,ww1] = np.mean(disc[inds]) - np.mean(baseline_discrim)
  
    
plt.figure()
ylims = [-5,15]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]
h = []
for aa in range(4):
    h.append(plt.plot(layers2plot,aniso_vals[aa,]))

plt.legend(['0','45','90','135'])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nDiscriminability at each orientation rel. to baseline\nAll SF, noise=%.2f' % (model_name_2plot, noise_levels[nn]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')
    
#%% Plot a dissimilarity matrix based on the standardized euclidean distance, within one SF
 
plt.close('all')
nn=0
ww2 = 0;
ww1 = 15;
tt = 0
sf2plot = [0,1,2]
    
for sf in sf2plot:
    
    w = allw[ww1][ww2]
        
    myinds_bool = np.all([typelist==tt, noiselist==nn, sflist==sf],axis=0)
    un,ia = np.unique(orilist, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==orilist)
    
    disc = np.zeros([180,180])
    
    for ii in np.arange(0,np.size(un)):
        
        # find all gratings with this label
        inds1 = np.where(np.logical_and(orilist==un[ii], myinds_bool))[0]    
           
        for jj in np.arange(ii+1, np.size(un)):
            
            # now all gratings with other label
            inds2 = np.where(np.logical_and(orilist==un[jj], myinds_bool))[0]    
        
            dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])
    
            disc[ii,jj]=  dist
            disc[jj,ii] = dist
      
    plt.figure()
    plt.pcolormesh(disc)
    plt.colorbar()
             
    plt.title('Standardized Euclidean distance, sf=%.2f, noise=%.2f\n%s' % (sf_vals[sf],noise_levels[nn],layer_labels[ww1]))
        
    plt.xlabel('orientation 1')
    plt.ylabel('orientation 2')
    
    plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
    plt.yticks(np.arange(0,180,45),np.arange(0,180,45))
            
#%% Plot a dissimilarity matrix based on the standardized euclidean distance, comparing different SFs
 
plt.close('all')
nn=0
ww2 = 0;
ww1 = 15
tt = 0

w = allw[ww1][ww2]
    
myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
un,ia = np.unique(orilist, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==orilist)

disc = np.zeros([180*nSF,180*nSF])

sf_tiled = np.repeat(np.arange(0,nSF,1),180)
ori_tiled = np.tile(np.arange(0,180,1),[1,nSF])

for ii1 in range(180*nSF):
    
     # find all gratings with this label
     inds1 = np.where(np.all([sflist==sf_tiled[ii1], orilist==ori_tiled[0,ii1], myinds_bool], axis=0))[0]    
             
     for ii2 in np.arange(ii1+1, 180*nSF):
        
        # now all gratings with other label
        inds2 = np.where(np.all([sflist==sf_tiled[ii2], orilist==ori_tiled[0,ii2], myinds_bool], axis=0))[0]    
  
        dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])

        disc[ii1,ii2]=  dist
     
        disc[ii2,ii1] = dist
 
plt.figure()
plt.pcolormesh(disc)
plt.colorbar()
         
plt.title('Standardized Euclidean distance, noise=%.2f\n%s' % (noise_levels[nn],layer_labels[ww1]))
    
plt.xlabel('orientation 1')
plt.ylabel('orientation 2')

plt.xticks(np.arange(0,180*nSF,45),np.tile(np.arange(0,180,45), [1,nSF])[0])
plt.yticks(np.arange(0,180*nSF,45),np.tile(np.arange(0,180,45), [1,nSF])[0])
        
for sf in range(nSF):
    plt.plot([sf*180,sf*180], [0,nSF*180],'k')
    plt.plot([0,nSF*180],[sf*180,sf*180],'k')

#%% Plot a dissimilarity matrix based on the standardized euclidean distance, across all SF
 
plt.close('all')
nn=0
ww2 = 0;
ww1 = 14
tt = 0

w = allw[ww1][ww2]
    
myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
un,ia = np.unique(orilist, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==orilist)

disc = np.zeros([180,180])

for ii in np.arange(0,np.size(un)):
    
    # find all gratings with this label
    inds1 = np.where(np.logical_and(orilist==un[ii], myinds_bool))[0]    
       
    for jj in np.arange(ii+1, np.size(un)):
        
        # now all gratings with other label
        inds2 = np.where(np.logical_and(orilist==un[jj], myinds_bool))[0]    
    
        dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])

        disc[ii,jj]=  dist
        disc[jj,ii] = dist
  
plt.figure()
plt.pcolormesh(disc)
plt.colorbar()
         
plt.title('Standardized Euclidean distance, across all SF, noise=%.2f\n%s' % (noise_levels[nn],layer_labels[ww1]))
    
plt.xlabel('orientation 1')
plt.ylabel('orientation 2')

plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
plt.yticks(np.arange(0,180,45),np.arange(0,180,45))
        