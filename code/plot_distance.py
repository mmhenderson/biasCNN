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

model_str = 'vgg16_oriTst4'
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

    
#%% Plot the standardized euclidean distance, within a single spatial frequency and noise level
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 0 # spat freq
tt = 0

plt.figure()
xx=1

steps = [1]
legendlabs = ['%d deg apart'%(steps[ss]) for ss in range(np.size(steps))]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
     
    disc = np.zeros([np.shape(un)[0],np.size(steps)])

    for ii in np.arange(0,np.size(un)):
        for ss in range(np.size(steps)):
                
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-(np.floor(steps[ss]/2)), 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+(np.ceil(steps[ss]/2)), 180), myinds_bool))[0]
            
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
   
    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, SF=%.2f, noise=%.2f' % (stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1
   
#%%  Plot the standardized euclidean distance, within spatial freq and noise level
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
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
        
            # find gratings at the orientations of interest
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
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
   
    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, SF=%.2f' % (stim_types[tt],sf_vals[sf]))
             
    xx=xx+1

#%%  Plot the standardized euclidean distance, within spatial freq and noise level
# Overlay spatial frequencies 
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf2plot = [0,1,2] # spat freq
tt = 0

plt.figure()
xx=1

legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    for sf in range(np.size(sf2plot)):
            
        myinds_bool = np.all([sflist==sf2plot[sf],   typelist==tt, noiselist==nn], axis=0)  
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
 
        disc = np.zeros([np.shape(un)[0],1])
    
        for ii in np.arange(0,np.size(un)):
        
            # find gratings at the orientations of interest
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
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
   
    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, noise=%.2f' % (stim_types[tt],noise_levels[nn]))
             
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
    
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
     
    disc = np.zeros([np.shape(un)[0],np.size(steps)])

    for ii in np.arange(0,np.size(un)):
        for ss in range(np.size(steps)):
                
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-(np.floor(steps[ss]/2)), 180), myinds_bool))[0]   
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+(np.ceil(steps[ss]/2)), 180), myinds_bool))[0]
            
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
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros([180,180])
    
    for ii in np.arange(0,np.size(un)):
        
        # find all gratings with this label
        inds1 = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
           
        for jj in np.arange(ii+1, np.size(un)):
            
            # now all gratings with other label
            inds2 = np.where(np.logical_and(actual_labels==un[jj], myinds_bool))[0]    
        
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
un,ia = np.unique(actual_labels, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==actual_labels)

disc = np.zeros([180*nSF,180*nSF])

sf_tiled = np.repeat(np.arange(0,nSF,1),180)
ori_tiled = np.tile(np.arange(0,180,1),[1,nSF])

for ii1 in range(180*nSF):
    
     # find all gratings with this label
     inds1 = np.where(np.all([sflist==sf_tiled[ii1], orilist==ori_tiled[0,ii1], myinds_bool], axis=0))[0]    
             
     for ii2 in np.arange(ii1+1, 180*nSF):
        
        # now all gratings with other label
        inds2 = np.where(np.all([sflist==sf_tiled[ii2], actual_labels==ori_tiled[0,ii2], myinds_bool], axis=0))[0]    
  
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
un,ia = np.unique(actual_labels, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==actual_labels)

disc = np.zeros([180,180])

for ii in np.arange(0,np.size(un)):
    
    # find all gratings with this label
    inds1 = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
       
    for jj in np.arange(ii+1, np.size(un)):
        
        # now all gratings with other label
        inds2 = np.where(np.logical_and(actual_labels==un[jj], myinds_bool))[0]    
    
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
        

#%% plot anisotropy versus layers: plot each spatial frequency
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = 0 # spat freq
tt = 0
nn = 0

aniso_vals = np.zeros([4,np.size(layers2plot)])

for ww1 in layers2plot:
    
   
    w = allw[ww1][ww2]
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
     
    disc = np.zeros([np.shape(un)[0],1])

    for ii in np.arange(0,np.size(un)):
        for ss in range(np.size(steps)):
                
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
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
        inds = np.where(np.abs(un+0.5-a[ii])<1)[0]
        aniso_vals[ii,ww1] = np.mean(disc[inds]) - np.mean(baseline_discrim)
  
    
plt.figure()
ylims = [-100,150]
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]
h = []
for aa in range(4):
    h.append(plt.plot(layers2plot,aniso_vals[aa,]))

plt.legend(['0','45','90','135'])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nDiscriminability at each orientation rel. to baseline\nsf=%.2f, noise=%.2f' % (model_name_2plot, sf_vals[sf], noise_levels[nn]))
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
    
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
     
    disc = np.zeros([np.shape(un)[0],1])

    for ii in np.arange(0,np.size(un)):
        for ss in range(np.size(steps)):
                
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii], 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
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
        inds = np.where(np.abs(un+0.5-a[ii])<1)[0]
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
