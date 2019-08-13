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

model_str = 'vgg16_oriTst3'
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

sf = 2 # spat freq
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
noise2plot=[0]
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 2 # spat freq
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
   
#%% Plot the standardized euclidean distance between BINNED orientations, within spatial frequency and noise level
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 3 # spat freq
tt = 0

plt.figure()
xx=1

binsize = int(3)
nbins = int(180/binsize)
oribins = np.reshape(np.roll(np.arange(0,180,1),int(np.floor(binsize/2))), [nbins,binsize])
bin_centers = np.round(scipy.stats.circmean(oribins,180,0,1))
bin_list = np.zeros(np.shape(orilist))
for bb in range(nbins):
    inds = np.where(np.isin(orilist, oribins[bb,:]))[0]
    bin_list[inds,0] = bb


for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
#    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
#    un,ia = np.unique(actual_labels, return_inverse=True)
#    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros([np.shape(bin_centers)[0],2])
#    sd_disc = np.zeros([np.shape(bin_centers)[0],1])
 
    for ii in range(nbins):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(bin_list==ii, myinds_bool))[0]    
        
#        for ss in steps:
                
        # then find the positions of nearest neighbor gratings
        inds_left = np.where(np.logical_and(bin_list==np.mod(ii+1, nbins), myinds_bool))[0]        
        inds_right = np.where(np.logical_and(bin_list==np.mod(ii-1, nbins), myinds_bool))[0]
        
        dist_left = classifiers.get_norm_euc_dist(w[inds,:],w[inds_left,:])
        dist_right = classifiers.get_norm_euc_dist(w[inds,:],w[inds_right,:])
        
        disc[ii,0] = dist_left
        disc[ii,1] = dist_right
#        disc[ii] = np.mean([dist_left,dist_right])
#        sd_disc[ii] = np.std([dist_left,dist_right])
       
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

#    for ss in steps:
    plt.plot(bin_centers,np.mean(disc,1))
#    plt.plot(bin_centers,disc[:,1])
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
#        plt.legend(['dist to clockwise bin','dist to counter-clockwise bin'])
    else:
        plt.xticks([])
   
    plt.suptitle('Discriminability (std. euc distance) between %d deg bins\n%s, SF=%.2f, noise=%.2f' % (binsize, stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1

    
    
#%% Plot the standardized euclidean distance, across all SF, multiple orientation steps
 
plt.close('all')
nn=0
ww2 = 1;

layers2plot = np.arange(0,nLayers,1)

tt = 0

plt.figure()
xx=1
steps = np.arange(0,5,1)
legendlabs = ['%d deg apart'%(steps[ss]+1) for ss in range(np.size(steps))]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
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

    for ss in range(np.size(steps)):
        plt.errorbar(un,disc[:,ss], sd_disc[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])

    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, all SF, noise=%.2f' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1
    
     
#%% Plot the standardized euclidean distance, centering on the average of the 
# two compared orientations (even numbered bins), across all SF.
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

tt = 0

plt.figure()
xx=1
steps = np.arange(2,9,2)
legendlabs = ['%d deg apart'%(steps[ss]) for ss in range(np.size(steps))]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros([np.shape(un)[0],np.size(steps)])
#    sd_disc = np.zeros([np.shape(un)[0],np.size(steps)])

    for ii in np.arange(0,np.size(un)):
    
#        # first find the position of all gratings with this exact label
#        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
#        
        for ss in range(np.size(steps)):
                
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-(steps[ss]/2), 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+(steps[ss]/2), 180), myinds_bool))[0]
            
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
#            dist_right = classifiers.get_norm_euc_dist(w[inds,:],w[inds_right,:])
            
            disc[ii,ss] = dist
#            sd_disc[ii,ss] = np.std([dist_left,dist_right])

    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for ss in range(np.size(steps)):
        plt.plot(un,disc[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])

    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, all SF, noise=%.2f' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1
    
#%% Plot the standardized euclidean distance, centering on the average of the 
# two compared orientations (odd numbered bins), across all SF.
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

tt = 0
plt.figure()
xx=1
steps = np.arange(1,10,2)
legendlabs = ['%d deg apart'%(steps[ss]) for ss in range(np.size(steps))]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros([np.shape(un)[0],np.size(steps)])
#    sd_disc = np.zeros([np.shape(un)[0],np.size(steps)])

    for ii in np.arange(0,np.size(un)):
    
#        # first find the position of all gratings with this exact label
#        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
#        
        for ss in range(np.size(steps)):
                
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-(np.floor(steps[ss]/2)), 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+(np.ceil(steps[ss]/2)), 180), myinds_bool))[0]
            
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
#            dist_right = classifiers.get_norm_euc_dist(w[inds,:],w[inds_right,:])
            
            disc[ii,ss] = dist
#            sd_disc[ii,ss] = np.std([dist_left,dist_right])

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
ww1 = 15
tt = 0
sf = 1

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
    
#%% plot the average of the above over squares...  
average_mat = np.zeros([180,180,6])
xx=0;
for ii in range(3):
    for jj in np.arange(ii+1,3):
           
         average_mat[:,:,xx] = disc[ii*180:(ii+1)*180,jj*180:(jj+1)*180]
         xx=xx+1 
         
mat = np.mean(average_mat,2)

plt.figure()
plt.pcolormesh(mat)
plt.colorbar()
         
plt.title('Standardized Euclidean distance (average of all squares), noise=%.2f\n%s' % (noise_levels[nn],layer_labels[ww1]))
    
plt.xlabel('orientation 1')
plt.ylabel('orientation 2')

plt.xticks(np.arange(0,180,45))
plt.yticks(np.arange(0,180,45))
#        
#for sf in range(nSF):
#    plt.plot([sf*180,sf*180], [0,nSF*180],'k')
#    plt.plot([0,nSF*180],[sf*180,sf*180],'k')
    
#%% Plot a dissimilarity matrix based on the standardized euclidean distance, across all SF
 
plt.close('all')
nn=0
ww2 = 1;
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
        

#%% Plot the standardized euclidean distance, across a subset of SF, multiple orientation steps
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf2use = [1,2,3]

tt = 0

plt.figure()
xx=1
steps = np.arange(0,5,1)
legendlabs = ['%d deg apart'%(steps[ss]+1) for ss in range(np.size(steps))]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([np.isin(sflist,sf2use),typelist==tt, noiselist==nn],axis=0)
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

    for ss in range(np.size(steps)):
        plt.errorbar(un,disc[:,ss], sd_disc[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])

    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, all SF, noise=%.2f' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1   
    
    #%% Plot the standardized euclidean distance, across a subset of PHASES, multiple orientation steps
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf2use = [0,1,2]

tt = 0

plt.figure()
xx=1
steps = np.arange(0,5,1)
legendlabs = ['%d deg apart'%(steps[ss]+1) for ss in range(np.size(steps))]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([np.isin(sflist,sf2use),phaselist<4,typelist==tt, noiselist==nn],axis=0)
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

    for ss in range(np.size(steps)):
        plt.errorbar(un,disc[:,ss], sd_disc[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])

    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, all SF, noise=%.2f' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1   
#%% Discriminability between clockwise/ccw tilt (mirror reflections)
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

tt = 0

plt.figure()
xx=1

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros([90,1])

    for ii in np.arange(0,90):
    
        # first find the position of all gratings with this exact label
        inds1 = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
        inds2 = np.where(np.logical_and(actual_labels==np.mod(-un[ii], 180), myinds_bool))[0]    
        
        
        dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])
        
        disc[ii] = dist
            
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    plt.plot(un[1:90], disc[1:90])
       
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('orientation of clockwise grating')
        plt.ylabel('discriminability (std. euc dist)')
#        plt.legend(legendlabs)
    else:
        plt.xticks([])

    plt.suptitle('Discriminability (std. euc distance) between left-right-flipped orientations \n%s, all SF, noise=%.2f' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1
    
        
#%% plot the within-orientation variance for the first few units - within a spatial frequency
    
    
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [8,9]
#layers2plot = []
sf = 3 # spat freq
tt = 0

plt.figure()
xx=1
units = np.arange(0,15,1)
#units = [0]
legendlabs = ['unit %d' %(ii+1) for ii in units]

for ww1 in layers2plot:

    # first get my covariance matrix, across all images. 
#    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
#    my_cov_inv = np.linalg.inv(my_cov)
     
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
#    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros(np.shape(un))
    sd_disc = np.zeros(np.shape(un))
    
    disc_same = np.zeros(np.shape(un))
    sd_disc_same = np.zeros(np.shape(un))
    
    var_list = np.zeros([np.shape(un)[0],np.size(units)])

    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
  
        # get the pooled variances - between the center and the left bin, and between the center and the right bin.
        var_within = np.var(allw[ww1][ww2][inds,:],0)

        var_list[ii,:] = var_within[units]
        
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for ss in units:
        plt.plot(un,var_list[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('variance')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
                 
    plt.suptitle('Variance of individual unit''s response to different phases at the same orientation\n%s, SF=%.2f, noise=%.2f' % (stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1
    
#%% plot the within-orientation variance for the first few units - across all spatial frequencies
    
    
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [8,9]
#layers2plot = []
#sf = 0 # spat freq
tt = 0

plt.figure()
xx=1
#units = np.arange(0,5,1)
units = [0,1,2]
legendlabs = ['unit %d' %(ii+1) for ii in units]

for ww1 in layers2plot:

    # first get my covariance matrix, across all images. 
#    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
#    my_cov_inv = np.linalg.inv(my_cov)
     
    
#    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros(np.shape(un))
    sd_disc = np.zeros(np.shape(un))
    
    disc_same = np.zeros(np.shape(un))
    sd_disc_same = np.zeros(np.shape(un))
    
    var_list = np.zeros([np.shape(un)[0],np.size(units)])

    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
  
        # get the pooled variances - between the center and the left bin, and between the center and the right bin.
        var_within = np.var(allw[ww1][ww2][inds,:],0)

        var_list[ii,:] = var_within[units]
        
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for ss in units:
        plt.plot(un,var_list[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('variance')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
                 
    plt.suptitle('Variance of individual unit''s response to different phases at the same orientation\n%s, all SF, noise=%.2f' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1
  
    
    #%% plot the within-orientation variance for the first few units - across all spatial frequencies and noise levels
    
    
plt.close('all')
#nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [8,9]
#layers2plot = []
#sf = 0 # spat freq
tt = 0

plt.figure()
xx=1
#units = np.arange(0,5,1)
units = [3,4,5]
legendlabs = ['unit %d' %(ii+1) for ii in units]

for ww1 in layers2plot:

    # first get my covariance matrix, across all images. 
#    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
#    my_cov_inv = np.linalg.inv(my_cov)
     
    
#    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    myinds_bool = np.all([typelist==tt],axis=0)
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros(np.shape(un))
    sd_disc = np.zeros(np.shape(un))
    
    disc_same = np.zeros(np.shape(un))
    sd_disc_same = np.zeros(np.shape(un))
    
    var_list = np.zeros([np.shape(un)[0],np.size(units)])

    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
  
        # get the pooled variances - between the center and the left bin, and between the center and the right bin.
        var_within = np.var(allw[ww1][ww2][inds,:],0)

        var_list[ii,:] = var_within[units]
        
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for ss in range(np.size(units)):
        plt.plot(un,var_list[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('variance')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
                 
    plt.suptitle('Variance of individual unit''s response to different phases at the same orientation\n%s, all SF and noise levels' % (stim_types[tt]))
             
    xx=xx+1

#%% Mahalanobis distance (not quite correct) within each envelope and SF
    
plt.close('all')
nn=0
ww2=0
layers2plot = [7]

sf2plot = np.arange(0,nSF)
type2plot = np.arange(0,nType)

for ww1 in layers2plot:
       
     # first get my covariance matrix, across all images. 
    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
    my_cov_inv = np.linalg.inv(my_cov)
     
    
    plt.figure()
    xx=1
    for sf in sf2plot:            
        for tt in type2plot:
        
            myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
            un,ia = np.unique(actual_labels, return_inverse=True)
            assert np.all(np.expand_dims(ia,1)==actual_labels)
            disc = np.zeros(np.shape(un))
            sd_disc = np.zeros(np.shape(un))
#            t_disc = np.zeros(np.shape(un))
            for ii in np.arange(0,np.size(un)):
            
                # first find the position of all gratings with this exact label
                inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    

                # then find the positions of nearest neighbor gratings
                inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                
                all_mahal_dist = []
                
                for pp in range(np.size(inds)):
                    
                    for left  in range(np.size(inds_left)):
                        
                        # get mahalanobis distance, feeding in my full covariance matrix computed above.
                        # this gives the same output as: 
                        # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                        this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], VI=my_cov_inv)
                        all_mahal_dist.append(this_dist)
                        
                    for right  in range(np.size(inds_right)):
                        this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], VI=my_cov_inv)
                        all_mahal_dist.append(this_dist)
                    
                disc[ii] = np.mean(all_mahal_dist)
                sd_disc[ii] = np.std(all_mahal_dist)

            plt.subplot(len(sf2plot), len(type2plot),xx)

#            plt.scatter(un, disc/np.max(disc))
            plt.errorbar(un, disc, sd_disc)
            plt.title('%s, SF=%.2f' % (stim_types[tt],sf_vals[sf]))
            
            if sf==len(sf2plot)-1:
                plt.xlabel('actual orientation of grating')
                if tt==0:
                    plt.ylabel('discriminability (Maha. dist) from neighbors')
                    
            else:
                plt.xticks([0,45,90,135,180])
 
            xx=xx+1

    plt.suptitle('Discriminability for %s layer\nnoise=%.2f' % (layer_labels[ww1], noise_levels[nn]))

#%% Mahalanobis distance (not quite correct) within one envelope and SF, overlay noise levels
    
plt.close('all')
noise2plot=np.arange(0,nNoiseLevels)
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = 3 # spat freq
tt = 0

plt.figure()
xx=1

legendlabs = [];

for ww1 in layers2plot:
    
     # first get my covariance matrix, across all images. take its inverse too
    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
    my_cov_inv = np.linalg.inv(my_cov)
    
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
#    plt.subplot(2,1,xx)
    for nn in noise2plot:
    
        legendlabs.append('noise=%.2f' % noise_levels[nn])
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
    
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        
        disc = np.zeros(np.shape(un))
        sd_disc = np.zeros(np.shape(un))
    #    t_disc = np.zeros(np.shape(un))
        
        for ii in np.arange(0,np.size(un)):
        
             # first find the position of all gratings with this exact label
            inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
    
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            
            all_mahal_dist = []
            
            for pp in range(np.size(inds)):
                
                for left  in range(np.size(inds_left)):
                    
                    # get mahalanobis distance, feeding in my full covariance matrix computed above.
                    # this gives the same output as: 
                    # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], VI=my_cov_inv)
#                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
                    
                for right  in range(np.size(inds_right)):
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], VI=my_cov_inv)
#                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
 
            disc[ii] = np.mean(all_mahal_dist)
            sd_disc[ii] = np.std(all_mahal_dist)
    
        
#        plt.scatter(un, disc)
        plt.errorbar(un,disc,sd_disc)
        plt.xticks([0,45,90,135,180])
#        plt.plot(un,disc,color='k')
        
        
    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-3]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Maha. dist)')
    else:
        plt.xticks([])
        
    if ww1==layers2plot[-1]:
        plt.legend(legendlabs)
                 
   
    plt.suptitle('Discriminability at each orientation\n%s, sf=%.2f' % (stim_types[tt],sf_vals[sf]))
             
    xx=xx+1
    


#%% Mahalanobis distance (not quite correct) across all layers, within one envelope and SF, overlay spatial freqs
    
plt.close('all')
sf2plot=np.arange(0,nSF)
ww2 = 0;
nn=0;
#layers2plot = np.arange(0,nLayers,1)
layers2plot = [12]
#sf = 3 # spat freq
tt = 0

plt.figure()
xx=1

legendlabs = [];

for ww1 in layers2plot:
    
    # first get my covariance matrix, across all images. take its inverse too
    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
    my_cov_inv = np.linalg.inv(my_cov)
    
    
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for sf in sf2plot:
    
        legendlabs.append('sf=%.2f' % sf_vals[sf])
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
    
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        
        disc = np.zeros(np.shape(un))
        sd_disc = np.zeros(np.shape(un))
    #    t_disc = np.zeros(np.shape(un))
        
        for ii in np.arange(0,np.size(un)):
        
             # first find the position of all gratings with this exact label
            inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
    
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            all_mahal_dist = []
            
            for pp in range(np.size(inds)):
                
                for left  in range(np.size(inds_left)):
                    
                    # get mahalanobis distance, feeding in my full covariance matrix computed above.
                    # this gives the same output as: 
                    # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], VI=my_cov_inv)
    #                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
                    
                for right  in range(np.size(inds_right)):
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], VI=my_cov_inv)
    #                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
                    
            disc[ii] = np.mean(all_mahal_dist)
            sd_disc[ii] = np.std(all_mahal_dist)
    
        
#        plt.scatter(un, disc)
        plt.errorbar(un,disc,sd_disc)
        
        
    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-3]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Maha. dist)')
    else:
        plt.xticks([])
        
    if ww1==layers2plot[-1]:
        plt.legend(legendlabs)
                 
   
    plt.suptitle('Discriminability at each orientation\n%s, noise=%.2f, all spatial freq' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1
#%% discriminability across all layers, within one envelope and SF, overlay spatial freqs
# using limited number of units
    
plt.close('all')
sf2plot=np.arange(0,nSF)
ww2 = 0;
nn=0;
#layers2plot = np.arange(0,nLayers,1)
layers2plot = [12,13,14,15]
#sf = 3 # spat freq
tt = 0

plt.figure()
xx=1

legendlabs = [];

nunits=int(50)

for ww1 in layers2plot:
    
    # first get my covariance matrix, across all images. take its inverse too
    my_cov = np.cov(np.transpose(allw[ww1][ww2][:,0:nunits]))
#    my_cov_inv = np.linalg.inv(my_cov)
    
    
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for sf in sf2plot:
    
        legendlabs.append('sf=%.2f' % sf_vals[sf])
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
    
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        
        disc = np.zeros(np.shape(un))
        sd_disc = np.zeros(np.shape(un))
    #    t_disc = np.zeros(np.shape(un))
        
        for ii in np.arange(0,np.size(un)):
        
             # first find the position of all gratings with this exact label
            inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
    
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            all_mahal_dist = []
            
            for pp in range(np.size(inds)):
                
                for left  in range(np.size(inds_left)):
                    
                    # get mahalanobis distance, feeding in my full covariance matrix computed above.
                    # this gives the same output as: 
                    # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
#                    this_dist = scipy.spatial.distance.euclidean(allw[ww1][ww2][inds[pp],0:nunits],allw[ww1][ww2][inds_left[left],0:nunits])
                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],0:nunits],allw[ww1][ww2][inds_left[left],0:nunits], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
                    
                for right  in range(np.size(inds_right)):
#                    this_dist = scipy.spatial.distance.euclidean(allw[ww1][ww2][inds[pp],0:nunits],allw[ww1][ww2][inds_right[right],0:nunits])
                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],0:nunits],allw[ww1][ww2][inds_right[right],0:nunits], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
                    
            disc[ii] = np.mean(all_mahal_dist)
            sd_disc[ii] = np.std(all_mahal_dist)
    
        
#        plt.scatter(un, disc)
        plt.errorbar(un,disc,sd_disc)
        
        
    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-3]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Maha. dist)')
    else:
        plt.xticks([])
        
    if ww1==layers2plot[-1]:
        plt.legend(legendlabs)
                 
   
    plt.suptitle('Discriminability at each orientation\n%s, noise=%.2f, all spatial freq' % (stim_types[tt],noise_levels[nn]))
             
    xx=xx+1



#%% plot anisotropy versus layers: one noise level and sf at a time 
      
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = 3 # spat freq
tt = 0
nn = 0

aniso_vals = np.zeros([4,np.size(layers2plot)])

for ww1 in layers2plot:
    
    # first get my covariance matrix, across all images. take its inverse too
    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
    my_cov_inv = np.linalg.inv(my_cov)
     
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
        
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    # get the full distance function across orientation space
    
    disc = np.zeros(np.shape(un))
    sd_disc = np.zeros(np.shape(un))
#    t_disc = np.zeros(np.shape(un))
    
    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    

        # then find the positions of nearest neighbor gratings
        inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
        inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
        
        
        all_mahal_dist = []
        
        for pp in range(np.size(inds)):
            
            for left  in range(np.size(inds_left)):
                
                # get mahalanobis distance, feeding in my full covariance matrix computed above.
                # this gives the same output as: 
                # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], VI=my_cov_inv)
#                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], V=np.diag(my_cov))
                all_mahal_dist.append(this_dist)
                
            for right  in range(np.size(inds_right)):
                this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], VI=my_cov_inv)
#                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], V=np.diag(my_cov))
                all_mahal_dist.append(this_dist)
 
        disc[ii] = np.mean(all_mahal_dist)
        sd_disc[ii] = np.std(all_mahal_dist)
    
    # take the bins of interest to get amplitude
    b = np.arange(22.5,180,45)
    baseline_discrim = [];
    for ii in range(np.size(b)):        
        inds = np.where(np.abs(un-b[ii])<4)[0]        
        baseline_discrim.append(disc[inds])
        
    a = np.arange(0,180,45)
    
    for ii in range(np.size(a)):       
        inds = np.where(np.abs(un-a[ii])<1)[0]
        aniso_vals[ii,ww1] = np.mean(disc[inds]) - np.mean(baseline_discrim)
  
    
plt.figure()
ylims = [-11,30]
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
plt.ylabel('Maha. distance difference')
plt.xlabel('Layer number')

#%% plot anisotropy versus layers: overlay spatial frequencies
 
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
#sf = 0 # spat freq
sf2plot = np.arange(0,nSF,1)
tt = 0
nn = 0
ylims = [-11,21]

aniso_vals = np.zeros([4,nSF,np.size(layers2plot)])
for ww1 in layers2plot:
    
    # first get my covariance matrix, across all images. take its inverse too
    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
    my_cov_inv = np.linalg.inv(my_cov)
    
    for sf in sf2plot:
         
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
            
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        
        # get the full distance function across orientation space
        
        disc = np.zeros(np.shape(un))
        sd_disc = np.zeros(np.shape(un))
    #    t_disc = np.zeros(np.shape(un))
        
        for ii in np.arange(0,np.size(un)):
        
             # first find the position of all gratings with this exact label
            inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
    
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            
            all_mahal_dist = []
            
            for pp in range(np.size(inds)):
                
                for left  in range(np.size(inds_left)):
                    
                    # get mahalanobis distance, feeding in my full covariance matrix computed above.
                    # this gives the same output as: 
                    # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], VI=my_cov_inv)
    #                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
                    
                for right  in range(np.size(inds_right)):
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], VI=my_cov_inv)
    #                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
     
            disc[ii] = np.mean(all_mahal_dist)
            sd_disc[ii] = np.std(all_mahal_dist)
        
        # take the bins of interest to get amplitude
        b = np.arange(22.5,180,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,180,45)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.abs(un-a[ii])<1)[0]
            aniso_vals[ii,sf,ww1] = np.mean(disc[inds]) - np.mean(baseline_discrim)
            
plt.close('all')
plt.figure()
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]
#ylims = [-11,21]
h = []
for sf in range(np.size(sf2plot)):
    card_mean = np.mean(aniso_vals[(0,2),sf2plot[sf],:],0)
    h.append(plt.plot(layers2plot,card_mean))

plt.legend(['sf=%.2f' %sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nDiscriminability at cardinals rel. to non-cardinals\nnoise=%.2f' % (model_name_2plot, noise_levels[nn]))
plt.ylabel('Maha. distance difference')
plt.xlabel('Layer number')

#%% plot anisotropy versus layers: overlay noise levels
 
plt.close('all')

ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [6, 12]
sf = 0 # spat freq
#sf2plot = np.arange(0,nSF,1)
tt = 0
noiselevels2plot = np.arange(0,nNoiseLevels,1)
ylims = [-11,21]

aniso_vals = np.zeros([4,nSF,np.size(layers2plot)])
for ww1 in layers2plot:
    
    # first get my covariance matrix, across all images. take its inverse too
    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
    my_cov_inv = np.linalg.inv(my_cov)
    
    for nn in noiselevels2plot:
         
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
            
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        
        # get the full distance function across orientation space
        
        disc = np.zeros(np.shape(un))
        sd_disc = np.zeros(np.shape(un))
    #    t_disc = np.zeros(np.shape(un))
        
        for ii in np.arange(0,np.size(un)):
        
             # first find the position of all gratings with this exact label
            inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
    
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            
            all_mahal_dist = []
            
            for pp in range(np.size(inds)):
                
                for left  in range(np.size(inds_left)):
                    
                    # get mahalanobis distance, feeding in my full covariance matrix computed above.
                    # this gives the same output as: 
                    # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], VI=my_cov_inv)
    #                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
                    
                for right  in range(np.size(inds_right)):
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], VI=my_cov_inv)
    #                    this_dist = scipy.spatial.distance.seuclidean(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], V=np.diag(my_cov))
                    all_mahal_dist.append(this_dist)
     
            disc[ii] = np.mean(all_mahal_dist)
            sd_disc[ii] = np.std(all_mahal_dist)
        
        # take the bins of interest to get amplitude
        b = np.arange(22.5,180,45)
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.abs(un-b[ii])<4)[0]        
            baseline_discrim.append(disc[inds])
            
        a = np.arange(0,180,45)
        
        for ii in range(np.size(a)):       
            inds = np.where(np.abs(un-a[ii])<1)[0]
            aniso_vals[ii,nn,ww1] = np.mean(disc[inds]) - np.mean(baseline_discrim)
            
plt.close('all')
plt.figure()
xlims = [np.min(layers2plot)-1, np.max(layers2plot)+2]
#ylims = [-11,21]
h = []
for nn in range(np.size(noiselevels2plot)):
    card_mean = np.mean(aniso_vals[(0,2),noiselevels2plot[nn],:],0)
    h.append(plt.plot(layers2plot,card_mean))

plt.legend(['noise=%.2f' %noise_levels[nn] for nn in noiselevels2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(layers2plot,[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nDiscriminability at cardinals rel. to non-cardinals\nsf=%.2f' % (model_name_2plot, sf_vals[sf]))
plt.ylabel('Maha. distance difference')
plt.xlabel('Layer number')
