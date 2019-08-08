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

model_str = 'nasnet_oriTst1'
model_name_2plot = 'NASnet'
#
#model_str = 'vgg16_oriTst1'
#model_name_2plot = 'VGG-16'

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

#%% plot predicted vs actual labels

plt.close('all')

noiselevels2plot = np.arange(nNoiseLevels)

un = np.unique(actual_labels)

for nn in noiselevels2plot:

    ind_bool = noiselist==nn
    
    avg_pred = np.zeros(np.shape(un))
    std_pred = np.zeros(np.shape(un))
    avg_pred_before = np.zeros(np.shape(un))
    std_pred_before = np.zeros(np.shape(un))
    
    pred_labels = all_labs[0][1]
    pred_labels_before = all_labs[0][0]
    
    for uu in range(len(un)):
        
        myinds = np.where(np.logical_and(actual_labels==un[uu], ind_bool))[0]
        
        avg_pred[uu] = scipy.stats.circmean(pred_labels[myinds], high=180,low=0)
        std_pred[uu] = scipy.stats.circstd(pred_labels[myinds], high=180,low=0)
        avg_pred_before[uu] = scipy.stats.circmean(pred_labels_before[myinds], high=180,low=0)
        std_pred_before[uu] = scipy.stats.circstd(pred_labels_before[myinds], high=180,low=0)
        
    plt.figure()
    plt.scatter(un,avg_pred_before)
    
    plt.scatter(un,avg_pred)
    
    plt.axis('equal')
    plt.xlim([0,180])
    plt.ylim([0,180])
    
    plt.title('Predicted labels versus actual labels\nAll stims, noise=%.2f' % noise_levels[nn])
    plt.legend(['before retraining','after retraining'])
    plt.plot([0,180],[0,180],'k-')
    plt.plot([90,90],[0,180],'k-')

#%% plot variability in predictions

plt.close('all')

noiselevels2plot = np.arange(nNoiseLevels)

un = np.unique(actual_labels)

for nn in noiselevels2plot:

    ind_bool = noiselist==nn
    
    avg_pred = np.zeros(np.shape(un))
    std_pred = np.zeros(np.shape(un))
    avg_pred_before = np.zeros(np.shape(un))
    std_pred_before = np.zeros(np.shape(un))
    
    pred_labels = all_labs[0][1]
    pred_labels_before = all_labs[0][0]
    
    for uu in range(len(un)):
        
        myinds = np.where(np.logical_and(actual_labels==un[uu], ind_bool))[0]
        
        avg_pred[uu] = scipy.stats.circmean(pred_labels[myinds], high=180,low=0)
        std_pred[uu] = scipy.stats.circstd(pred_labels[myinds], high=180,low=0)
        avg_pred_before[uu] = scipy.stats.circmean(pred_labels_before[myinds], high=180,low=0)
        std_pred_before[uu] = scipy.stats.circstd(pred_labels_before[myinds], high=180,low=0)
        
        
    plt.figure()
#    plt.errorbar(un,avg_pred_before,std_pred_before)
    plt.errorbar(un,avg_pred,std_pred)

    plt.axis('equal')
    plt.xlim([0,180])
#    plt.ylim([0,180])
    
    plt.title('Predicted labels versus actual labels\nAll stims, noise=%.2f' % noise_levels[nn])
#    plt.legend(['before retraining','after retraining'])
    plt.plot([0,180],plt.get(plt.gca(),'xlim'),'k-')
    plt.plot([90,90],plt.get(plt.gca(),'xlim'),'k-')
    plt.plot([0,0],plt.get(plt.gca(),'xlim'),'k-')
    plt.plot([180,180],plt.get(plt.gca(),'xlim'),'k-')
    plt.plot([45,45],plt.get(plt.gca(),'xlim'),'k-')
    plt.plot([135,135],plt.get(plt.gca(),'xlim'),'k-')
    plt.xlabel('actual')
    plt.ylabel('predicted')

#%% plot predicted vs actual labels - within each stim type and spat freq
plt.close('all')
nn=0

pred_labels = all_labs[0][1]
pred_labels_before = all_labs[0][0]
    

for tt in range(len(stim_types)):
    for sf in range(len(sf_vals)):

        myinds_bool = np.logical_and(np.logical_and(typelist==tt, sflist==sf), noiselist==nn)
        
        un = np.unique(actual_labels)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
        avg_pred_before = np.zeros(np.shape(un))
        std_pred_before = np.zeros(np.shape(un))
        
        

        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(np.logical_and(actual_labels==un[uu], myinds_bool))], high=179,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(np.logical_and(actual_labels==un[uu], myinds_bool))], high=179,low=0)
            avg_pred_before[uu] = scipy.stats.circmean(pred_labels_before[np.where(np.logical_and(actual_labels==un[uu], myinds_bool))], high=179,low=0)
            std_pred_before[uu] = scipy.stats.circstd(pred_labels_before[np.where(np.logical_and(actual_labels==un[uu], myinds_bool))], high=179,low=0)
            
        plt.figure()
#        plt.errorbar(un,avg_pred,std_pred)
        plt.scatter(un,avg_pred)
#        plt.scatter(un,avg_pred_before)
        plt.axis('equal')
        plt.xlim([0,180])
        plt.ylim([0,180])
        plt.plot([0,180],[0,180],'k-')
        plt.plot([90,90],[0,180],'k-')
        plt.title('After retraining, noise=%.2f\n predicted labels versus actual labels, %s, SF=%.2f' % (noise_levels[nn], stim_types[tt],sf_vals[sf]))


#%% Plot the standardized euclidean distance, within spatial frequency and noise level
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 3 # spat freq
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

    for ss in steps:
        plt.errorbar(un,disc[:,ss], sd_disc[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(['%d deg apart'%(steps[ss]+1) for ss in range(np.size(steps))])
    else:
        plt.xticks([])
   
    plt.suptitle('Discriminability (std. euc distance) between pairs of orientations\n%s, SF=%.2f, noise=%.2f' % (stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1
    

#%% Plot the standardized euclidean distance, across all SF, multiple orientation steps
 
plt.close('all')
nn=0
ww2 = 0;

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
    

    
#%% plot the within-orientation variance for the first few units - within a spatial frequency
    
    
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [8,9]
#layers2plot = []
sf =3 # spat freq
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
    
#%% Compare within bin/across neighboring bin similarity
 
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = []
sf = 3 # spat freq
tt = 0

plt.figure()
xx=1

for ww1 in layers2plot:

    # first get my covariance matrix, across all images. 
#    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
#    my_cov_inv = np.linalg.inv(my_cov)
     
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc_neighbor = np.zeros(np.shape(un))
    sd_disc_neighbor = np.zeros(np.shape(un))
    
    disc_same = np.zeros(np.shape(un))
    sd_disc_same = np.zeros(np.shape(un))
#    t_disc = np.zeros(np.shape(un))
    
    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    

        # then find the positions of nearest neighbor gratings
        inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
        inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
        
        all_mahal_dist = []
        
        for pp in range(np.size(inds)):
            
            for pp2 in np.arange(pp+1,np.size(inds)):
                
                # get mahalanobis distance, feeding in my full covariance matrix computed above.
                # this gives the same output as: 
                # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds[pp2],:], VI=my_cov_inv)
                all_mahal_dist.append(this_dist)
                
        disc_same[ii] = np.mean(all_mahal_dist)
        sd_disc_same[ii] = np.std(all_mahal_dist)
        
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
            
        disc_neighbor[ii] = np.mean(all_mahal_dist)
        sd_disc_neighbor[ii] = np.std(all_mahal_dist)

    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

#    plt.scatter(un, disc)
#    plt.errorbar(un,disc_neighbor, sd_disc_neighbor)
#    plt.errorbar(un,disc_same, sd_disc_same)
    plt.plot(un,disc_neighbor)
    plt.plot(un,disc_same)
    plt.plot(un,disc_neighbor-disc_same)
#    plt.plot(un,disc_same)
    
    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Maha. dist)')
        plt.legend(['1 degree apart','0 degree apart','subtraction'])
    else:
        plt.xticks([])
                 
   
    plt.suptitle('Discriminability (Maha. distance) between pairs of images\n%s, SF=%.2f, noise=%.2f' % (stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1
#%% Compare within bin/across neighboring bin similarity
    
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = []
sf = 3 # spat freq
tt = 0

plt.figure()
xx=1

nsteps = 5
#legendlabs = ['0 degree apart']
legendlabs = []
for ss in range(nsteps):
    legendlabs.append('%d degree apart-0 degree apart' % (ss+1))
    
for ww1 in layers2plot:

    # first get my covariance matrix, across all images. 
    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
    my_cov_inv = np.linalg.inv(my_cov)
     
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc_neighbor = np.zeros([np.shape(un)[0],nsteps])
    sd_disc_neighbor = np.zeros([np.shape(un)[0],nsteps])
    
    disc_same = np.zeros(np.shape(un))
    sd_disc_same = np.zeros(np.shape(un))
#    t_disc = np.zeros(np.shape(un))
    
    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    

        all_mahal_dist_same = []
        
        for pp in range(np.size(inds)):
            
            for pp2 in np.arange(pp+1,np.size(inds)):
                
                # get mahalanobis distance, feeding in my full covariance matrix computed above.
                # this gives the same output as: 
                # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds[pp2],:], VI=my_cov_inv)
                all_mahal_dist_same.append(this_dist)
                
#        disc_same[ii] = np.mean(all_mahal_dist_same)
#        sd_disc_same[ii] = np.std(all_mahal_dist_same)
        
        for ss in range(nsteps):
            
            all_mahal_dist_diff = []
            
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-(ss+1), 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+(ss+1), 180), myinds_bool))[0]
            
            for pp in range(np.size(inds)):
                
                for left  in range(np.size(inds_left)):
                    
                    # get mahalanobis distance, feeding in my full covariance matrix computed above.
                    # this gives the same output as: 
                    # np.sqrt(np.dot(np.dot((x-y), my_cov_inv), np.transpose(x-y)))
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_left[left],:], VI=my_cov_inv)
                    all_mahal_dist_diff.append(this_dist)
                    
                for right  in range(np.size(inds_right)):
                    this_dist = scipy.spatial.distance.mahalanobis(allw[ww1][ww2][inds[pp],:],allw[ww1][ww2][inds_right[right],:], VI=my_cov_inv)
                    all_mahal_dist_diff.append(this_dist)
                
#            disc_neighbor[ii,ss] = np.mean(all_mahal_dist_diff)
#            sd_disc_neighbor[ii,ss] = np.std(all_mahal_dist_diff)

            (t,p) = scipy.stats.ttest_ind(all_mahal_dist_diff,all_mahal_dist_same,equal_var=False)
            disc_neighbor[ii,ss] = t
#            sd_disc_neighbor[ii,ss] = scipy.stats.ttest_ind(all_mahal_dist_diff,all_mahal_dist_same,equal_var=False)

    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

#    plt.scatter(un, disc)
#    plt.errorbar(un,disc_neighbor, sd_disc_neighbor)
#    plt.errorbar(un,disc_same, sd_disc_same)
    
#    plt.plot(un,disc_same)
    
    for ss in range(nsteps):
        plt.plot(un,disc_neighbor[:,ss])
#    plt.plot(un,disc_neighbor-disc_same)
#    plt.plot(un,disc_same)
    
    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (t-stat)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
                 
#    ylims = [0,30]
    plt.ylim(ylims)
    for ii in np.arange(0,180,45):
        plt.plot([ii,ii],ylims,'k')
    plt.suptitle('Discriminability (t-score) between pairs of images\n%s, SF=%.2f, noise=%.2f' % (stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
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
layers2plot = np.arange(0,nLayers,1)

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
