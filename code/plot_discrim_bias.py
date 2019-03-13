#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt
import scipy
import numpy as np

#%% get the data ready to go...then can run any below cells independently.

model_str = 'inception_oriTst0'

root = '/usr/local/serenceslab/maggie/biasCNN/';
import os
os.chdir(os.path.join(root, 'code'))

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
nn=2

pred_labels = all_labs[0][1]
pred_labels_before = all_labs[0][0]
    

for tt in range(len(stim_types)):
    for bb in range(len(sf_vals)):

        myinds_bool = np.logical_and(np.logical_and(typelist==tt, sflist==bb), noiselist==nn)
        
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
        plt.title('After retraining, noise=%.2f\n predicted labels versus actual labels, %s, SF=%.2f' % (noise_levels[nn], stim_types[tt],sf_vals[bb]))


#%% Discriminability within each envelope and SF
    
plt.close('all')
nn=1
ww2=0
layers2plot = [0,2,5,10,18]

sf2plot = np.arange(0,nSF)
type2plot = np.arange(0,nType)


for ww1 in layers2plot:
        
    plt.figure()
    xx=1
    for bb in sf2plot:            
        for tt in type2plot:
        
            myinds_bool = np.all([sflist==bb,   typelist==tt, noiselist==nn], axis=0)
    
            un,ia = np.unique(actual_labels, return_inverse=True)
            assert np.all(np.expand_dims(ia,1)==actual_labels)
            disc = np.zeros(np.shape(un))
#            t_disc = np.zeros(np.shape(un))
            for ii in np.arange(0,np.size(un)):
            
                # first find the position of all gratings with this exact label
                inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    

                # then find the positions of nearest neighbor gratings
                inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
                
                diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0)))      
                assert np.size(diffs2)==64
                diffs2 = diffs2[0:4,4:8]
                diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
        
                diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0)))      
                assert np.size(diffs3)==64
                diffs3 = diffs3[0:4,4:8]
                diffs3 = np.reshape(diffs3, np.prod(np.shape(diffs3)))
            
                diffs_off = np.concatenate((diffs2,diffs3),axis=0)

                disc[ii] = np.mean(diffs_off)

            plt.subplot(len(sf2plot), len(type2plot),xx)

            plt.scatter(un, disc)
            plt.title('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
            
            if bb==len(sf2plot)-1:
                plt.xlabel('actual orientation of grating')
                if tt==0:
                    plt.ylabel('discriminability (Euc. dist) from neighbors')
                    
            else:
                plt.xticks([])
 
            xx=xx+1

    plt.suptitle('Discriminability for %s layer\nnoise=%.2f' % (layer_labels[ww1], noise_levels[nn]))

#%% Plot discriminability across all layers, within one envelope and SF
    
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

bb = 3 # spat freq
tt = 0

plt.figure()
xx=1

for ww1 in layers2plot:

    myinds_bool = np.all([sflist==bb,   typelist==tt, noiselist==nn], axis=0)
    
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros(np.shape(un))
#    t_disc = np.zeros(np.shape(un))
    
    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    

        # then find the positions of nearest neighbor gratings
        inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
        inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
        
        diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0)))      
        assert np.size(diffs2)==64
        diffs2 = diffs2[0:4,4:8]
        diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))

        diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0)))      
        assert np.size(diffs3)==64
        diffs3 = diffs3[0:4,4:8]
        diffs3 = np.reshape(diffs3, np.prod(np.shape(diffs3)))
    
        diffs_off = np.concatenate((diffs2,diffs3),axis=0)

        disc[ii] = np.mean(diffs_off)


    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    plt.scatter(un, disc)
    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-3]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Euc. dist)')
    else:
        plt.xticks([])
                 
   
    plt.suptitle('Discriminability at each orientation\n%s, SF=%.2f, noise=%.2f' % (stim_types[tt],sf_vals[bb],noise_levels[nn]))
             
    xx=xx+1

#%% Plot discriminability across all layers, within one envelope and SF, overlay noise levels
    
plt.close('all')
noise2plot=np.arange(0,nNoiseLevels)
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

bb = 3 # spat freq
tt = 0

plt.figure()
xx=1

legendlabs = [];

for ww1 in layers2plot:
    
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for nn in noise2plot:
    
        legendlabs.append('noise=%.2f' % noise_levels[nn])
        
        myinds_bool = np.all([sflist==bb,   typelist==tt, noiselist==nn], axis=0)
    
    
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        
        disc = np.zeros(np.shape(un))
    #    t_disc = np.zeros(np.shape(un))
        
        for ii in np.arange(0,np.size(un)):
        
             # first find the position of all gratings with this exact label
            inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
    
            # then find the positions of nearest neighbor gratings
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, 180), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, 180), myinds_bool))[0]
            
            diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0)))      
            assert np.size(diffs2)==64
            diffs2 = diffs2[0:4,4:8]
            diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
    
            diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0)))      
            assert np.size(diffs3)==64
            diffs3 = diffs3[0:4,4:8]
            diffs3 = np.reshape(diffs3, np.prod(np.shape(diffs3)))
        
            diffs_off = np.concatenate((diffs2,diffs3),axis=0)
    
            disc[ii] = np.mean(diffs_off)
    
    
        
        plt.scatter(un, disc)
        
        
    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-3]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Euc. dist)')
    else:
        plt.xticks([])
        
    if ww1==layers2plot[-1]:
        plt.legend(legendlabs)
                 
   
    plt.suptitle('Discriminability at each orientation\n%s, SF=%.2f, all noise levels' % (stim_types[tt],sf_vals[bb]))
             
    xx=xx+1
