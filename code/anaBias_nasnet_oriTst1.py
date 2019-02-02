#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

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

from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict


#import pycircstat


weight_path_before = os.path.join(root, 'activations', 'nasnet_oriTst1_short_reduced')
weight_path_after = os.path.join(root, 'activations', 'nasnet_oriTst1_long_reduced')
#weight_path_after = os.path.join(root, 'activations', 'nasnet_oriTst1_long_reduced')
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
noise_levels = [0, 0.25, 0.5]
nNoiseLevels = np.size(noise_levels)
sf_vals = np.logspace(np.log10(0.4), np.log10(2.2),4)
stim_types = ['Gaussian']
nOri=180
nSF=np.size(sf_vals)
nPhase=4
nType=np.size(stim_types)

# list all the image features in a big matrix, where every row is unique.
noiselist = np.expand_dims(np.repeat(np.arange(nNoiseLevels), nPhase*nOri*nSF*nType),1)
typelist = np.transpose(np.tile(np.repeat(np.arange(nType), nPhase*nOri*nSF), [1,nNoiseLevels]))
orilist=np.transpose(np.tile(np.repeat(np.arange(nOri),nSF*nPhase), [1,nType*nNoiseLevels]))
sflist=np.transpose(np.tile(np.repeat(np.arange(nSF),nPhase),[1,nOri*nType*nNoiseLevels]))
phaselist=np.transpose(np.tile(np.arange(nPhase),[1,nOri*nSF*nType*nNoiseLevels]))

featureMat = np.concatenate((noiselist,typelist,orilist,sflist,phaselist),axis=1)

assert np.array_equal(featureMat, np.unique(featureMat, axis=0))

actual_labels = orilist


#%% load the data (already in reduced/PCA-d format)

allw = []

for ll in range(np.size(layer_labels)):
    
    tmp = []
    
#    for nn in range(np.size(noise_levels)):
#        print(nn)
#        print(noise_levels[nn])
    file = os.path.join(weight_path_before, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
    w1 = np.load(file)
    
    file = os.path.join(weight_path_after, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
    w2 = np.load(file)
    
    tmp.append([w1,w2])
        
    allw.append(tmp)
#    allw.append([w1])
    
nLayers = len(allw)
nTimepts = len(allw[0][0])
# can change these if you want just a subset of plots made at a time
layers2plot = np.arange(0,nLayers,1)
#timepts2plot = np.arange(0,nTimepts,1)
timepts2plot = np.arange(0,1)

#%% load the predicted orientation labels from the re-trained network

num_batches = 80

all_labs = []

file = os.path.join(weight_path_before, 'allStimsLabsPredicted_Cell_1.npy')    
labs1 = np.load(file)
 
file = os.path.join(weight_path_after,'allStimsLabsPredicted_Cell_1.npy')    
labs2 = np.load(file)

all_labs.append([labs1,labs2])
 

#%% plot predicted vs actual labels

plt.close('all')

noiselevels2plot = [0,1,2]

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

noiselevels2plot = [0,1,2]

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
                inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, nOri), myinds_bool))[0]        
                inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, nOri), myinds_bool))[0]
                
                diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][0][ww2][inds,:], allw[ww1][0][ww2][inds_left,:]),0)))      
                assert np.size(diffs2)==64
                diffs2 = diffs2[0:4,4:8]
                diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
        
                diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][0][ww2][inds,:], allw[ww1][0][ww2][inds_right,:]),0)))      
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
        inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, nOri), myinds_bool))[0]        
        inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, nOri), myinds_bool))[0]
        
        diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][0][ww2][inds,:], allw[ww1][0][ww2][inds_left,:]),0)))      
        assert np.size(diffs2)==64
        diffs2 = diffs2[0:4,4:8]
        diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))

        diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][0][ww2][inds,:], allw[ww1][0][ww2][inds_right,:]),0)))      
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
noise2plot=np.arange(0,3)
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
            inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, nOri), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, nOri), myinds_bool))[0]
            
            diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][0][ww2][inds,:], allw[ww1][0][ww2][inds_left,:]),0)))      
            assert np.size(diffs2)==64
            diffs2 = diffs2[0:4,4:8]
            diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
    
            diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][0][ww2][inds,:], allw[ww1][0][ww2][inds_right,:]),0)))      
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
#%% Plot discriminability across all layers, generalizing across stims. Also get t-statistic...still playing with this.
    
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

plt.figure()

center_deg=90
 
xx=1
for ww1 in layers2plot:
#    for ww2 in timepts2plot:    

#    myinds_bool = np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt))

    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    disc = np.zeros(np.shape(un))
    t_disc = np.zeros(np.shape(un))
    
    for ii in np.arange(0,np.size(un)):
    
        # first find the discriminability of all gratings with this exact label (should be close to zero)
        inds = np.where(actual_labels==un[ii])[0]    
        diffs1 = np.triu(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(allw[ww1][nn][ww2][inds,:])),1)        
        # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
#        assert not np.any(corrs1==1)        
        diffs1 = diffs1[np.where(diffs1!=0)]

        
        # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
        inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, nOri), myinds_bool))[0]        
        inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, nOri), myinds_bool))[0]
        
        diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][nn][ww2][inds,:], allw[ww1][nn][ww2][inds_left,:]),0)))      
        diffs2 = diffs2[0:np.size(inds),np.size(inds):]
        diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))

        diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][nn][ww2][inds,:], allw[ww1][nn][ww2][inds_right,:]),0)))      
        diffs3 = diffs3[0:np.size(inds),np.size(inds):]
        diffs3 = np.reshape(diffs3, np.prod(np.shape(diffs3)))
    
       
        diffs_off = np.concatenate((diffs2,diffs3),axis=0)
        diffs_same = diffs1
        
        # ddof=1 for sample variance, 0 for population variance.
        pooled_var = (np.var(diffs_off, ddof=1)/np.size(diffs_off)) + (np.var(diffs_same,ddof=1)/np.size(diffs_same))
        my_t = (np.mean(diffs_off) - np.mean(diffs_same))/np.sqrt(pooled_var)

        # this gives same result
#        t, p = scipy.stats.ttest_ind(diffs_off, diffs_same, equal_var=False)
        disc[ii] = np.mean(diffs_off)
#        disc[ii] = np.mean(diffs_off) - np.mean(diffs_same)
        t_disc[ii] = my_t
        
#    disc = disc/np.max(disc)
    plt.figure(1)
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    plt.scatter(un, disc)
#            plt.title('Discriminability versus orientation - before training')
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-3]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Euc. dist)')
    else:
        plt.xticks([])
#    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
#    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
#    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')             
   
      
    plt.figure(2)
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
#                plt.figure()
    xx=xx+1
    plt.scatter(un, t_disc)
#            plt.title('Discriminability versus orientation - before training')
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-3]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (t-statistic)')
    else:
        plt.xticks([])
#    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
#    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
#    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')              
  
plt.figure(1)
plt.suptitle('Discriminability at each orientation\nall stims, noise=%d' % (noise_levels[nn]))
 
plt.figure(2)
plt.suptitle('T-scored discriminability at each orientation\nall stims, noise=%d' % (noise_levels[nn]))
        
      
#%% Bias across all stims (everything below this is old)

plt.close('all')
nn=0
layers2plot = np.arange(0,nLayers,6)

#sf2plot = np.arange(4,5)
##type2plot = np.arange(1,2)
#sf2plot = np.arange(0,nSF)
#type2plot = np.arange(0,nType)


for ww1 in layers2plot:
    for ww2 in timepts2plot:
 
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        bias = np.zeros(np.shape(un))
        
        for ii in np.arange(0,np.size(un)):
        
            # first find the discriminability of all gratings with this exact label (should be close to zero)
            inds = np.where(actual_labels==un[ii])[0]    
            diffs1 = np.triu(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(allw[ww1][nn][ww2][inds,:])),1)        
            # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
    #        assert not np.any(corrs1==1)        
            diffs1 = diffs1[np.where(diffs1!=0)]
            
            
            # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
            inds_left = np.where(actual_labels==np.mod(un[ii]-1, nOri))[0]        
            inds_right = np.where(actual_labels==np.mod(un[ii]+1, nOri))[0]
            
            diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][nn][ww2][inds,:], allw[ww1][nn][ww2][inds_left,:]),0)))      
            diffs2 = diffs2[0:np.size(inds),np.size(inds):]
            diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
            
            diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][nn][ww2][inds,:], allw[ww1][nn][ww2][inds_right,:]),0)))      
            diffs3 = diffs3[0:np.size(inds),np.size(inds):]
            diffs3 = np.reshape(diffs3,np.prod(np.shape(diffs3)))
                
            diffs_off = np.concatenate((diffs2,diffs3),axis=0)
    #        disc[ii] = np.mean(diffs_off)/(np.std(diffs_off)/np.sqrt(np.size(diffs_off)-1)) -\
    #                    np.mean(diffs1)/(np.std(diffs1)/np.sqrt(np.size(diffs1)-1))
            bias[ii] = np.mean(diffs2) - np.mean(diffs3)
            
         
            
        plt.figure()
        plt.scatter(un, bias)
        plt.title('bias for %s layer\n %s - all stims, noise=%.2f' % (layer_labels[ww1],timepoint_labels[ww2], noise_levels[nn]))
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Euc. dist) from neighbors')
        ylims = [np.min(bias), np.max(bias)]
        plt.ylim(ylims)
        for ii in np.arange(0,181,45):
            plt.plot([ii,ii],ylims)

#%% Bias at each envelope and SF
plt.close('all')
nn=0
layers2plot = np.arange(0,nLayers,6)
#sf2plot = np.arange(4,5)
#type2plot = np.arange(1,2)
sf2plot = np.arange(0,nSF)
type2plot = np.arange(0,nType)


#ylims = np.asarray([-4,4])

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        plt.figure()
        xx=1
        
        for bb in sf2plot:            
            for tt in type2plot:
            
                myinds_bool = np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt))
        
                un,ia = np.unique(actual_labels, return_inverse=True)
                assert np.all(np.expand_dims(ia,1)==actual_labels)
                bias = np.zeros(np.shape(un))
                
                for ii in np.arange(0,np.size(un)):
                
                    # first find the discriminability of all gratings with this exact label (should be close to zero)
                    inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
                    diffs1 = np.triu(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(allw[ww1][nn][ww2][inds,:])),1)        
                    # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
            #        assert not np.any(corrs1==1)        
                    diffs1 = diffs1[np.where(diffs1!=0)]
            
                    
                    # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
                    inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, nOri), myinds_bool))[0]        
                    inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, nOri), myinds_bool))[0]
                    
                    diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][nn][ww2][inds,:], allw[ww1][nn][ww2][inds_left,:]),0)))      
                    diffs2 = diffs2[0:np.size(inds),np.size(inds):]
                    diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
            
                    diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][nn][ww2][inds,:], allw[ww1][nn][ww2][inds_right,:]),0)))      
                    diffs3 = diffs3[0:np.size(inds),np.size(inds):]
                    diffs3 = np.reshape(diffs3, np.prod(np.shape(diffs3)))
                
                    bias[ii] =  np.mean(diffs2) - np.mean(diffs3)
                    
                plt.subplot(len(sf2plot), len(type2plot),xx)
        #                plt.figure()
                xx=xx+1
                plt.scatter(un, bias)
        #            plt.title('Discriminability versus orientation - before training')
                plt.title('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
                ylims = [np.min(bias)-0.5, np.max(bias)+0.5]
                plt.ylim(ylims)
                for ii in np.arange(0,181,45):
                    plt.plot([ii,ii],ylims)
                    
                if bb==len(sf2plot)-1:
                    plt.xlabel('actual orientation of grating')
                    plt.ylabel('clockwise bias (euc dist)')
                else:
                    plt.xticks([])
                     
        plt.suptitle('Bias for %s layer\n %s, noise=%.2f' % (layer_labels[ww1],timepoint_labels[ww2], noise_levels[nn]))
 

#%% linear classification
plt.close('all')

layers2plot = np.arange(0,nLayers,6)

# prepare for random cross-validation
ntrials = np.shape(allw[0][0])[0]
prop_test = 0.10;
assert np.mod(ntrials*prop_test,1)==0
ntest = int(ntrials*prop_test)
nCV = int(1/prop_test)

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        
        alldat = allw[ww1][ww2]
        
        print('processing %s %s\n' % (layer_labels[ww1],timepoint_labels[ww2]) )
 
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        disc = np.zeros(np.shape(un))
        
        idealobs_labels = np.zeros(np.shape(actual_labels))
        eucdist_labels = np.zeros(np.shape(actual_labels))
        
        randinds = np.arange(0,ntrials)
        np.random.shuffle(randinds)
        randinds = np.reshape(randinds, [ntest, nCV])
        for cv in range(nCV):
#        for cv in range(0,1):
            
           print('cross-validation fold %d of %d\n' % (cv, nCV))
            
           tstinds = randinds[:,cv]
           trninds = np.setdiff1d(randinds, tstinds, assume_unique=True)
           
           likelihoods, ll, labels = classifiers.ideal_observer_gaussian(alldat[trninds,:], alldat[tstinds,:], actual_labels[trninds])
          
           idealobs_labels[tstinds,:] = np.expand_dims(labels,1)
           
#           normeucdist, labels, pooledvar = classifiers.norm_euc_dist(alldat[trninds,:], alldat[tstinds,:], actual_labels[trninds])
#          
#           eucdist_labels[tstinds,:] = np.expand_dims(labels,1)
#           
#           lin_reg = sklearn.linear_model.LinearRegression
#           
#           oledvar = classifiers.norm_euc_dist(alldat[trninds,:], alldat[tstinds,:], actual_labels[trninds])
#          
#           eucdist_labels[tstinds,:] = np.expand_dims(labels,1)
           
           
        acc = np.mean(idealobs_labels==actual_labels)
        errs = abs(idealobs_labels-actual_labels)
        
        print(' Ideal observer method:\n')
        print('overall accuracy: %.2f percent\n' % (acc*100))
        print('mean error: %.2f +/- %.2f degrees\n' % (np.mean(errs), np.std(errs)))
        
#        acc = np.mean(eucdist_labels==actual_labels)
#        errs = abs(eucdist_labels-actual_labels)
#        
#        print(' Normalized euclidean distance:\n')
#        print('overall accuracy: %.2f percent\n' % (acc*100))
#        print('mean error: %.2f +/- %.2f degrees\n' % (np.mean(errs), np.std(errs)))
        
        plt.figure()
        plt.scatter(actual_labels, idealobs_labels)
        plt.title('%s - %s, all stims\n Gaussian Ideal Observer Classifier' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('Actual Orientation')
        plt.ylabel('Predicted Orientation')
        
#        plt.figure()
#        plt.scatter(actual_labels, eucdist_labels)
#        plt.title('%s - %s, all stims\n Normalized Euclidean Distance Classifier' % (layer_labels[ww1], timepoint_labels[ww2]))
#        plt.xlabel('Actual Orientation')
#        plt.ylabel('Predicted Orientation')
        
#%% linear discriminant classification
plt.close('all')
nn=0
layers2plot = np.arange(0,nLayers,6)

# prepare for random cross-validation
ntrials = np.shape(allw[0][0])[0]
prop_test = 0.10;
assert np.mod(ntrials*prop_test,1)==0
ntest = int(ntrials*prop_test)
nCV = int(1/prop_test)

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        
        alldat = allw[ww1][nn][ww2]
        
        print('processing %s %s\n' % (layer_labels[ww1],timepoint_labels[ww2]) )
 
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        disc = np.zeros(np.shape(un))
        
        lindisc_labels = np.zeros(np.shape(actual_labels))
#        eucdist_labels = np.zeros(np.shape(actual_labels))
        
        randinds = np.arange(0,ntrials)
        np.random.shuffle(randinds)
        randinds = np.reshape(randinds, [ntest, nCV])
        for cv in range(nCV):
#        for cv in range(0,1):
            
           print('cross-validation fold %d of %d\n' % (cv, nCV))
            
           tstinds = randinds[:,cv]
           trninds = np.setdiff1d(randinds, tstinds, assume_unique=True)
           
           disc = discriminant_analysis.LinearDiscriminantAnalysis()
           disc = disc.fit(alldat[trninds,:], actual_labels[trninds])
           labels = disc.predict(alldat[tstinds,:])
          
           lindisc_labels[tstinds,:] = np.expand_dims(labels,1)
 
        acc = np.mean(lindisc_labels==actual_labels)
        errs = abs(lindisc_labels-actual_labels)
        
        print(' Linear discriminant classifier:\n')
        print('overall accuracy: %.2f percent\n' % (acc*100))
        print('mean error: %.2f +/- %.2f degrees\n' % (np.mean(errs), np.std(errs)))
  
        plt.figure()
        plt.scatter(actual_labels, lindisc_labels)
        plt.title('%s - %s, all stims\n linear discriminant classifier' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('Actual Orientation')
        plt.ylabel('Predicted Orientation')
        
#%% linear regression
plt.close('all')
nn=0
layers2plot = np.arange(0,nLayers,6)

# prepare for random cross-validation
ntrials = np.shape(allw[0][0])[0]
prop_test = 0.10;
assert np.mod(ntrials*prop_test,1)==0
ntest = int(ntrials*prop_test)
nCV = int(1/prop_test)

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        
        alldat = allw[ww1][nn][ww2]
        
        print('processing %s %s\n' % (layer_labels[ww1],timepoint_labels[ww2]) )
 
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        disc = np.zeros(np.shape(un))
        
        linreg_labels = np.zeros(np.shape(actual_labels))
#        eucdist_labels = np.zeros(np.shape(actual_labels))
        
        randinds = np.arange(0,ntrials)
        np.random.shuffle(randinds)
        randinds = np.reshape(randinds, [ntest, nCV])
        for cv in range(nCV):
#        for cv in range(0,1):
            
           print('cross-validation fold %d of %d\n' % (cv, nCV))
            
           tstinds = randinds[:,cv]
           trninds = np.setdiff1d(randinds, tstinds, assume_unique=True)
           
           reg = sklearn.linear_model.LinearRegression()
           reg = reg.fit(X=alldat[trninds,:], y=actual_labels[trninds])
           labels = reg.predict(alldat[tstinds,:])
          
           linreg_labels[tstinds,:] = labels
 
#        acc = np.mean(lindisc_labels==actual_labels)
        errs = abs(linreg_labels-actual_labels)
        
        print(' Linear regression:\n')
#        print('overall accuracy: %.2f percent\n' % (acc*100))
        print('mean error: %.2f +/- %.2f degrees\n' % (np.mean(errs), np.std(errs)))
  
        plt.figure()
        plt.scatter(actual_labels, linreg_labels)
        plt.title('%s - %s, all stims\n linear regression' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('Actual Orientation')
        plt.ylabel('Predicted Orientation')
        plt.axis('equal')
#        plt.xlim([0,180])
#        plt.ylim([0,180])
        plt.plot([0,180],[0,180],color='k')
        
#%% circular-circular regression
plt.close('all')
nn=0
layers2plot = np.arange(0,nLayers,6)

# prepare for random cross-validation
ntrials = np.shape(allw[0][0])[0]
prop_test = 0.10;
assert np.mod(ntrials*prop_test,1)==0
ntest = int(ntrials*prop_test)
nCV = int(1/prop_test)

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        
        alldat = allw[ww1][nn][ww2]
        
        print('processing %s %s\n' % (layer_labels[ww1],timepoint_labels[ww2]) )
 
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        disc = np.zeros(np.shape(un))
        
        linreg_labels = np.zeros(np.shape(actual_labels))
#        eucdist_labels = np.zeros(np.shape(actual_labels))
        
        randinds = np.arange(0,ntrials)
        np.random.shuffle(randinds)
        randinds = np.reshape(randinds, [ntest, nCV])
        for cv in range(nCV):
#        for cv in range(0,1):
            
           print('cross-validation fold %d of %d\n' % (cv, nCV))
            
           tstinds = randinds[:,cv]
           trninds = np.setdiff1d(randinds, tstinds, assume_unique=True)
           
           reg = regression.CCTrigonometricPolynomialRegression()
           reg = reg.train(alldat[trninds,:], actual_labels[trninds])
           labels = reg.test(alldat[tstinds,:])
          
           linreg_labels[tstinds,:] = labels
 
#        acc = np.mean(lindisc_labels==actual_labels)
        errs = abs(linreg_labels-actual_labels)
        
        print(' Linear regression:\n')
#        print('overall accuracy: %.2f percent\n' % (acc*100))
        print('mean error: %.2f +/- %.2f degrees\n' % (np.mean(errs), np.std(errs)))
  
        plt.figure()
        plt.scatter(actual_labels, linreg_labels)
        plt.title('%s - %s, all stims\n linear regression' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('Actual Orientation')
        plt.ylabel('Predicted Orientation')
        plt.axis('equal')
#        plt.xlim([0,180])
#        plt.ylim([0,180])
        plt.plot([0,180],[0,180],color='k')
        