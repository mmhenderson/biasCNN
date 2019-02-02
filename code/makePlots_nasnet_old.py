#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

root = '/usr/local/serenceslab/maggie/biasCNN/';
import os

os.chdir(root)


import numpy as np
import matplotlib.pyplot as plt

import scipy

from sklearn import decomposition
 
from sklearn import manifold
 
import sklearn
 
import classifiers

#root = '/usr/local/serenceslab/maggie/tensorflow/novel_objects/';

weight_path_before = os.path.join(root, 'activations', 'nasnet_grating_orient_sf_short')
weight_path_after = os.path.join(root, 'activations', 'nasnet_grating_orient_sf_long')
#weight_path_before = os.path.join(root, 'weights', 'inception_v3_grating_orient_short')
#weight_path_after = os.path.join(root, 'weights', 'inception_v3_grating_orient_long')
dataset_path = os.path.join(root, 'datasets', 'datasets_Grating_Orient_SF')

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
stim_types = ['Fullfield','Gaussian']
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

#%% load the data (already in reduced/PCA-d format)

allw = []

for ll in range(np.size(layer_labels)):
    file = os.path.join(weight_path_before, 'allStimsReducedWts_' + layer_labels[ll] +'.npy')
    w1 = np.load(file)
    
    file = os.path.join(weight_path_after, 'allStimsReducedWts_' + layer_labels[ll] +'.npy')
    w2 = np.load(file)
    
    allw.append([w1,w2])
#    allw.append([w1])
    
nLayers = len(allw)
nTimepts = len(allw[0])

# can change these if you want just a subset of plots made at a time
layers2plot = np.arange(0,nLayers,1)
#timepts2plot = np.arange(0,nTimepts,1)
timepts2plot = np.arange(0,1)

#%% load the predicted orientation labels from the re-trained network

num_batches = 80

for bb in np.arange(0,num_batches,1):

    file = os.path.join(weight_path_after, 'batch' + str(bb) + '_labels_predicted.npy')    
    labs = np.expand_dims(np.load(file),1)
 
    if bb==0:
        pred_labels = labs
    else:
        pred_labels = np.concatenate((pred_labels,labs),axis=0) 
 
    file = os.path.join(weight_path_before, 'batch' + str(bb) + '_labels_predicted.npy')    
    labs = np.expand_dims(np.load(file),1)
 
    if bb==0:
        pred_labels_before = labs
    else:
        pred_labels_before = np.concatenate((pred_labels_before,labs),axis=0) 
 
#%%       
plt.close('all')
#actual_labels = np.mod(xlist,180)

                      
#%% PCA , plotting pts by orientation
#plt.close('all')
layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        pca = decomposition.PCA(n_components = 4)
        
        weights_reduced = pca.fit_transform(allw[ww1][ww2])
        plt.figure()
         
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = [];
        for bb in range(nBins):   
            
            myinds = np.where(np.isin(actual_labels, binned_labs[bb,:]))[0]
        #    print(myinds)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1])
            legend_labs.append('%d through %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
            
        plt.title('PC 2 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(legend_labs)
        
        plt.figure()
         
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = [];
        for bb in range(nBins):   
            
            myinds = np.where(np.isin(actual_labels, binned_labs[bb,:]))[0]
        #    print(myinds)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2])
            legend_labs.append('%d through %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
            
        plt.title('PC 3 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        plt.legend(legend_labs)
        
        plt.figure()
         
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = [];
        for bb in range(nBins):   
            
            myinds = np.where(np.isin(actual_labels, binned_labs[bb,:]))[0]
        #    print(myinds)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3])
            legend_labs.append('%d through %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
            
        plt.title('PC4 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC4')
        plt.legend(legend_labs)
        
        var_expl = pca.explained_variance_ratio_
        
#        plt.figure()
#        plt.scatter(np.arange(0,np.shape(var_expl)[0],1), var_expl)
#        plt.title('Percentage of variance explained by each component, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))

#%% PCA , plotting pts by spatial freq
plt.close('all')
layers2plot = np.arange(2,3)
timepts2plot = np.arange(0,1)
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
     
        pca = decomposition.PCA(n_components = 4)
    
        weights_reduced = pca.fit_transform(allw[ww1][ww2])
        plt.figure()
    
        legend_labs = [];
        for bb in range(nSF):   
            for tt in range(nType):
            
                myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
            #    print(myinds)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1])
                legend_labs.append('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
                
        plt.title('PC 2 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(legend_labs)
        
        plt.figure()
    
        legend_labs = [];
        for bb in range(nSF):   
            for tt in range(nType):
            
                myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
            #    print(myinds)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2])
                legend_labs.append('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
                
        plt.title('PC 3 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        plt.legend(legend_labs)
        
        plt.figure()
    
        legend_labs = [];
        for bb in range(nSF):   
            for tt in range(nType):
            
                myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
            #    print(myinds)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3])
                legend_labs.append('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
                
        plt.title('PC 4 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC4')
        plt.legend(legend_labs)
        
        var_expl = pca.explained_variance_ratio_
        
#        plt.figure()
#        plt.scatter(np.arange(0,np.shape(var_expl)[0],1), var_expl)
#        plt.title('Percentage of variance explained by each component, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))

#%%   MDS BEFORE
# this block of code runs so slow that i've never made this plot...someday maybe
plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        print('processing %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        distmat = scipy.spatial.distance.pdist(allw[ww1][ww2], 'euclidean')
        distmat = scipy.spatial.distance.squareform(distmat)   
        
        mds = manifold.MDS(n_components = 2, dissimilarity = 'precomputed')
        mds_coords = mds.fit_transform(distmat)
        
        plt.figure()
         
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = []
        for bb in range(nBins):   
            
            myinds = np.where(np.isin(actual_labels, binned_labs[bb,:]))[0]
        #    print(actual_labels[myinds])
            plt.scatter(mds_coords[myinds,0], mds_coords[myinds,1])
            legend_labs.append('%d through %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
            
        plt.title('2D MDS representation, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('MDS axis 1')
        plt.ylabel('MDS axis 2')
        plt.legend(legend_labs)


#%% CORR MATRIX, each spatial freq and type separately
plt.close('all')
tick_spacing = 45

sf2plot = np.arange(4,5)
type2plot = np.arange(1,2)

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
 
        corrmat = np.corrcoef(allw[ww1][ww2])
       
        for bb in sf2plot:
            for tt in type2plot:
            
                myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                plt.figure()
                plt.pcolormesh(corrmat[myinds,:][:,myinds])
                plt.title('Correlations for %s layer, %s - %s, SF=%.2f' % (layer_labels[ww1],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                 
                plt.colorbar()
                plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
                plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
    
                plt.xlabel('Grating 1 deg')
                plt.ylabel('Grating 2 deg')

#%% DIST MATRIX, each spatial freq and type separately
plt.close('all')
tick_spacing = 45

layers2plot =np.arange(0,nLayers,6)

sf2plot = np.arange(4,5)
type2plot = np.arange(1,2)

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
 
        
        for bb in sf2plot:
            for tt in type2plot:
            
                myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                
                distmat = scipy.spatial.distance.pdist(allw[ww1][ww2][myinds,:], 'euclidean')
                distmat = scipy.spatial.distance.squareform(distmat)/np.max(distmat)
        
                
                plt.figure()
                plt.pcolormesh(distmat)
                plt.title('Euclidean distances for %s layer, %s\n%s, SF=%.2f' % (layer_labels[ww1],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                 
                plt.colorbar()
                plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
                plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
    
                plt.xlabel('Grating 1 deg')
                plt.ylabel('Grating 2 deg')


