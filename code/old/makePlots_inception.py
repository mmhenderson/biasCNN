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
 
#import sklearn
 
#import classifiers

#root = '/usr/local/serenceslab/maggie/tensorflow/novel_objects/';
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict

noise_levels = [0, 0.2, 0.4, 0.6, 0.8]

weight_path_before = os.path.join(root, 'activations', 'inception_activations_short_reduced')
weight_path_after = os.path.join(root, 'activations', 'inception_activations_long_reduced')
#weight_path_before = os.path.join(root, 'weights', 'inception_v3_grating_orient_short')
#weight_path_after = os.path.join(root, 'weights', 'inception_v3_grating_orient_long')
#dataset_path = os.path.join(root, 'datasets', 'datasets_Grating_Orient_SF')

#layer_labels = ['Conv2d_1a_3x3', 'Conv2d_4a_3x3','Mixed_7c','logits']
timepoint_labels = ['before retraining','after retraining']

# list of all the endpoints in this network.
layer_labels = ['Conv2d_1a_3x3',
 'Conv2d_2a_3x3',
 'Conv2d_2b_3x3',
 'MaxPool_3a_3x3',
 'Conv2d_3b_1x1',
 'Conv2d_4a_3x3',
 'MaxPool_5a_3x3',
 'Mixed_5b',
 'Mixed_5c',
 'Mixed_5d',
 'Mixed_6a',
 'Mixed_6b',
 'Mixed_6c',
 'Mixed_6d',
 'Mixed_6e',
 'Mixed_7a',
 'Mixed_7b',
 'Mixed_7c',
 'AuxLogits',
 'AvgPool_1a',
 'PreLogits',
 'Logits',
 'Predictions']

#%% information about the stimuli. There are two types - first is a full field 
# sinusoidal grating (e.g. a rectangular image with the whole thing a grating)
# second is a gaussian windowed grating.
sf_vals = np.logspace(np.log10(0.2), np.log10(2),5)
stim_types = ['Full','Gaus']
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
    
    tmp = []
    
    for nn in range(np.size(noise_levels)):
#        print(nn)
#        print(noise_levels[nn])
        file = os.path.join(weight_path_before, 'noise%.2f' % noise_levels[nn], 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w1 = np.load(file)
        
        file = os.path.join(weight_path_after, 'noise%.2f' % noise_levels[nn], 'allStimsReducedWts_%s.npy' % layer_labels[ll])
        w2 = np.load(file)
        
        tmp.append([w1,w2])
        
    allw.append(tmp)
#    allw.append([w1])
    
nLayers = len(allw)
nTimepts = len(allw[0][0])
nNoiseLevels = len(allw[0])
# can change these if you want just a subset of plots made at a time
layers2plot = np.arange(0,nLayers,1)
#timepts2plot = np.arange(0,nTimepts,1)
timepts2plot = np.arange(0,1)

#%% load the predicted orientation labels from the re-trained network

num_batches = 80

all_labs = []

for nn in range(np.size(noise_levels)):

    file = os.path.join(weight_path_before, 'noise%.2f' %(noise_levels[nn]), 'allStimsLabsPredicted_Logits.npy')    
    labs1 = np.load(file)
     
    file = os.path.join(weight_path_after, 'noise%.2f' %(noise_levels[nn]), 'allStimsLabsPredicted_Logits.npy')    
    labs2 = np.load(file)
    
    all_labs.append([labs1,labs2])
  
#%%       
plt.close('all')
#actual_labels = np.mod(xlist,180)

                     
#%% PCA , plotting pts by orientation
plt.close('all')
layers2plot = np.arange(19,20)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0]
nBins = int(12)
clist = cm.plasma(np.linspace(0,1,nBins))

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for ww3 in noiselevels2plot:
            
            pca = decomposition.PCA(n_components = 4)
            
            weights_reduced = pca.fit_transform(allw[ww1][ww3][ww2])
            plt.figure()
#            plt.set_cmap('plasma')
            ax = plt.gca()
           
            nPerBin = int(180/nBins)
            binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
            
            legend_labs = [];
            for bb in range(nBins):   
                
                myinds = np.where(np.isin(actual_labels, binned_labs[bb,:]))[0]
            #    print(myinds)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],c=np.expand_dims(clist[bb],axis=0))
                legend_labs.append('%d - %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
                
            plt.title('PC 2 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[ww3]))
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend(legend_labs, bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
            plt.figure()
            plt.set_cmap('plasma')
            ax = plt.gca()
            nBins = int(12)
            nPerBin = int(180/nBins)
            binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
            
            legend_labs = [];
            for bb in range(nBins):   
                
                myinds = np.where(np.isin(actual_labels, binned_labs[bb,:]))[0]
            #    print(myinds)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2],c=np.expand_dims(clist[bb],axis=0))
                legend_labs.append('%d - %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
                
            plt.title('PC 3 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[ww3]))
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
            
            
            plt.figure()
            
            ax = plt.gca()
            nBins = int(12)
            nPerBin = int(180/nBins)
            binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
            
            legend_labs = [];
            for bb in range(nBins):   
                
                myinds = np.where(np.isin(actual_labels, binned_labs[bb,:]))[0]
            #    print(myinds)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3],c=np.expand_dims(clist[bb],axis=0))
                legend_labs.append('%d - %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
                
            plt.title('PC4 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[ww3]))
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
            var_expl = pca.explained_variance_ratio_
            plt.set_cmap('plasma')
#        plt.figure()
#        plt.scatter(np.arange(0,np.shape(var_expl)[0],1), var_expl)
#        plt.title('Percentage of variance explained by each component, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))

#%% PCA , plotting pts by spatial freq
#plt.close('all')
layers2plot = np.arange(19,20)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for ww3 in noiselevels2plot:
        
         
            pca = decomposition.PCA(n_components = 4)
        
            weights_reduced = pca.fit_transform(allw[ww1][ww3][ww2])
            plt.figure()
            ax = plt.gca()
            legend_labs = [];
            for bb in range(nSF):   
                for tt in range(nType):
                
                    myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                #    print(myinds)
                    plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1])
                    legend_labs.append('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
                    
            plt.title('PC2 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[ww3]))
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))          
            box = ax.get_position()
            
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
       
            # Put a legend below current axis
#            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#                      fancybox=True, shadow=True, ncol=5)

            
            plt.figure()
            ax = plt.gca()
            legend_labs = [];
            for bb in range(nSF):   
                for tt in range(nType):
                
                    myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                #    print(myinds)
                    plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2])
                    legend_labs.append('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
                    
            plt.title('PC3 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[ww3]))
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))          
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
            plt.figure()
            ax = plt.gca()
            legend_labs = [];
            for bb in range(nSF):   
                for tt in range(nType):
                
                    myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                #    print(myinds)
                    plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3])
                    legend_labs.append('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
                    
            plt.title('PC4 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[ww3]))
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))          
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
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

sf2plot = np.arange(0,1)
type2plot = np.arange(1,2)
layers2plot = np.arange(13,14)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0,1,2,3,4]

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for ww3 in noiselevels2plot:
     
            corrmat = np.corrcoef(allw[ww1][ww3][ww2])
           
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                    plt.figure()
                    plt.pcolormesh(corrmat[myinds,:][:,myinds])
                    plt.title('Correlations for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[ww3],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                     
                    plt.colorbar()
                    plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
                    plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


#%% Compare output before/after training...all but last layers should be identical.
                    
plt.close('all')
tick_spacing = 45

sf2plot = np.arange(0,1)
type2plot = np.arange(1,2)
layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,2)
noiselevels2plot = [0]

plt.close('all')
for ww1 in layers2plot:
#    for ww2 in timepts2plot:
    for ww3 in noiselevels2plot:
 
#        corrmat = np.corrcoef(allw[ww1][ww3][0], allw[ww1][ww3][1])
       
        for bb in sf2plot:
            for tt in type2plot:
            
                myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                
                corrmat = np.corrcoef(allw[ww1][ww3][0][myinds,:], allw[ww1][ww3][1][myinds,:])
                
                plt.figure()
                plt.pcolormesh(corrmat)
#                plt.title('Correlations for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[ww3],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                 
                plt.colorbar()
#                plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
#                plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
#    
#                plt.xlabel('Grating 1 deg')
#                plt.ylabel('Grating 2 deg')


#%% DIST MATRIX, each spatial freq and type separately
plt.close('all')
tick_spacing = 45

layers2plot =np.arange(0,nLayers,6)

sf2plot = np.arange(0,1)
type2plot = np.arange(1,2)
layers2plot = np.arange(13,14)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0,1,2,3,4]

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for ww3 in noiselevels2plot:
 
            
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                    
                    distmat = scipy.spatial.distance.pdist(allw[ww1][ww3][ww2][myinds,:], 'euclidean')
                    distmat = scipy.spatial.distance.squareform(distmat)/np.max(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[ww3],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
                    plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


