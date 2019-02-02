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
 

#%%       
plt.close('all')
#actual_labels = np.mod(xlist,180)

                     
#%% PCA , plotting pts by orientation
plt.close('all')
layers2plot = np.arange(17,18)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0]

clist = cm.plasma(np.linspace(0,1,12))

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            
            pca = decomposition.PCA(n_components = 4)
            
            weights_reduced = pca.fit_transform(allw[ww1][0][ww2])
           
            nBins = int(12)
            nPerBin = int(180/nBins)
            binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
            
            legend_labs = [];
            for bb in range(nBins):   
                
                myinds = np.where(np.logical_and(np.isin(actual_labels, binned_labs[bb,:]), noiselist==nn))[0]
                plt.figure(1)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],c=np.expand_dims(clist[bb],axis=0))
                plt.figure(2)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2],c=np.expand_dims(clist[bb],axis=0))
                plt.figure(3)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3],c=np.expand_dims(clist[bb],axis=0))
                legend_labs.append('%d - %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
            
            plt.figure(1)
            ax = plt.gca()               
            plt.title('PC 2 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend(legend_labs, bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
            plt.figure(2)           
            ax = plt.gca()         
            plt.title('PC 3 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
                                   
            plt.figure(3)            
            ax = plt.gca()           
            plt.title('PC4 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[ww3]))
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            

#%% PCA , plotting pts by spatial freq
plt.close('all')
layers2plot = [2]
timepts2plot = [0,1]
noiselevels2plot = [0]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
        
         
            pca = decomposition.PCA(n_components = 4)
        
            weights_reduced = pca.fit_transform(allw[ww1][nn][ww2])
            
            legend_labs = [];
            for bb in range(nSF):   
                for tt in range(nType):
                
                    myinds = np.where(np.logical_and(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)), noiselist==nn))[0]
                    plt.figure(1+ww2*3)
                    plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1])
                    plt.figure(2+ww2*3)
                    plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2])
                    plt.figure(3+ww2*3)
                    plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3])
                    legend_labs.append('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
             
            plt.figure(1+ww2*3)
            ax = plt.gca()
            plt.title('PC2 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))          
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
      
            plt.figure(2+ww2*3)
            ax = plt.gca()               
            plt.title('PC3 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))          
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
            plt.figure(3+ww2*3)
            ax = plt.gca()
            legend_labs = [];
            plt.title('PC4 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))          
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
            
#%% PCA across all noise levels, plotting pts by orientation
            
plt.close('all')
layers2plot = [2]
timepts2plot = [0]

clist = cm.plasma(np.linspace(0,1,12))

for ww1 in layers2plot:
    for ww2 in timepts2plot:
#        for ww3 in noiselevels2plot:
            
        pca = decomposition.PCA(n_components = 4)
        
        weights_reduced = pca.fit_transform(allw[ww1][0][ww2])
       
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = [];
        for bb in range(nBins):   
             
            myinds = np.where(np.isin(actual_labels, binned_labs[bb,:]))[0]
        #    print(myinds)
            plt.figure(1)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],c=np.expand_dims(clist[bb],axis=0))
            plt.figure(2)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2],c=np.expand_dims(clist[bb],axis=0))
            plt.figure(3)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3],c=np.expand_dims(clist[bb],axis=0))
            
            legend_labs.append('%d - %d deg' % (actual_labels[myinds[0]], actual_labels[myinds[-1]]))
            
        plt.figure(1)
        plt.title('PC 2 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(legend_labs, bbox_to_anchor = (1,1))
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.80, box.height])
        
        plt.figure(2)        
        ax = plt.gca()
        plt.title('PC 3 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        plt.legend(legend_labs,bbox_to_anchor = (1,1))
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.80, box.height])
            
        plt.figure(3)
        ax = plt.gca()
        plt.title('PC4 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC4')
        plt.legend(legend_labs,bbox_to_anchor = (1,1))
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.80, box.height])
        

#%% PCA across all noise levels, plotting pts by SF/noise level
            
plt.close('all')
layers2plot = [2]
timepts2plot = [0]
tt=0

csteps=8
my_purples = np.expand_dims(cm.Purples(np.linspace(1,0,csteps)),2)
my_greens = np.expand_dims(cm.Greens(np.linspace(1,0,csteps)),2)
my_oranges = np.expand_dims(cm.Oranges(np.linspace(1,0,csteps)),2)
my_blues = np.expand_dims(cm.Blues(np.linspace(1,0,csteps)),2)

clist = np.concatenate((my_purples, my_greens, my_oranges, my_blues),2)



for ww1 in layers2plot:
    for ww2 in timepts2plot:
#        for ww3 in noiselevels2plot:
            
        pca = decomposition.PCA(n_components = 4)
        
        weights_reduced = pca.fit_transform(allw[ww1][0][ww2])
       
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = [];
        for bb in range(nSF):
            for nn in range(nNoiseLevels):
             
                myinds = np.where(np.all([sflist==bb, typelist==tt, noiselist==nn],0))[0]
            #    print(myinds)
                plt.figure(1)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],c=np.expand_dims(clist[nn,:,bb],axis=0))
                plt.figure(2)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2],c=np.expand_dims(clist[nn,:,bb],axis=0))
                plt.figure(3)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3],c=np.expand_dims(clist[nn,:,bb],axis=0))
                
                legend_labs.append('SF=%.2f,noise=%.2f' % (sf_vals[bb], noise_levels[nn]))
                
        plt.figure(1)
        plt.title('PC 2 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(legend_labs, bbox_to_anchor = (1,1))
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.70, box.height])
        
        plt.figure(2)        
        ax = plt.gca()
        plt.title('PC 3 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        plt.legend(legend_labs,bbox_to_anchor = (1,1))
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.70, box.height])
            
        plt.figure(3)
        ax = plt.gca()
        plt.title('PC4 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC4')
        plt.legend(legend_labs,bbox_to_anchor = (1,1))
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.70, box.height])
        

#%% CORR MATRIX, each spatial freq and type separately
plt.close('all')
tick_spacing = 45

sf2plot = [3]
type2plot = [0]
layers2plot = [13]
timepts2plot = [0]
noiselevels2plot = [0,1,2]

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
     
            corrmat = np.corrcoef(allw[ww1][0][ww2])
           
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds = np.where(np.logical_and(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)), noiselist==nn))[0]
                    plt.figure()
                    plt.pcolormesh(corrmat[myinds,:][:,myinds])
                    plt.title('Correlations for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                     
                    plt.colorbar()
                    plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
                    plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')



#%% DIST MATRIX, each spatial freq and type separately
plt.set_cmap('plasma')
plt.close('all')
tick_spacing = 45

sf2plot = [3]
type2plot = [0]
layers2plot = [13]
timepts2plot = [0]
noiselevels2plot = [0,1,2]

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
 
            
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds = np.where(np.logical_and(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)), noiselist==nn))[0]
                    distmat = scipy.spatial.distance.pdist(allw[ww1][0][ww2][myinds,:], 'euclidean')
                    distmat = scipy.spatial.distance.squareform(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
#                    plt.clim([0,4000])
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
                    plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


#%% An alternative way of plotting the discriminability (averaging in bins)
plt.set_cmap('plasma')
plt.close('all')
tick_spacing = 45

sf2plot = [3]
type2plot = [0]
layers2plot = [13]
timepts2plot = [0]
noiselevels2plot = [0,1,2]

un,ia = np.unique(actual_labels, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==actual_labels)

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
 
            
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds_bool = np.logical_and(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)), noiselist==nn)
                    
                    distmat = np.zeros([np.size(un),np.size(un)])
                    for uu1 in np.arange(0,np.size(un)):
                        for uu2 in np.arange(uu1,np.size(un)):
                            
                            inds1 = np.where(np.logical_and(actual_labels==un[uu1], myinds_bool))[0]    
                            inds2 = np.where(np.logical_and(actual_labels==un[uu2], myinds_bool))[0]    
    
                            vals = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][0][ww2][inds1,:],  allw[ww1][0][ww2][inds2,:]),0)))      
            
                            
                            
                            distmat[uu1,uu2] = np.mean(vals)
                            distmat[uu2,uu1] = np.mean(vals)
                    
#                    distmat = scipy.spatial.distance.pdist(allw[ww1][nn][ww2][myinds,:], 'euclidean')
#                    distmat = scipy.spatial.distance.squareform(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
                    plt.clim([0,500])
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,nOri, tick_spacing),np.arange(0,nOri+1,tick_spacing))
                    plt.yticks(np.arange(0,nOri, tick_spacing),np.arange(0,nOri+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


