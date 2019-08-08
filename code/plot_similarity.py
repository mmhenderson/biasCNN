#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn import decomposition 
from matplotlib import cm

   
#%% get the data ready to go...then can run any below cells independently.

#model_str = 'inception_oriTst1';
model_str = 'vgg16_oriTst1';

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
                
#%% PCA , plotting pts by orientation
plt.close('all')
layers2plot = np.arange(6,7)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0]

clist = cm.plasma(np.linspace(0,1,12))

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            
#            pca = decomposition.PCA(n_components = 4)
#            
#            weights_reduced = pca.fit_transform(allw[ww1][ww2])
            weights_reduced = allw[ww1][ww2]
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
            plt.title('PC4 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
#%% PCA , plotting pts by orientation
plt.close('all')
layers2plot = np.arange(0,10,1)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0]

#clist = cm.plasma(np.linspace(0,1,12))
c_map = cm.get_cmap('plasma')
markers = ['^','+','o','x']
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            
#            pca = decomposition.PCA(n_components = 4)
#            
#            weights_reduced = pca.fit_transform(allw[ww1][ww2])
            
            weights_reduced = allw[ww1][ww2]
#            nBins = int(12)
#            nPerBin = int(180/nBins)
#            binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
#            
#            legend_labs = [];
            myinds = np.where(noiselist==nn)[0]
                
            plt.figure()
            for sf in range(nSF):
                myinds = np.where(np.logical_and(phaselist==1,np.logical_and(sflist==sf, noiselist==nn)))[0]
             
                sc = plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],
                                 c=actual_labels[myinds,0],
                                 vmin = 0,vmax = 180, cmap=c_map,marker=markers[sf])
                
#            plt.figure(1)
            ax = plt.gca()               
            plt.title('PC 2 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.colorbar(sc,ticks=[0,45,90,135,180])
            
            figname = os.path.join(figfolder, '%s_zeronoise_PC1_vs_PC2.eps' % (layer_labels[ww1]))
            plt.savefig(figname, format='eps')
#            plt.legend(legend_labs, bbox_to_anchor = (1,1))
#            box = ax.get_position()
#            ax.set_position([box.x0 , box.y0,
#                             box.width*0.80, box.height])
#            
#            plt.figure(2)           
#            ax = plt.gca()         
#            plt.title('PC 3 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
#            plt.xlabel('PC1')
#            plt.ylabel('PC3')
#            plt.legend(legend_labs,bbox_to_anchor = (1,1))
#            box = ax.get_position()
#            ax.set_position([box.x0 , box.y0,
#                             box.width*0.80, box.height])
#                                   
#            plt.figure(3)            
#            ax = plt.gca()           
#            plt.title('PC4 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
#            plt.xlabel('PC1')
#            plt.ylabel('PC4')
#            plt.legend(legend_labs,bbox_to_anchor = (1,1))
#            box = ax.get_position()
#            ax.set_position([box.x0 , box.y0,
#                             box.width*0.80, box.height])
#            

#%% PCA , plotting pts by spatial freq
plt.close('all')
layers2plot = [4]
timepts2plot = [0,1]
noiselevels2plot = [0]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
        
         
            pca = decomposition.PCA(n_components = 4)
        
            weights_reduced = pca.fit_transform(allw[ww1][ww2])
            
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
        
        weights_reduced = pca.fit_transform(allw[ww1][ww2])
       
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
        
        weights_reduced = pca.fit_transform(allw[ww1][ww2])
       
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = [];
        for bb in range(4):
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
     
            corrmat = np.corrcoef(allw[ww1][ww2])
           
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds = np.where(np.logical_and(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)), noiselist==nn))[0]
                    plt.figure()
                    plt.pcolormesh(corrmat[myinds,:][:,myinds])
                    plt.title('Correlations for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                     
                    plt.colorbar()
                    plt.xticks(np.arange(0,180*nPhase+1, nPhase*tick_spacing),np.arange(0,180+1,tick_spacing))
                    plt.yticks(np.arange(0,180*nPhase+1, nPhase*tick_spacing),np.arange(0,180+1,tick_spacing))
        
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
                    distmat = scipy.spatial.distance.pdist(allw[ww1][ww2][myinds,:], 'euclidean')
                    distmat = scipy.spatial.distance.squareform(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
#                    plt.clim([0,4000])
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,180*nPhase+1, nPhase*tick_spacing),np.arange(0,180+1,tick_spacing))
                    plt.yticks(np.arange(0,180*nPhase+1, nPhase*tick_spacing),np.arange(0,180+1,tick_spacing))
        
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
    
                            vals = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds1,:],  allw[ww1][ww2][inds2,:]),0)))      
            
                            
                            
                            distmat[uu1,uu2] = np.mean(vals)
                            distmat[uu2,uu1] = np.mean(vals)
                    
#                    distmat = scipy.spatial.distance.pdist(allw[ww1][nn][ww2][myinds,:], 'euclidean')
#                    distmat = scipy.spatial.distance.squareform(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
                    plt.clim([0,500])
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,180, tick_spacing),np.arange(0,180+1,tick_spacing))
                    plt.yticks(np.arange(0,180, tick_spacing),np.arange(0,180+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


