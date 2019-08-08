#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""
#%% get the data ready to go...then can run any below cells independently.

%reset

root = '/usr/local/serenceslab/maggie/biasCNN/';
import os
os.chdir(os.path.join(root, 'code'))
%run load_data_nasnet_oriTst0

nVox2Use = 100
center_deg=90
n_chans=9

import matplotlib.pyplot as plt
from sklearn import decomposition 
from sklearn import manifold 
from sklearn import discriminant_analysis
import IEM
import sklearn 
import classifiers
#import pycircstat
import scipy
             
#%% PCA , plotting pts by orientation
plt.close('all')
layers2plot = np.arange(17,18)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0]

clist = cm.plasma(np.linspace(0,1,12))

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for ww3 in noiselevels2plot:
            
            pca = decomposition.PCA(n_components = 4)
            
            weights_reduced = pca.fit_transform(allw[ww1][ww3][ww2])
            plt.figure()
#            plt.set_cmap('plasma')
            ax = plt.gca()
            nBins = int(12)
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
plt.close('all')
layers2plot = np.arange(2,3)
timepts2plot = np.arange(1,2)
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



#%% DIST MATRIX, each spatial freq and type separately
plt.set_cmap('plasma')
plt.close('all')
tick_spacing = 45

layers2plot =np.arange(0,nLayers,6)

sf2plot = np.arange(4,5)
type2plot = np.arange(1,2)
layers2plot = np.arange(2,3)
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
                    distmat = scipy.spatial.distance.squareform(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
                    plt.clim([0,4000])
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[ww3],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
                    plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


#%% DIST MATRIX, each spatial freq and type separately
plt.set_cmap('plasma')
plt.close('all')
tick_spacing = 45

layers2plot =np.arange(0,nLayers,6)

sf2plot = np.arange(4,5)
type2plot = np.arange(1,2)
layers2plot = np.arange(2,3)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0,1,2,3,4]

un,ia = np.unique(actual_labels, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==actual_labels)

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for ww3 in noiselevels2plot:
 
            
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds = np.where(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)))[0]
                    
                    distmat = np.zeros([np.size(un),np.size(un)])
                    for uu1 in np.arange(0,np.size(un)):
                        for uu2 in np.arange(uu1,np.size(un)):
                            
                            inds1 = np.where(np.logical_and(actual_labels==un[uu1], myinds))[0]    
                            inds2 = np.where(np.logical_and(actual_labels==un[uu2], myinds))[0]    
    
                            vals = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][nn][ww2][inds1,:],  allw[ww1][nn][ww2][inds2,:]),0)))      
            
                            
                            
                            distmat[uu1,uu2] = np.mean(vals)
                            distmat[uu2,uu1] = np.mean(vals)
                    
                    distmat = scipy.spatial.distance.pdist(allw[ww1][ww3][ww2][myinds,:], 'euclidean')
                    distmat = scipy.spatial.distance.squareform(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
                    plt.clim([0,4000])
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[ww3],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
                    plt.yticks(np.arange(0,nOri*nPhase+1, nPhase*tick_spacing),np.arange(0,nOri+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


