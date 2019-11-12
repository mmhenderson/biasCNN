#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

root = '/usr/local/serenceslab/maggie/tensorflow/novel_objects/';
import os

os.chdir(root)

import numpy as np
import matplotlib.pyplot as plt

import scipy

from sklearn import decomposition
 
from sklearn import manifold
 
import sklearn
 
import classifiers


weight_path_before = os.path.join(root, 'weights', 'nasnet_grating_orient_sf_short')
weight_path_after = os.path.join(root, 'weights', 'nasnet_grating_orient_sf_long')
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

#%% calculate psychometric curve
plt.close('all')
min_diff_deg = 1
max_diff_deg = 10
pthresh = 0.01

#xlims = [0,46]
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        print('processing %s %s\n' % (layer_labels[ww1],timepoint_labels[ww2]) )
 
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        disc = np.zeros(np.shape(un))
        
        
        # first find the discriminability of all gratings with this exact label (should be close to zero)
        for ii in np.arange(0,nOri, 1):
        
            inds = np.where(actual_labels==un[ii])[0]    
            diffs1 = scipy.spatial.distance.pdist(allw[ww1][ww2][inds,:])
            diffs1 = np.triu(scipy.spatial.distance.squareform(diffs1),1)        
            # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
            #        assert not np.any(corrs1==1)        
            diffs1 = diffs1[np.where(diffs1!=0)]
                
            curve_t = np.zeros([max_diff_deg,1])
            curve_p = np.zeros([max_diff_deg,1])
            for bb in np.arange(min_diff_deg,max_diff_deg+1):

                # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
                inds_left = np.where(actual_labels==np.mod(un[ii]-bb, nOri))[0]        
                inds_right = np.where(actual_labels==np.mod(un[ii]+bb, nOri))[0]
                
                diffs2 = scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0))
                diffs2 = scipy.spatial.distance.squareform(diffs2)     
                diffs2 = diffs2[0:np.size(inds),np.size(inds):]
                diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
                
                diffs3 = scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0))
                diffs3 = scipy.spatial.distance.squareform(diffs3)      
                diffs3 = diffs3[0:np.size(inds),np.size(inds):]
                diffs3 = np.reshape(diffs3,np.prod(np.shape(diffs3)))
                    
                diffs_off = np.concatenate((diffs2,diffs3),axis=0)
        #        disc[ii] = np.mean(diffs_off)/(np.std(diffs_off)/np.sqrt(np.size(diffs_off)-1)) -\
        #                    np.mean(diffs1)/(np.std(diffs1)/np.sqrt(np.size(diffs1)-1))
#                pooled_var = (np.var(diffs_off)/np.size(diffs_off)) + (np.std(diffs_off)/np.size(diffs_off))
#                curve[bb-1] = (np.mean(diffs_off) - np.mean(diffs1))/np.sqrt(pooled_var)
                t, p = scipy.stats.ttest_ind(diffs_off, diffs1, axis=0,equal_var=False)
                curve_t[bb-1] = t
                curve_p[bb-1] = p
                

            if not np.shape(np.where(curve_p<pthresh))[1]==0:
                disc[ii] = np.where(curve_p<pthresh)[0][0] + min_diff_deg
            else:
                disc[ii] = max_diff_deg+1
                
                
        plt.figure()
        plt.scatter(un, disc)
        plt.title('Discrimination Threshold %s layer\n %s - all stims' % (layer_labels[ww1],timepoint_labels[ww2]))
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Smallest orientation difference that can be discrim. (deg)')
#%% calculate psychometric curve
plt.close('all')
min_diff_deg = 1
max_diff_deg = 10
pthresh = 0.01
sf2plot=np.arange(4,5)
type2plot=np.arange(1,2)

#xlims = [0,46]
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        
         for bb in sf2plot:            
            for tt in type2plot:
            
                myinds_bool = np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt))
        
        
                print('processing %s %s, %s SF=%.2f\n' % (layer_labels[ww1],timepoint_labels[ww2], stim_types[tt], sf_vals[bb]) )
         
                un,ia = np.unique(actual_labels, return_inverse=True)
                assert np.all(np.expand_dims(ia,1)==actual_labels)
                disc = np.zeros(np.shape(un))
                
                myinds_bool = np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt))
        
                # first find the discriminability of all gratings with this exact label (should be close to zero)
                
                
                
                for ii in np.arange(0,nOri, 1):
                
                    inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
                    diffs1 = scipy.spatial.distance.pdist(allw[ww1][ww2][inds,:])
                    diffs1 = np.triu(scipy.spatial.distance.squareform(diffs1),1)        
                    # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
                    #        assert not np.any(corrs1==1)        
                    diffs1 = diffs1[np.where(diffs1!=0)]
                        
                    curve_t = np.zeros([max_diff_deg,1])
                    curve_p = np.zeros([max_diff_deg,1])
                    for bb in np.arange(min_diff_deg,max_diff_deg+1):
        
                        # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
                        inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-bb, nOri), myinds_bool))[0]        
                        inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+bb, nOri), myinds_bool))[0]
                        
                        diffs2 = scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0))
                        diffs2 = scipy.spatial.distance.squareform(diffs2)     
                        diffs2 = diffs2[0:np.size(inds),np.size(inds):]
                        diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
                        
                        diffs3 = scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0))
                        diffs3 = scipy.spatial.distance.squareform(diffs3)      
                        diffs3 = diffs3[0:np.size(inds),np.size(inds):]
                        diffs3 = np.reshape(diffs3,np.prod(np.shape(diffs3)))
                            
                        diffs_off = np.concatenate((diffs2,diffs3),axis=0)
                #        disc[ii] = np.mean(diffs_off)/(np.std(diffs_off)/np.sqrt(np.size(diffs_off)-1)) -\
                #                    np.mean(diffs1)/(np.std(diffs1)/np.sqrt(np.size(diffs1)-1))
        #                pooled_var = (np.var(diffs_off)/np.size(diffs_off)) + (np.std(diffs_off)/np.size(diffs_off))
        #                curve[bb-1] = (np.mean(diffs_off) - np.mean(diffs1))/np.sqrt(pooled_var)
                        t, p = scipy.stats.ttest_ind(diffs_off, diffs1, axis=0,equal_var=False)
                        curve_t[bb-1] = t
                        curve_p[bb-1] = p
                        
        
                    if not np.shape(np.where(curve_p<pthresh))[1]==0:
                        disc[ii] = np.where(curve_p<pthresh)[0][0] + min_diff_deg
                    else:
                        disc[ii] = max_diff_deg+1
                        
                        
                plt.figure()
                plt.scatter(un, disc)
                plt.title('Discrimination Threshold %s layer\n %s - all stims' % (layer_labels[ww1],timepoint_labels[ww2]))
                plt.xlabel('actual orientation of grating')
                plt.ylabel('Smallest orientation difference that can be discrim. (deg)')

#%% calculate psychometric curve
plt.close('all')
min_diff_deg = 1
max_diff_deg = 45
xlims = [0,46]
for ww1 in layers2plot:
    for ww2 in timepts2plot:
 
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        disc = np.zeros(np.shape(un))
        
        plt.figure()
        legend_labs = ['zero']
        for ii in np.arange(0,180, 45):
        
            curve = np.zeros([max_diff_deg,1])
            for bb in np.arange(min_diff_deg,max_diff_deg+1):
                # first find the discriminability of all gratings with this exact label (should be close to zero)
                inds = np.where(actual_labels==un[ii])[0]    
                diffs1 = scipy.spatial.distance.pdist(allw[ww1][ww2][inds,:])
                diffs1 = np.triu(scipy.spatial.distance.squareform(diffs1),1)        
                # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
        #        assert not np.any(corrs1==1)        
                diffs1 = diffs1[np.where(diffs1!=0)]
                
                
                # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
                inds_left = np.where(actual_labels==np.mod(un[ii]-bb, nOri))[0]        
                inds_right = np.where(actual_labels==np.mod(un[ii]+bb, nOri))[0]
                
                diffs2 = scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0))
                diffs2 = scipy.spatial.distance.squareform(diffs2)     
                diffs2 = diffs2[0:np.size(inds),np.size(inds):]
                diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
                
                diffs3 = scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0))
                diffs3 = scipy.spatial.distance.squareform(diffs3)      
                diffs3 = diffs3[0:np.size(inds),np.size(inds):]
                diffs3 = np.reshape(diffs3,np.prod(np.shape(diffs3)))
                    
                diffs_off = np.concatenate((diffs2,diffs3),axis=0)
        #        disc[ii] = np.mean(diffs_off)/(np.std(diffs_off)/np.sqrt(np.size(diffs_off)-1)) -\
        #                    np.mean(diffs1)/(np.std(diffs1)/np.sqrt(np.size(diffs1)-1))
                curve[bb-1] = (np.mean(diffs_off) - np.mean(diffs1))
                
            
        
            plt.scatter(np.arange(min_diff_deg,max_diff_deg+1), curve)
            legend_labs.append('%s deg' % un[ii])
          
        plt.xlim(xlims)
        plt.plot(xlims, [0,0])
        plt.title('Psychometric curves for \n%s layer, %s - all stims' % (layer_labels[ww1],timepoint_labels[ww2]))
        plt.xlabel('orientation difference')
        plt.ylabel('discriminability (Euc. dist difference)')
        plt.legend(legend_labs)
