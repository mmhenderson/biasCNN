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

#import pycircstat

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

#%% plot predicted vs actual labels

un = np.unique(actual_labels)
avg_pred = np.zeros(np.shape(un))
std_pred = np.zeros(np.shape(un))
avg_pred_before = np.zeros(np.shape(un))
std_pred_before = np.zeros(np.shape(un))

for uu in range(len(un)):
    avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(actual_labels==un[uu])], high=180,low=0)
    std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(actual_labels==un[uu])], high=180,low=0)
    avg_pred_before[uu] = scipy.stats.circmean(pred_labels_before[np.where(actual_labels==un[uu])], high=180,low=0)
    std_pred_before[uu] = scipy.stats.circstd(pred_labels_before[np.where(actual_labels==un[uu])], high=180,low=0)
    
plt.figure()
plt.scatter(un,avg_pred_before)

plt.scatter(un,avg_pred)

plt.axis('equal')
plt.xlim([0,180])
plt.ylim([0,180])

plt.title('Predicted labels versus actual labels, all stims')
plt.legend(['before retraining','after retraining'])
plt.plot([0,180],[0,180],'k-')
plt.plot([90,90],[0,180],'k-')

#%% plot variability in predictions

un = np.unique(actual_labels)
avg_pred = np.zeros(np.shape(un))
std_pred = np.zeros(np.shape(un))
avg_pred_before = np.zeros(np.shape(un))
std_pred_before = np.zeros(np.shape(un))

for uu in range(len(un)):
    avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(actual_labels==un[uu])], high=179,low=0)
    std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(actual_labels==un[uu])], high=179,low=0)
    avg_pred_before[uu] = scipy.stats.circmean(pred_labels_before[np.where(actual_labels==un[uu])], high=179,low=0)
    std_pred_before[uu] = scipy.stats.circstd(pred_labels_before[np.where(actual_labels==un[uu])], high=179,low=0)
    
plt.figure()
#plt.scatter(un,avg_pred_before)
#plt.scatter(un,avg_pred)
plt.errorbar(un,avg_pred_before,std_pred_before)
plt.errorbar(un,avg_pred,std_pred)

plt.axis('equal')
plt.xlim([0,180])
#plt.ylim([0,180])

plt.title('Predicted labels versus actual labels, all stims')
plt.legend(['before retraining','after retraining'])
plt.plot([0,180],[0,180],'k-')
plt.plot([90,90],[0,180],'k-')


#%% plot predicted vs actual labels - within each stim type and spat freq

for tt in range(len(stim_types)):
    for bb in range(len(sf_vals)):

        myinds_bool = np.logical_and(typelist==tt, sflist==bb)
        
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
        plt.title('After retraining: predicted labels versus actual labels, %s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))

#%% calculate discriminability at each point in orientation space

for ww1 in layers2plot:
    for ww2 in timepts2plot:
 
        un,ia = np.unique(actual_labels, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==actual_labels)
        disc = np.zeros(np.shape(un))
        
        for ii in np.arange(0,np.size(un)):
        
            # first find the discriminability of all gratings with this exact label (should be close to zero)
            inds = np.where(actual_labels==un[ii])[0]    
            diffs1 = scipy.spatial.distance.pdist(allw[ww1][ww2][inds,:])
            diffs1 = np.triu(scipy.spatial.distance.squareform(diffs1),1)        
            # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
    #        assert not np.any(corrs1==1)        
            diffs1 = diffs1[np.where(diffs1!=0)]
            
            
            # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
            inds_left = np.where(actual_labels==np.mod(un[ii]-1, nOri))[0]        
            inds_right = np.where(actual_labels==np.mod(un[ii]+1, nOri))[0]
            
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
            disc[ii] = (np.mean(diffs_off) - np.mean(diffs1))
            
            
        plt.figure()
        plt.scatter(un, disc)
        plt.title('Discriminability for %s layer, %s - all stims' % (layer_labels[ww1],timepoint_labels[ww2]))
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Euc. dist) from neighbors')
#%% Discriminability within each envelope and SF
    
plt.close('all')
layers2plot = np.arange(0,nLayers,6)
#sf2plot = np.arange(4,5)
#type2plot = np.arange(1,2)
sf2plot = np.arange(0,nSF)
type2plot = np.arange(0,nType)

for ww1 in layers2plot:
    for ww2 in timepts2plot:    
        
        plt.figure()
        xx=1
        for bb in sf2plot:            
            for tt in type2plot:
            
                myinds_bool = np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt))
        
                un,ia = np.unique(actual_labels, return_inverse=True)
                assert np.all(np.expand_dims(ia,1)==actual_labels)
                disc = np.zeros(np.shape(un))
                
                for ii in np.arange(0,np.size(un)):
                
                    # first find the discriminability of all gratings with this exact label (should be close to zero)
                    inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
                    diffs1 = np.triu(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(allw[ww1][ww2][inds,:])),1)        
                    # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
            #        assert not np.any(corrs1==1)        
                    diffs1 = diffs1[np.where(diffs1!=0)]
            
                    
                    # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
                    inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, nOri), myinds_bool))[0]        
                    inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, nOri), myinds_bool))[0]
                    
                    diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0)))      
                    diffs2 = diffs2[0:np.size(inds),np.size(inds):]
                    diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
            
                    diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0)))      
                    diffs3 = diffs3[0:np.size(inds),np.size(inds):]
                    diffs3 = np.reshape(diffs3, np.prod(np.shape(diffs3)))
                
                    disc[ii] =  np.mean(np.concatenate((diffs2,diffs3),axis=0)) - np.mean(diffs1)
                    
                    
                plt.subplot(len(sf2plot), len(type2plot),xx)
#                plt.figure()
                xx=xx+1
                plt.scatter(un, disc)
    #            plt.title('Discriminability versus orientation - before training')
                plt.title('%s, SF=%.2f' % (stim_types[tt],sf_vals[bb]))
                
                if bb==len(sf2plot)-1:
                    plt.xlabel('actual orientation of grating')
                    plt.ylabel('discriminability (Euc. dist) from neighbors')
                else:
                    plt.xticks([])
                     
        plt.suptitle('Discriminability for %s layer, %s' % (layer_labels[ww1],timepoint_labels[ww2]))
    #            
#%% Bias across all stims

plt.close('all')
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
            diffs1 = np.triu(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(allw[ww1][ww2][inds,:])),1)        
            # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
    #        assert not np.any(corrs1==1)        
            diffs1 = diffs1[np.where(diffs1!=0)]
            
            
            # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
            inds_left = np.where(actual_labels==np.mod(un[ii]-1, nOri))[0]        
            inds_right = np.where(actual_labels==np.mod(un[ii]+1, nOri))[0]
            
            diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0)))      
            diffs2 = diffs2[0:np.size(inds),np.size(inds):]
            diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
            
            diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0)))      
            diffs3 = diffs3[0:np.size(inds),np.size(inds):]
            diffs3 = np.reshape(diffs3,np.prod(np.shape(diffs3)))
                
            diffs_off = np.concatenate((diffs2,diffs3),axis=0)
    #        disc[ii] = np.mean(diffs_off)/(np.std(diffs_off)/np.sqrt(np.size(diffs_off)-1)) -\
    #                    np.mean(diffs1)/(np.std(diffs1)/np.sqrt(np.size(diffs1)-1))
            bias[ii] = np.mean(diffs2) - np.mean(diffs3)
            
         
            
        plt.figure()
        plt.scatter(un, bias)
        plt.title('bias for %s layer, %s - all stims' % (layer_labels[ww1],timepoint_labels[ww2]))
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (Euc. dist) from neighbors')
        ylims = [np.min(bias), np.max(bias)]
        plt.ylim(ylims)
        for ii in np.arange(0,181,45):
            plt.plot([ii,ii],ylims)

#%% Bias at each envelope and SF
plt.close('all')
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
                    diffs1 = np.triu(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(allw[ww1][ww2][inds,:])),1)        
                    # make sure we got rid of all the identity comparisons, these are all different exemplars (images) with the same orientation.
            #        assert not np.any(corrs1==1)        
                    diffs1 = diffs1[np.where(diffs1!=0)]
            
                    
                    # then find the discriminability of this label from nearest neighbor gratings (should be higher) 
                    inds_left = np.where(np.logical_and(actual_labels==np.mod(un[ii]-1, nOri), myinds_bool))[0]        
                    inds_right = np.where(np.logical_and(actual_labels==np.mod(un[ii]+1, nOri), myinds_bool))[0]
                    
                    diffs2 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_left,:]),0)))      
                    diffs2 = diffs2[0:np.size(inds),np.size(inds):]
                    diffs2 = np.reshape(diffs2,np.prod(np.shape(diffs2)))
            
                    diffs3 = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds,:], allw[ww1][ww2][inds_right,:]),0)))      
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
                     
        plt.suptitle('Bias for %s layer, %s' % (layer_labels[ww1],timepoint_labels[ww2]))
 

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
        