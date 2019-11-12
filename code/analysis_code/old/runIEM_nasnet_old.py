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

import scipy

weight_path_before = os.path.join(root, 'activations', 'nasnet_short_reduced')
weight_path_after = os.path.join(root, 'activations', 'nasnet_long_reduced'
#weight_path_before = os.path.join(root, 'weights', 'inception_v3_grating_orient_short')
#weight_path_after = os.path.join(root, 'weights', 'inception_v3_grating_orient_long')
dataset_path = os.path.join(root, 'datasets', 'testing')

#layer_labels = ['Conv2d_1a_3x3', 'Conv2d_4a_3x3','Mixed_7c','logits']
timepoint_labels = ['before retraining','after retraining']

layer_labels = []
#for cc in range(17):
#    layer_labels.append('Cell_%d' % (cc+1))
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
#    file = os.path.join(weight_path_before, 'allStimsReducedWts_' + layer_labels[ll] +'.npy')
#    w1 = np.load(file)
    
    file = os.path.join(weight_path_after, 'allStimsReducedWts_' + layer_labels[ll] +'.npy')
    w2 = np.load(file)
    w1=w2
    allw.append([w1,w2])
    
    print(ll)
    print(np.shape(w1))
    print(np.shape(w2))
    
#    allw.append([w1])
    
nLayers = len(allw)
nTimepts = len(allw[0])

# can change these if you want just a subset of plots made at a time
layers2plot = np.arange(0,nLayers,1)
#timepts2plot = np.arange(0,nTimepts,1)
timepts2plot = np.arange(0,1)

#%% load the predicted orientation labels from the re-trained network

#num_batches = 80
#
#for bb in np.arange(0,num_batches,1):
#
#    file = os.path.join(weight_path_after, 'batch' + str(bb) + '_labels_predicted.npy')    
#    labs = np.expand_dims(np.load(file),1)
# 
#    if bb==0:
#        pred_labels = labs
#    else:
#        pred_labels = np.concatenate((pred_labels,labs),axis=0) 
# 
#    file = os.path.join(weight_path_before, 'batch' + str(bb) + '_labels_predicted.npy')    
#    labs = np.expand_dims(np.load(file),1)
# 
#    if bb==0:
#        pred_labels_before = labs
#    else:
#        pred_labels_before = np.concatenate((pred_labels_before,labs),axis=0) 
# 
#%%       
plt.close('all')
#actual_labels = np.mod(xlist,180)
#%% run the IEM - within each stimulus type and sf separately

plt.close('all')

#layers2plot = np.arange(0,nLayers,6)
layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

xx=np.arange(0,180,1)
ylims = [-1,1]

nVox2Use=100

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        plt.figure()
        ii=0      
        for sf in range(nSF):
             for tt in range(nType):
                ii=ii+1
                plt.subplot(nSF,nType,ii)
 
                # first find the discriminability of all gratings with this exact label (should be close to zero)
                inds = np.where(np.logical_and(typelist==tt, sflist==sf))[0]
                       
                ori_labs = actual_labels[inds]
                center_deg=90
                n_chans=9
                n_folds = 10
                
                alldat = allw[ww1][ww2][inds,0:nVox2Use]

                chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,n_folds,n_chans)
                
                average_recons = IEM.shift_and_average(chan_resp_all,ori_labs,center_deg);
                
                plt.plot(xx,average_recons)
                plt.ylim(ylims)
                plt.plot([center_deg,center_deg], ylims)
                plt.title('%s, SF=%.2f' % (stim_types[tt],sf_vals[sf]))
                if sf==nSF-1:
                    plt.xlabel('Orientation Channel (deg)')
                    plt.ylabel('Channel Activation Weight')
                else:
                    plt.tick_params(axis='x', bottom=False,labelbottom = False)
#                    plt.xlabel('')
#                    plt.xticks(ticks=None,labels=None)
#                    plt.ylabel('')
        plt.suptitle('Average reconstruction, trn/test within stimulus type and SF. \nWeights from %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
#%% train the IEM across all trials
        
plt.close('all')
#plt.figure()
#layers2plot = np.arange(0,nLayers,6)
layers2plot = np.arange(0,nLayers,6)
timepts2plot = np.arange(0,2)
ylims = [-5,0]
nVox2Use = 100
ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        ii=ii+1
        plt.subplot(np.size(layers2plot),np.size(timepts2plot),ii)
    
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
#        n_folds = 10
        
#        alldat = allw[ww1][ww2]
        alldat = allw[ww1][ww2][:,0:nVox2Use]
#        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
#        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,10,9)
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
         
        average_recons = IEM.shift_and_average(chan_resp_all,ori_labs,center_deg);
          
        plt.plot(xx,average_recons)
        plt.ylim(ylims)
        plt.plot([center_deg,center_deg], ylims)
        plt.title('%s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
        if ww1==np.max(layers2plot):
            plt.xlabel('Orientation Channel (deg)')
            plt.ylabel('Channel Activation Weight')
        else:
            plt.tick_params(axis='x', bottom=False,labelbottom = False)
            
        if (np.size(timepts2plot)>1 and ww2==np.max(timepts2plot)):
            
            plt.tick_params(axis='y', left=False,labelleft = False)
            plt.ylabel(None)
    
plt.suptitle('Average reconstructions, leave one stimulus type out')
#%% train all stimuli, separate the test set according to type/sf

plt.close('all')

#layers2plot = np.arange(0,nLayers,6)
layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

xx=np.arange(0,180,1)
ylims = [-5,0]

nVox2Use = 100

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        plt.figure()
        ii=0
        
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
#        n_folds = 20
        
        # run the IEM across all stims as the training set
        alldat = allw[ww1][ww2][:,0:nVox2Use]
#        alldat = scipy.stats.zscore(alldat, axis=1)
#        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
        
#        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,10,9)
        
        # cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
#        whichCV = np.mod(whichCV,5)
#        np.random.shuffle(whichCV)

        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
#         
        
        for tt in range(nType):
            ii=ii+1
            plt.subplot(1,nType,ii)
            
            for sf in range(nSF):
                
                # average recons within just this spatial frequency
                inds = np.where(np.logical_and(typelist==tt, sflist==sf))[0]
            
                average_recons = IEM.shift_and_average(chan_resp_all[inds,:],ori_labs[inds,:],center_deg);
                
                plt.plot(xx,average_recons)
                
            plt.ylim(ylims)                
            plt.title('%s' % (stim_types[tt]))
            plt.xlabel('Orientation Channel (deg)')
                
            if tt==0:
                plt.ylabel('Channel Activation Weight')
                plt.legend(np.round(sf_vals,1))
            else:
                plt.tick_params(axis='y', left=False,labelleft = False)
                
            plt.plot([center_deg,center_deg], ylims,'k-')
            
        plt.suptitle('Average reconstruction, train all stimuli, %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
        
#%% separate the recons according to cardinal-ness
plt.close('all')

layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

ori_labs = actual_labels
# bin the orientations according to adjacency to vertical or horizontal
bin_labs = np.zeros(np.shape(ori_labs))

# the first and last bins end up slightly smaller, but they are equal size 
# the third bin is exactly centered on 45 degrees
nBins = 5
dist_from_vertical = np.min(np.concatenate((np.abs(ori_labs), np.abs(180-ori_labs)), axis=1), axis=1)
nPerBin = int(np.ceil(np.size(np.unique(dist_from_vertical))/nBins))
startind = -2

bin_labels = [];
for bb in range(nBins):
    inds = np.logical_and(dist_from_vertical>=startind, dist_from_vertical < startind+nPerBin)
   
    startind = startind+nPerBin
    bin_labs[inds] = bb
    bin_labels.append('%d through %d deg' % (np.min(dist_from_vertical[inds]), np.max(dist_from_vertical[inds])))
    
# this set of labels describes whether we need to go clockwise or counter-clockwise to get to the nearest of 45 or 135 degrees. 
# Use this to flip some recons about their center, before averaging.
dir_to_oblique = np.zeros(np.shape(ori_labs))
dir_to_oblique[np.logical_and(ori_labs>0, ori_labs<45)] = 1
dir_to_oblique[np.logical_and(ori_labs>45, ori_labs<90)] = -1
dir_to_oblique[np.logical_and(ori_labs>90, ori_labs<135)] = 1
dir_to_oblique[np.logical_and(ori_labs>135, ori_labs<180)] = -1


xx=np.arange(0,180,1)
ylims = [-5,0]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        plt.figure()
#        ii=0
        
        
        center_deg=90
        n_chans=9
        n_folds = 500  
        
        alldat = allw[ww1][ww2]
#        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
        
        # cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
         
        for bb in range(nBins):

            inds = np.where(bin_labs==bb)[0]
        
            average_recons = IEM.shift_flip_and_average(chan_resp_all[inds,:],ori_labs[inds,:],dir_to_oblique[inds,:],center_deg);
#            average_recons = IEM.shift_and_average(chan_resp_all[inds,:],ori_labs[inds,:],center_deg);
           
            plt.plot(xx,average_recons)
            
        plt.ylim(ylims)

        plt.xlabel('Orientation Channel (deg)')
        plt.ylabel('Channel Activation Weight')
        plt.legend(bin_labels)
        plt.plot([center_deg,center_deg], ylims,'k-')
        plt.title('Average reconstruction, train all stimuli, %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
        

        
#%% train the IEM across all trials, examine prediction bias
        
plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        ii=ii+1
        plt.figure()
     
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
        n_folds = 10
        
        alldat = allw[ww1][ww2]
#        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
         
#        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,n_folds,n_chans)
                
        
        pred_labels = np.argmax(chan_resp_all, axis=1)
        un = np.unique(ori_labs)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)
           
#        plot_order = np.argsort(ori_labs, axis=0) 
#        ori_labs_sorted = np.squeeze(ori_labs[plot_order])
#        pred_labels_sorted = np.squeeze(pred_labels[plot_order])

#        plt.plot(ori_labs_sorted, pred_labels_sorted,'o')
        plt.errorbar(un,avg_pred,std_pred)
        plt.title('Reconstruction peaks, train/test all stimuli, %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
#        if ww1==np.max(layers2plot):
        plt.xlabel('Actual Orientation (deg)')
        plt.ylabel('Predicted Orientation(deg)')
        plt.plot([0,180],[0,180],'k-')
        plt.plot([center_deg, center_deg],[0,180],'k-')
        plt.axis('square')
#        plt.suptitle()
      
#%% overlay the bias curves from each layer, compare them      
               
plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,2)
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

legend_labs = [];

plt.figure()
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        legend_labs.append(layer_labels[ww1])
#        ii=ii+1
#        plt.figure()
     
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
        n_folds = 10
        
        alldat = allw[ww1][ww2]
#        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
         
#        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,n_folds,n_chans)
                
        
        pred_labels = np.argmax(chan_resp_all, axis=1)
        un = np.unique(ori_labs)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)
           
#        plot_order = np.argsort(ori_labs, axis=0) 
#        ori_labs_sorted = np.squeeze(ori_labs[plot_order])
#        pred_labels_sorted = np.squeeze(pred_labels[plot_order])

#        plt.plot(ori_labs_sorted, pred_labels_sorted,'o')
#        plt.errorbar(un,avg_pred,std_pred)
        plt.plot(un,avg_pred)
        
plt.title('Reconstruction peaks, train/test all stimuli, %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
#        if ww1==np.max(layers2plot):
plt.xlabel('Actual Orientation (deg)')
plt.ylabel('Predicted Orientation(deg)')
plt.plot([0,180],[0,180],'k-')
plt.plot([center_deg, center_deg],[0,180],'k-')
plt.axis('square')
plt.legend(legend_labs)
plt.xlim([80,100]);plt.ylim([80,100])
#        plt.suptitle()

        
        
        
#%% plot prediction bias within each type and SF separately
    
plt.close('all')

layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

xx=np.arange(0,180,1)
ylims = [-1,1]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        plt.figure()
        ii=0
        
        for sf in range(nSF):
            for tt in range(nType):
                ii=ii+1
                plt.subplot(nSF,nType,ii) 

                inds = np.where(np.logical_and(typelist==tt, sflist==sf))[0]           
            
                ori_labs = actual_labels[inds]
                center_deg=90
                n_chans=9
                n_folds = 10
                
                alldat = allw[ww1][ww2][inds,:]

                chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,n_folds,n_chans)
                              
                pred_labels = np.argmax(chan_resp_all, axis=1)
                un = np.unique(ori_labs)
                avg_pred = np.zeros(np.shape(un))
                std_pred = np.zeros(np.shape(un))
               
                for uu in range(len(un)):
                    avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)
                    std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)
                   
        #        plot_order = np.argsort(ori_labs, axis=0) 
        #        ori_labs_sorted = np.squeeze(ori_labs[plot_order])
        #        pred_labels_sorted = np.squeeze(pred_labels[plot_order])
        
        #        plt.plot(ori_labs_sorted, pred_labels_sorted,'o')
                plt.errorbar(un,avg_pred,std_pred)
                
                plt.title('%s, SF=%.2f' % (stim_types[tt],sf_vals[sf]))
                plt.axis('square')
                plt.plot([0,180],[0,180],'k-')
                plt.plot([center_deg, center_deg],[0,180],'k-')
#                plt.xlim([80,100])
#                plt.ylim([80,100])
                if sf==np.max(nSF):
                    plt.xlabel('Actual Orientation (deg)')
                    plt.ylabel('Predicted Orientation(deg)')
        plt.suptitle('Reconstruction peaks, %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
        
        
        
#%% adding noise to the test set       

plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

noise_levels = np.arange(0, 0.65, 0.2)

nVox2Use=  50;
ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        ii=ii+1
        plt.figure()
     
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
        n_folds = 10
        
        alldat = allw[ww1][ww2][:,0:nVox2Use]
        ii=0
        for nn in range(np.size(noise_levels)):
            ii=ii+1
            plt.subplot(2,2,ii)
#        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
#         cross-validate, leaving one stimulus type and SF out at a time
            chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
            un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
            for cv in range(np.size(np.unique(whichCV))):
                trninds = np.where(whichCV!=cv)[0]
                tstinds = np.where(whichCV==cv)[0]
    #            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
            
                tstdat = alldat[tstinds,:]
                tstdat_wnoise = tstdat + np.random.normal(0,noise_levels[nn], np.shape(tstdat))*tstdat
                
                # call get_recons for this fold
                chan_resp_tmp = IEM.get_recons(alldat[trninds,:], ori_labs[trninds,:], tstdat_wnoise ,n_chans)
               
    
                chan_resp_all[tstinds,:] = chan_resp_tmp
    #        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,n_folds,n_chans)
                    
            
            pred_labels = np.argmax(chan_resp_all, axis=1)
            un = np.unique(ori_labs)
            avg_pred = np.zeros(np.shape(un))
            std_pred = np.zeros(np.shape(un))
           
            for uu in range(len(un)):
                avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)
                std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)
               
    #        plot_order = np.argsort(ori_labs, axis=0) 
    #        ori_labs_sorted = np.squeeze(ori_labs[plot_order])
    #        pred_labels_sorted = np.squeeze(pred_labels[plot_order])
    
    #        plt.plot(ori_labs_sorted, pred_labels_sorted,'o')
            plt.errorbar(un,avg_pred,std_pred)
#            plt.plot(un, avg_pred,'-')
            plt.title('test set noise = %.1f' % noise_levels[nn])
            plt.xlabel('Actual Orientation (deg)')
            plt.ylabel('Predicted Orientation(deg)')
            
            plt.axis('square')
    #        plt.legend(noise_levels)
            plt.plot([0,180],[0,180],'k-')
            plt.plot([center_deg, center_deg],[0,180],'k-')
        plt.suptitle('Reconstruction peaks, train no noise, %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))

