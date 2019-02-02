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
 

#%% run the IEM - within each stimulus type and sf separately

plt.close('all')

layers2plot = [2]
timepts2plot = [0]
noiselevels2plot =[0]

xx=np.arange(0,180,1)

ylims = [-1,1]

nVox2Use=100

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            plt.figure()
            ii=0      
            for sf in range(nSF):
                 for tt in range(nType):
                    ii=ii+1
                    plt.subplot(nSF,nType,ii)
     
                    # first find the discriminability of all gratings with this exact label (should be close to zero)
                    inds = np.where(np.logical_and(np.logical_and(typelist==tt, sflist==sf), noiselist==nn))[0]
                           
                    ori_labs = actual_labels[inds]
#                    ori_labs_shuff = ori_labs
#                    np.random.shuffle(ori_labs_shuff)
                    center_deg=90
                    n_chans=9
                    n_folds = 10
                    
                    alldat = allw[ww1][0][ww2][inds,0:nVox2Use]
    
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
            plt.suptitle('Average reconstruction, trn/test within stimulus type and SF. \nWeights from %s - %s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
#%% train the IEM across all trials
        
plt.close('all')

layers2plot = np.arange(0,19,1)
timepts2plot = [0]
noiselevels2plot = [0]

ylims = [-1,1]

nVox2Use = 100

for nn in noiselevels2plot:
    ii=0;
    plt.figure()
    for ww1 in layers2plot:
       
        ii=ii+1

        ori_labs = actual_labels
        center_deg=90
        n_chans=9
        
        myinds = np.where(noiselist==nn)[0]

        alldat = allw[ww1][0][ww2][myinds,0:nVox2Use]
        typelist_now = typelist[myinds]
        sflist_now = sflist[myinds]

        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist_now,sflist_now), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
            
        average_recons = IEM.shift_and_average(chan_resp_all,ori_labs,center_deg);
          
        plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
        plt.plot(xx,average_recons)
        plt.ylim(ylims)
        plt.plot([center_deg,center_deg], ylims)
        plt.title('%s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
        if ww1==np.max(layers2plot)-2:
            plt.xlabel('Orientation Channel (deg)')
            plt.ylabel('Channel Activation Weight')
        else:
            plt.tick_params(axis='x', bottom=False,labelbottom = False)
            
        if (np.size(timepts2plot)>1 and ww2==np.max(timepts2plot)):
            
            plt.tick_params(axis='y', left=False,labelleft = False)
            plt.ylabel(None)
        
    plt.suptitle('Average reconstructions, leave one stimulus type out\nnoise=%.2f' % (noise_levels[nn]))
#%% train all stimuli, separate the test set according to type/sf

plt.close('all')

layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
#layers2plot = [0]
timepts2plot = np.arange(0,1)

noiselevels2plot = [0]


ylims = [-5,0]

nVox2Use = 100

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            plt.figure()
            ii=0
            
            ori_labs = actual_labels
            center_deg=90
            n_chans=9
    #        n_folds = 20
            
            # run the IEM across all stims as the training set
            alldat = allw[ww1][nn][ww2][:,0:nVox2Use]
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
                
            plt.suptitle('Train within noise level, %s - %s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
 
           
#%% separate the recons according to cardinal-ness
plt.close('all')

layers2plot = np.arange(0,nLayers,4)
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

noiselevels2plot = [0]

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
ylims = [-3,-1]

plt.figure()
ii=0
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            
            ii=ii+1
            plt.subplot(np.ceil(len(layers2plot)/2),2,ii)
            
            
            center_deg=90
            n_chans=9
            n_folds = 500  
            
            alldat = allw[ww1][nn][ww2]
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
    
            
            if ww1==layers2plot[-1]:
                plt.legend(bin_labels, bbox_to_anchor = (1.2,1))
                plt.xlabel('Orientation Channel (deg)')
                plt.ylabel('Channel Activation Weight')
            else:
                plt.tick_params(axis='y', left=False,labelleft = False)
                plt.tick_params(axis='x', bottom=False,labelbottom = False)
            plt.plot([center_deg,center_deg], ylims,'k-')
            plt.title('%s' % (layer_labels[ww1]))
            
plt.suptitle('Average reconstruction, train all stimuli\nnoise=%.2f' %  noise_levels[nn])
  
#%% separate the recons according to cardinal-ness, train within one SF/type
plt.close('all')

layers2plot = np.arange(0,nLayers,4)
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

noiselevels2plot = [0]

ori_labs = actual_labels
# bin the orientations according to adjacency to vertical or horizontal
bin_labs = np.zeros(np.shape(ori_labs))

sf = 2   
tt = 0

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
ylims = [-1,2]

plt.figure()
ii=0
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            
            ii=ii+1
            plt.subplot(np.ceil(len(layers2plot)/2),2,ii)
            
            inds = np.where(np.logical_and(typelist==tt, sflist==sf))[0]
                           
            ori_labs = actual_labels[inds]
        #                    ori_labs_shuff = ori_labs
        #                    np.random.shuffle(ori_labs_shuff)
        
            dir_to_oblique_small = dir_to_oblique[inds]
            bin_labs_small = bin_labs[inds]
            
            center_deg=90
            n_chans=9
            n_folds = 10
            
            alldat = allw[ww1][nn][ww2][inds,0:nVox2Use]
        
            chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,n_folds,n_chans)
            
            for bb in range(nBins):
    
                inds = np.where(bin_labs_small==bb)[0]
            
                average_recons = IEM.shift_flip_and_average(chan_resp_all[inds,:],ori_labs[inds,:],dir_to_oblique_small[inds,:],center_deg);
    #            average_recons = IEM.shift_and_average(chan_resp_all[inds,:],ori_labs[inds,:],center_deg);
               
                plt.plot(xx,average_recons)
                
            plt.ylim(ylims)
    
            
            if ww1==layers2plot[-1]:
                plt.legend(bin_labels, bbox_to_anchor = (1.2,1))
                plt.xlabel('Orientation Channel (deg)')
                plt.ylabel('Channel Activation Weight')
            else:
                plt.tick_params(axis='y', left=False,labelleft = False)
                plt.tick_params(axis='x', bottom=False,labelbottom = False)
            plt.plot([center_deg,center_deg], ylims,'k-')
            plt.title('%s' % (layer_labels[ww1]))
            
  
plt.suptitle('Average reconstruction, %s, SF=%.2f\nnoise=%.2f' % (stim_types[tt], sf_vals[sf], noise_levels[nn]))

#%% Plot predicted peak versus actual location - 1-180, subplots of different areas
        
plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,1)
#layers2plot = [0]
#layers2plot = np.arange(0,1)
#layers2plot = [6]
timepts2plot = np.arange(0,1)
noiselevels2plot = np.arange(0,1)
plt.figure()
         
xlims = [0,180]
ylims = [0,180]

ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        for nn in noiselevels2plot:
            
            ii=ii+1
            plt.subplot(np.ceil(len(layers2plot)/4),np.min([len(layers2plot),4]),ii)
            ori_labs = actual_labels
            center_deg=90
            n_chans=9
            n_folds = 10
            
            alldat = allw[ww1][nn][ww2]
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
                    
            
            pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
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
            plt.title('%s' % (layer_labels[ww1]))
            if ww1==np.max(layers2plot):
                plt.xlabel('Actual Orientation (deg)')
                plt.ylabel('Predicted Orientation(deg)')
            else:
                plt.tick_params(axis='y', left=False,labelleft = False)
                plt.tick_params(axis='x', bottom=False,labelbottom = False)
            plt.plot([0,180],[0,180],'k-')
            plt.plot([center_deg, center_deg],[0,180],'k-')
            plt.axis('square')
            plt.xlim(xlims)
            plt.ylim(ylims)
    #        plt.suptitle()
    
plt.suptitle('Reconstruction peaks, train/test all stimuli\nnoise=%.2f' % noise_levels[nn])

#%% Plot predicted peak versus actual location - 80-100, subplots of different areas
        
plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,1)
#layers2plot = [0]
#layers2plot = np.arange(0,1)
#layers2plot = [6]
timepts2plot = np.arange(0,1)
noiselevels2plot = np.arange(0,1)
plt.figure()
         
xlims = [70,110]
ylims = [70,110]

ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        for nn in noiselevels2plot:
            
            ii=ii+1
            plt.subplot(np.ceil(len(layers2plot)/4),np.min([len(layers2plot),4]),ii)
            ori_labs = actual_labels
            center_deg=90
            n_chans=9
            n_folds = 10
            
            alldat = allw[ww1][nn][ww2]
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
                    
            
            pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
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
            plt.title('%s' % (layer_labels[ww1]))
            
            if ww1==np.max(layers2plot):
                plt.xlabel('Actual Orientation (deg)')
                plt.ylabel('Predicted Orientation(deg)')
            else:
                plt.tick_params(axis='y', left=False,labelleft = False)
                plt.tick_params(axis='x', bottom=False,labelbottom = False)
            plt.plot([0,180],[0,180],'k-')
            plt.plot([center_deg, center_deg],[0,180],'k-')
            plt.axis('square')
            plt.xlim(xlims)
            plt.ylim(ylims)
    #        plt.suptitle()
    
plt.suptitle('Reconstruction peaks, train/test all stimuli\nnoise=%.2f' % noise_levels[nn])

#%% Plot bias curves from each area, trn/test all stims
        
plt.close('all')

layers2plot = np.arange(0,nLayers,1)

ww2=0
nn=0
plt.figure()

ylims = [-10,10]

ii=0;
for ww1 in layers2plot:
       
    ii=ii+1
    plt.subplot(np.ceil(len(layers2plot)/4),np.min([len(layers2plot),4]),ii)
    ori_labs = actual_labels
    center_deg=90
    n_chans=9
    n_folds = 10
    
    alldat = allw[ww1][nn][ww2]

#         cross-validate, leaving one stimulus type and SF out at a time
    chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
    un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
    for cv in range(np.size(np.unique(whichCV))):
        trninds = np.where(whichCV!=cv)[0]
        tstinds = np.where(whichCV==cv)[0]

        chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)

    pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
    un = np.unique(ori_labs)
    avg_pred = np.zeros(np.shape(un))
    std_pred = np.zeros(np.shape(un))
   
    for uu in range(len(un)):
        avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)
        std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)

     # calculate bias
    # first correct values that wrapped around
    avg_pred_corr = avg_pred
    indslow = np.where(np.logical_and(un<45, avg_pred>135))
    indshigh = np.where(np.logical_and(un>135, avg_pred<45))
    print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
    avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
    avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
    
    avg_bias = avg_pred_corr - un
    
#    plt.plot(un,avg_bias)
    plt.errorbar(un,avg_bias,std_pred)
#    plt.plot(un,avg_bias,'k-')
    plt.title('%s' % (layer_labels[ww1]))
    
    if ww1==np.max(layers2plot)-2:
        plt.xlabel('Actual Orientation (deg)')
        plt.ylabel('Bias (deg)')
    else:
        plt.tick_params(axis='y', left=False,labelleft = False)
        plt.tick_params(axis='x', bottom=False,labelbottom = False)
   
#    plt.axis('square')
    plt.xlim([0,180])
    plt.ylim(ylims)
    plt.plot([0,180],[0,0],'k-')
    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
    
plt.suptitle('Reconstruction bias, train/test all stimuli\nnoise=%.2f' % noise_levels[nn])

#%% Plot bias curves from each area, within one SF and stimulus type
        
plt.close('all')

layers2plot = np.arange(0,nLayers,1)

ww2=0
nn=0
plt.figure()

sf=2
tt=0

ylims = [-10,10]

ii=0;
for ww1 in layers2plot:
       
    ii=ii+1
    plt.subplot(np.ceil(len(layers2plot)/4),np.min([len(layers2plot),4]),ii)
    inds = np.where(np.logical_and(typelist==tt, sflist==sf))[0]
                           
    ori_labs = actual_labels[inds]
#                    ori_labs_shuff = ori_labs
#                    np.random.shuffle(ori_labs_shuff)
    center_deg=90
    n_chans=9
    n_folds = 10
    
    alldat = allw[ww1][nn][ww2][inds,0:nVox2Use]

    chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,n_folds,n_chans)
    
    pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
    un = np.unique(ori_labs)
    avg_pred = np.zeros(np.shape(un))
    std_pred = np.zeros(np.shape(un))
   
    for uu in range(len(un)):
        avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)
        std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=180,low=0)

     # calculate bias
    # first correct values that wrapped around
    avg_pred_corr = avg_pred
    indslow = np.where(np.logical_and(un<45, avg_pred>135))
    indshigh = np.where(np.logical_and(un>135, avg_pred<45))
    print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
    avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
    avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
    
    avg_bias = avg_pred_corr - un
    
#    plt.plot(un,avg_bias)
    plt.errorbar(un,avg_bias,std_pred)
#    plt.plot(un,avg_bias)
    plt.title('%s' % (layer_labels[ww1]))
    
    if ww1==np.max(layers2plot)-2:
        plt.xlabel('Actual Orientation (deg)')
        plt.ylabel('Bias (deg)')
    else:
        plt.tick_params(axis='y', left=False,labelleft = False)
        plt.tick_params(axis='x', bottom=False,labelbottom = False)
   
#    plt.axis('square')
    plt.xlim([0,180])
    plt.ylim(ylims)
    plt.plot([0,180],[0,0],'k-')
    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
    
plt.suptitle('Reconstruction bias, %s, SF=%.2f\nnoise=%.2f' % (stim_types[tt], sf_vals[sf], noise_levels[nn]))
#%% overlay the bias curves from each layer, compare them      
               
plt.close('all')
#plt.figure()
#layers2plot = np.arange(0,nLayers,4)
layers2plot = [0,3,6,9,12]
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)

noiselevels2plot = [0]

legend_labs = [];

ylims = [-10,10]

for nn in noiselevels2plot:
    
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
            
            alldat = allw[ww1][nn][ww2]
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
                       
            
            pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
            un = np.unique(ori_labs)
            avg_pred = np.zeros(np.shape(un))
            std_pred = np.zeros(np.shape(un))
           
            for uu in range(len(un)):
                avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)
                std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)
               

            # calculate bias
            # first correct values that wrapped around
            avg_pred_corr = avg_pred
            indslow = np.where(np.logical_and(un<45, avg_pred>135))
            indshigh = np.where(np.logical_and(un>135, avg_pred<45))
            print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
            avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
            avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
            
            avg_bias = avg_pred_corr - un
            plt.plot(un,avg_bias)
#            plt.errorbar(un,avg_bias,std_pred)
    plt.ylim(ylims)
    plt.xlim([0,180])        
    plt.title('Reconstruction bias, train/test all stimuli\nnoise=%.2f' % (noise_levels[nn]))
    #        if ww1==np.max(layers2plot):
    plt.xlabel('Actual Orientation (deg)')
    plt.ylabel('Bias (deg)')
    plt.plot([0,180],[0,0],'k-')
    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
#    plt.axis('square')
    plt.legend(legend_labs,bbox_to_anchor=(0.8,1))
    
#    plt.xlim([80,100]);plt.ylim([80,100])
    #        plt.suptitle()

        
#%% trn/test within noise level, plot bias
    
plt.close('all')

layers2plot =[4,5,6,7]
ww2 = 0
noiselevels2plot = [0,1,2,3,4]

ylims = [-10,10]

for ww1 in layers2plot:
    
    plt.figure()
    
    legend_labs = [];

    for nn in noiselevels2plot:
 
        legend_labs.append('noise=%.2f' % noise_levels[nn])

        ori_labs = actual_labels
        center_deg=90
        n_chans=9
        n_folds = 10
        
        alldat = allw[ww1][nn][ww2]

#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
   
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
         

        pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
        un = np.unique(ori_labs)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)

        # calculate bias
        # first correct values that wrapped around
        avg_pred_corr = avg_pred
        indslow = np.where(np.logical_and(un<45, avg_pred>135))
        indshigh = np.where(np.logical_and(un>135, avg_pred<45))
        print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
        avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
        avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
        
        avg_bias = avg_pred_corr - un
        plt.plot(un,avg_bias)
#            plt.errorbar(un,avg_bias,std_pred)
            
    plt.ylim(ylims)
    plt.xlim([0,180])        
    plt.title('Reconstruction bias, train/test all stimuli (within noise level)\n%s' % (layer_labels[ww1]))
    #        if ww1==np.max(layers2plot):
    plt.xlabel('Actual Orientation (deg)')
    plt.ylabel('Bias (deg)')
    plt.plot([0,180],[0,0],'k-')
    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
#    plt.axis('square')
    plt.legend(legend_labs,bbox_to_anchor=(0.8,1))
#%% trn highest noise, test noise, plot bias (doesn't work)
    
plt.close('all')

layers2plot =[5]
ww2 = 0
noiselevels2plot = [0,1,2,3,4]
ylims = [-10,10]

for ww1 in layers2plot:
    
    plt.figure()
    
    legend_labs = [];

    for nn in noiselevels2plot:
 
        legend_labs.append('noise=%.2f' % noise_levels[nn])

        ori_labs = actual_labels
        center_deg=90
        n_chans=9
        n_folds = 10
        
        trndat = allw[ww1][4][ww2]
        tstdat = allw[ww1][nn][ww2]
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(tstdat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
   
            chan_resp_all[tstinds,:] = IEM.get_recons(trndat[trninds,:],ori_labs[trninds],tstdat[tstinds,:],n_chans)
             

        pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
        un = np.unique(ori_labs)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)

        # calculate bias
        # first correct values that wrapped around
        avg_pred_corr = avg_pred
        indslow = np.where(np.logical_and(un<45, avg_pred>135))
        indshigh = np.where(np.logical_and(un>135, avg_pred<45))
        print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
        avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
        avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
        
        avg_bias = avg_pred_corr - un
        plt.plot(un,avg_bias)
#            plt.errorbar(un,avg_bias,std_pred)
            
    plt.ylim(ylims)
    plt.xlim([0,180])        
    plt.title('Reconstruction bias, train 0.80 noise\n%s' % (layer_labels[ww1]))
    #        if ww1==np.max(layers2plot):
    plt.xlabel('Actual Orientation (deg)')
    plt.ylabel('Bias (deg)')
    plt.plot([0,180],[0,0],'k-')
    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
#    plt.axis('square')
    plt.legend(legend_labs,bbox_to_anchor=(0.8,.4))

#%% trn no noise, test noise, plot bias (doesn't work)
    
plt.close('all')

layers2plot =[0]
ww2 = 0
noiselevels2plot = [0,1,2,3,4]
ylims=[-10,10]

for ww1 in layers2plot:
    
    plt.figure()
    
    legend_labs = [];

    for nn in noiselevels2plot:
 
        legend_labs.append('noise=%.2f' % noise_levels[nn])

        ori_labs = actual_labels
        center_deg=90
        n_chans=9
        n_folds = 10
        
        trndat = allw[ww1][0][ww2]
        tstdat = allw[ww1][nn][ww2]
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(tstdat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
   
            chan_resp_all[tstinds,:] = IEM.get_recons(trndat[trninds,:],ori_labs[trninds],tstdat[tstinds,:],n_chans)
             

        pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
        un = np.unique(ori_labs)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs==un[uu])[0]], high=179,low=0)

        # calculate bias
        # first correct values that wrapped around
        avg_pred_corr = avg_pred
        indslow = np.where(np.logical_and(un<45, avg_pred>135))
        indshigh = np.where(np.logical_and(un>135, avg_pred<45))
        print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
        avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
        avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
        
        avg_bias = avg_pred_corr - un
        plt.plot(un,avg_bias)
#            plt.errorbar(un,avg_bias,std_pred)
            
    plt.ylim(ylims)
    plt.xlim([0,180])        
    plt.title('Reconstruction bias, train zero noise\n%s' % (layer_labels[ww1]))
    #        if ww1==np.max(layers2plot):
    plt.xlabel('Actual Orientation (deg)')
    plt.ylabel('Bias (deg)')
    plt.plot([0,180],[0,0],'k-')
    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
#    plt.axis('square')
    plt.legend(legend_labs,bbox_to_anchor=(0.8,1))

 #%% train the IEM within each noise level, test on same noise level - plot recons
    
plt.close('all')
#plt.figure()
#layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
layers2plot = [0, 5, 10]
timepts2plot = np.arange(0,1)
#noiselevels2plot = np.arange(0,5)
ylims = [-4,0]
ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        plt.figure()
         
        legendlabs = [];
        
        ii=0;
        
        for nn in [0,1,2,3,4]:
            ii=ii+1
            
#            plt.subplot(3,2,ii)
            
            legendlabs.append('train/test noise=%.2f' % noise_levels[nn])
#            ii=ii+1
            
            ori_labs = actual_labels
            center_deg=90
            n_chans=9
            n_folds = 10
            
            trndat = allw[ww1][nn][ww2]
            tstdat = allw[ww1][nn][ww2]
    #        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
    #         cross-validate, leaving one stimulus type and SF out at a time
            chan_resp_all = np.zeros([np.shape(tstdat)[0], 180])
            un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
            for cv in range(np.size(np.unique(whichCV))):
                trninds = np.where(whichCV!=cv)[0]
                tstinds = np.where(whichCV==cv)[0]
    #            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
            
                chan_resp_all[tstinds,:] = IEM.get_recons(trndat[trninds,:],ori_labs[trninds],tstdat[tstinds,:],n_chans)
             

            average_recons = IEM.shift_and_average(chan_resp_all,ori_labs,center_deg);
                    
            plt.plot(xx,average_recons)
                    
#            plt.ylim(ylims)                
#        plt.title('test noise=%.2f' % (noise_levels[nn]))
        plt.xlabel('Orientation Channel (deg)')
            
#        if ii==0:
        plt.ylabel('Channel Activation Weight')
#        plt.legend(np.round(sf_vals,1))
#        else:
#    plt.tick_params(axis='y', left=False,labelleft = False)
            
        plt.plot([center_deg,center_deg], ylims,'k-')
        
        plt.legend(legendlabs)
        
        plt.title('Reconstruction peaks, train within noise level\n %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
#%% train within noise level, separate by spatial frequency and window

plt.close('all')

#layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
layers2plot = [5]
timepts2plot = np.arange(0,1)

noiselevels2plot = [0,1,2,3,4]

#types2plot = [0,1]
#sf2plot = np.arange(0,nSF,1)

ylims = [-5,0]

nVox2Use = 100

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        
                plt.figure()
#        ii=0
#        for tt in range(nType):            
#            for sf in range(nSF):
#                ii=ii+1
#                plt.subplot(nSF,nType,ii)
                legend_labs = []
                for nn in noiselevels2plot:
                    legend_labs.append('train/test noise=%.2f' % noise_levels[nn])
            
                    ori_labs = actual_labels
                    center_deg=90
                    n_chans=9

                    # run the IEM across all stims as the training set
                    alldat = allw[ww1][nn][ww2][:,0:nVox2Use]

                    # cross-validate, leaving one stimulus type and SF out at a time
                    chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
                    un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
           
                    for cv in range(np.size(np.unique(whichCV))):
                        trninds = np.where(whichCV!=cv)[0]
                        tstinds = np.where(whichCV==cv)[0]
                  
                        chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
                       # average recons within just this spatial frequency
#                        inds = np.where(np.logical_and(typelist==tt, sflist==sf))[0]
                    
                        average_recons = IEM.shift_and_average(chan_resp_all[tstinds,:],ori_labs[tstinds,:],center_deg);
                        plt.subplot(nSF,nType,cv)
                        plt.plot(xx,average_recons)
                        
                plt.ylim(ylims)                
                plt.title('SF=%.2f - %s' % (sflist[sf], stim_types[tt]))
                plt.xlabel('Orientation Channel (deg)')
                    
                if tt==0:
                    plt.ylabel('Channel Activation Weight')
#                    plt.legend(np.round(sf_vals,1))
                else:
                    plt.tick_params(axis='y', left=False,labelleft = False)
                    if sf==nSF-1:
                         plt.legend(legend_labs)
                         
                    
                plt.plot([center_deg,center_deg], ylims,'k-')
               
                    
            plt.suptitle('Average reconstruction, train within noise level, %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
               
#%% train the IEM on no noise, test noise - plot recons
    
plt.close('all')
#plt.figure()
#layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
layers2plot = [0, 5, 10]
timepts2plot = np.arange(0,1)
#noiselevels2plot = np.arange(0,5)
ylims = [-4,0]
ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        plt.figure()
         
        legendlabs = [];
        
        ii=0;
        
        for nn in [0,1,2,3,4]:
            ii=ii+1
            
#            plt.subplot(3,2,ii)
            
            legendlabs.append('test noise=%.2f' % noise_levels[nn])
#            ii=ii+1
            
            ori_labs = actual_labels
            center_deg=90
            n_chans=9
            n_folds = 10
            
            trndat = allw[ww1][0][ww2]
            tstdat = allw[ww1][nn][ww2]
    #        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
    #         cross-validate, leaving one stimulus type and SF out at a time
            chan_resp_all = np.zeros([np.shape(tstdat)[0], 180])
            un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
            for cv in range(np.size(np.unique(whichCV))):
                trninds = np.where(whichCV!=cv)[0]
                tstinds = np.where(whichCV==cv)[0]
    #            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
            
                chan_resp_all[tstinds,:] = IEM.get_recons(trndat[trninds,:],ori_labs[trninds],tstdat[tstinds,:],n_chans)
             

            average_recons = IEM.shift_and_average(chan_resp_all,ori_labs,center_deg);
                    
            plt.plot(xx,average_recons)
                    
#            plt.ylim(ylims)                
#        plt.title('test noise=%.2f' % (noise_levels[nn]))
        plt.xlabel('Orientation Channel (deg)')
            
#        if ii==0:
        plt.ylabel('Channel Activation Weight')
#        plt.legend(np.round(sf_vals,1))
#        else:
#    plt.tick_params(axis='y', left=False,labelleft = False)
            
        plt.plot([center_deg,center_deg], ylims,'k-')
        
        plt.legend(legendlabs)
        
        plt.title('Average reconstruction, train zero noise\n %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
#%% train the IEM on highest noise, test noise - plot recons
    
plt.close('all')
#plt.figure()
#layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
layers2plot = [0, 5, 10]
timepts2plot = np.arange(0,1)
#noiselevels2plot = np.arange(0,5)
ylims = [-4,0]
ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        plt.figure()
         
        legendlabs = [];
        
        ii=0;
        
        for nn in [0,1,2,3,4]:
            ii=ii+1
            
#            plt.subplot(3,2,ii)
            
            legendlabs.append('test noise=%.2f' % noise_levels[nn])
#            ii=ii+1
            
            ori_labs = actual_labels
            center_deg=90
            n_chans=9
            n_folds = 10
            
            trndat = allw[ww1][4][ww2]
            tstdat = allw[ww1][nn][ww2]
    #        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
    #         cross-validate, leaving one stimulus type and SF out at a time
            chan_resp_all = np.zeros([np.shape(tstdat)[0], 180])
            un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
            for cv in range(np.size(np.unique(whichCV))):
                trninds = np.where(whichCV!=cv)[0]
                tstinds = np.where(whichCV==cv)[0]
    #            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
            
                chan_resp_all[tstinds,:] = IEM.get_recons(trndat[trninds,:],ori_labs[trninds],tstdat[tstinds,:],n_chans)
             

            average_recons = IEM.shift_and_average(chan_resp_all,ori_labs,center_deg);
                    
            plt.plot(xx,average_recons)
                    
#            plt.ylim(ylims)                
#        plt.title('test noise=%.2f' % (noise_levels[nn]))
        plt.xlabel('Orientation Channel (deg)')
            
#        if ii==0:
        plt.ylabel('Channel Activation Weight')
#        plt.legend(np.round(sf_vals,1))
#        else:
#    plt.tick_params(axis='y', left=False,labelleft = False)
            
        plt.plot([center_deg,center_deg], ylims,'k-')
        
        plt.legend(legendlabs)
        
        plt.title('Average reconstruction, train 0.80 noise\n %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))

#%% train the IEM on no noise, test noise - plot recons separated by cardinality
    
plt.close('all')
#plt.figure()
#layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
layers2plot = [6]
timepts2plot = np.arange(0,1)
#noiselevels2plot = np.arange(0,5)




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


ylims = [-4,0]
ii=0;
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        
        plt.figure()
         
#        legendlabs = [];
        
        ii=0;
        
        for nn in [0,1,2,3,4]:
            ii=ii+1
            
            plt.subplot(3,2,ii)
            
#            legendlabs.append('test noise=%.2f' % noise_levels[nn])
#            ii=ii+1
            
            ori_labs = actual_labels
            center_deg=90
            n_chans=9
            n_folds = 10
            
            trndat = allw[ww1][0][ww2]
            tstdat = allw[ww1][nn][ww2]
    #        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
    #         cross-validate, leaving one stimulus type and SF out at a time
            chan_resp_all = np.zeros([np.shape(tstdat)[0], 180])
            un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
            for cv in range(np.size(np.unique(whichCV))):
                trninds = np.where(whichCV!=cv)[0]
                tstinds = np.where(whichCV==cv)[0]
    #            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
            
                chan_resp_all[tstinds,:] = IEM.get_recons(trndat[trninds,:],ori_labs[trninds],tstdat[tstinds,:],n_chans)
             

            
            for bb in range(nBins):
    
                inds = np.where(bin_labs==bb)[0]
            
                average_recons = IEM.shift_flip_and_average(chan_resp_all[inds,:],ori_labs[inds,:],dir_to_oblique[inds,:],center_deg);
    #            average_recons = IEM.shift_and_average(chan_resp_all[inds,:],ori_labs[inds,:],center_deg);
               
                plt.plot(xx,average_recons)
                
                    
#            plt.ylim(ylims)                
            plt.title('test noise=%.2f' % (noise_levels[nn]))
            plt.xlabel('Orientation Channel (deg)')
            plt.legend(bin_labels)
                
#            if tt==0:
            plt.ylabel('Channel Activation Weight')
#                plt.legend(np.round(sf_vals,1))
#            else:
#                plt.tick_params(axis='y', left=False,labelleft = False)
                
            plt.plot([center_deg,center_deg], ylims,'k-')
            
        
#        plt.legend(legendlabs)
        plt.suptitle('Reconstruction peaks, train zero noise\n %s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))