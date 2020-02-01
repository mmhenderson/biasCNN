#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt
import scipy
import scipy.stats
   
import classifiers    

import numpy as np
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True
#%% get the data ready to go...then can run any below cells independently.

#model_str = 'inception_oriTst1'
#model_name_2plot = 'Inception-V3'

#model_str = 'nasnet_oriTst1'
#model_name_2plot = 'NASnet'

model_str = 'vgg16_oriTst2'
model_name_2plot = 'VGG-16'

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

#%% plot predicted vs actual labels

plt.close('all')

noiselevels2plot = np.arange(nNoiseLevels)

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

noiselevels2plot = np.arange(nNoiseLevels)

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
plt.close('all')
nn=0

pred_labels = all_labs[0][1]
pred_labels_before = all_labs[0][0]
    

for tt in range(len(stim_types)):
    for sf in range(len(sf_vals)):

        myinds_bool = np.logical_and(np.logical_and(typelist==tt, sflist==sf), noiselist==nn)
        
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
        plt.title('After retraining, noise=%.2f\n predicted labels versus actual labels, %s, SF=%.2f' % (noise_levels[nn], stim_types[tt],sf_vals[sf]))
