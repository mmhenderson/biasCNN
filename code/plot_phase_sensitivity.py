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

#model_str = 'inception_oriTst5a'
#model_name_2plot = 'Inception-V3'


#model_str = 'nasnet_oriTst5a'
#model_name_2plot = 'NASnet'

model_str = 'vgg16_oriTst5a'
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
if 'phase_vals' in info.keys():
    phase_vals = info['phase_vals']
else:
    phase_vals = []    

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

    
    
#%% Plot the PHASE discriminability - across orientations.
    
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 0 # spat freq
tt = 0

plt.figure()
xx=1

steps = [1]
legendlabs = ['%d deg apart'%(steps[ss]) for ss in range(np.size(steps))]

for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
    
    un,ia = np.unique(phaselist, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==phaselist)
    
     
    disc = np.zeros([np.shape(un)[0],np.size(steps)])

    for ii in np.arange(0,np.size(un)):
        for ss in range(np.size(steps)):
                
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(phaselist==np.mod(un[ii]-(np.floor(steps[ss]/2)), nPhase), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(phaselist==np.mod(un[ii]+(np.ceil(steps[ss]/2)), nPhase), myinds_bool))[0]
            assert(np.size(inds_left)==180)
            assert(np.size(inds_right)==180)
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii,ss] = dist

    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for ss in range(np.size(steps)):
        xvals = phase_vals[un]
        xvals = xvals+phase_vals[1]/2
        plt.plot(xvals,disc[:,ss])


    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual PHASE of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
    lines = np.arange(0,360,45)
    for ll in lines:
        plt.plot([ll,ll],plt.gca().get_ylim(),'k')
    plt.suptitle('%s\nDiscriminability between pairs of PHASES, collapsed over orientation\n%s, SF=%.2f, noise=%.2f' % (model_str,stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1
    
#%% Plot the PHASE discriminability - within orientation bins
    
plt.close('all')
nn=0
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)

sf = 0 # spat freq
tt = 0

plt.figure()
xx=1


binsize = int(18)
nbins = int(180/binsize)
oribins = np.reshape(np.roll(np.arange(0,180,1),int(np.floor(binsize/2))), [nbins,binsize])
bin_centers = np.round(scipy.stats.circmean(oribins,180,0,1))
bin_list = np.zeros(np.shape(orilist))
for bb in range(nbins):
    inds = np.where(np.isin(orilist, oribins[bb,:]))[0]
    bin_list[inds,0] = bb
    
    
    
for ww1 in layers2plot:

    w = allw[ww1][ww2]
    
    disc = np.zeros([np.shape(un)[0],nbins])
    
    for bb in range(np.size(bin_centers)):
        
        myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn, bin_list==bb], axis=0)
        
        un,ia = np.unique(phaselist, return_inverse=True)
        assert np.all(np.expand_dims(ia,1)==phaselist)

        for ii in np.arange(0,np.size(un)):
               
            # find all gratings at the positions of interest
            inds_left = np.where(np.logical_and(phaselist==np.mod(un[ii]-0, nPhase), myinds_bool))[0]        
            inds_right = np.where(np.logical_and(phaselist==np.mod(un[ii]+1, nPhase), myinds_bool))[0]
#            assert(np.size(inds_left)==180)
#            assert(np.size(inds_right)==180)
            dist = classifiers.get_norm_euc_dist(w[inds_right,:],w[inds_left,:])
 
            disc[ii,bb] = dist

    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for bb in range(np.size(bin_centers)):
        xvals = phase_vals[un]
        xvals = xvals+phase_vals[1]/2
        plt.plot(xvals,disc[:,bb])

#    plt.ylim([0,5])
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual PHASE of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(['orients around %.0f deg'%bin_centers[bb] for bb in range(np.size(bin_centers))])
    else:
        plt.xticks([])
    lines = np.arange(0,360,45)
    for ll in lines:
        plt.plot([ll,ll],plt.gca().get_ylim(),'k')
    plt.suptitle('%s\nDiscriminability between pairs of PHASES,within orientation bins\n%s, SF=%.2f, noise=%.2f' % (model_str,stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1
 
    
        
#%% plot the within-orientation variance for the first few units - within a spatial frequency
    
    
plt.close('all')
nn=2
ww2 = 0;

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [8,9]
#layers2plot = []
sf = 0 # spat freq
tt = 0

plt.figure()
xx=1
units = np.arange(0,15,1)
#units = [0]
legendlabs = ['unit %d' %(ii+1) for ii in units]

for ww1 in layers2plot:

    # first get my covariance matrix, across all images. 
#    my_cov = np.cov(np.transpose(allw[ww1][ww2]))
#    my_cov_inv = np.linalg.inv(my_cov)
     
    
    myinds_bool = np.all([sflist==sf,   typelist==tt, noiselist==nn], axis=0)
#    myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
    un,ia = np.unique(actual_labels, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==actual_labels)
    
    disc = np.zeros(np.shape(un))
    sd_disc = np.zeros(np.shape(un))
    
    disc_same = np.zeros(np.shape(un))
    sd_disc_same = np.zeros(np.shape(un))
    
    var_list = np.zeros([np.shape(un)[0],np.size(units)])

    for ii in np.arange(0,np.size(un)):
    
         # first find the position of all gratings with this exact label
        inds = np.where(np.logical_and(actual_labels==un[ii], myinds_bool))[0]    
        
#        assert(np.shape(inds)[0]==48)
        
        # get the pooled variances - between the center and the left bin, and between the center and the right bin.
        var_within = np.var(allw[ww1][ww2][inds,:],0)

        var_list[ii,:] = var_within[units]
        
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

    for ss in units:
        plt.plot(un,var_list[:,ss])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('variance')
        plt.legend(legendlabs)
    else:
        plt.xticks([])
                 
    plt.suptitle('Variance of individual unit''s response to different phases at the same orientation\n%s, SF=%.2f, noise=%.2f' % (stim_types[tt],sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1
    