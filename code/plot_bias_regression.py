#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt
import scipy
import scipy.stats
import circ_reg_tools  
import numpy as np

#%% get the data ready to go...then can run any below cells independently.

#model_str = 'inception_oriTst4a'
#model_name_2plot = 'Inception-V3'
#
#model_str = 'nasnet_oriTst4a'
#model_name_2plot = 'NASnet'

model_str = 'vgg16_oriTst11'
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

#%% Run circular regression model - plot predicted versus actual

plt.close('all')

nn = 0
sf = 0
ww2 = 0
layers2plot = np.arange(0,nLayers,1)
tt=0

#c = np.zeros([nLayers,nPhase])
c_all = np.zeros([nLayers,1])


plt.figure()
xx=1

for ww1 in layers2plot:
    
    w = allw[ww1][ww2]
    
    inds = np.where(np.all([sflist==sf, noiselist==nn,typelist==tt],axis=0))[0]
    
    phase = phaselist[inds]
    un_phase = np.unique(phase)
    
    all_dat = w[inds,:]
    real_ori = orilist[inds]
    pred_ori = np.zeros(np.shape(real_ori))
    
    for pp in un_phase:

        trninds = np.where(phase!=pp)[0]
        tstinds = np.where(phase==pp)[0]
        
        predlabs, corr = circ_reg_tools.circ_regression(all_dat[trninds,:],real_ori[trninds], all_dat[tstinds,:], real_ori[tstinds])
#        c[ww1,pp] = corr
        pred_ori[tstinds] = predlabs
        
    # make sure to convert to radians here!!    
    corr_all = classifiers.circ_corr_coef(real_ori/180*2*np.pi,pred_ori/180*2*np.pi)
    c_all[ww1,0] = corr_all
        
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
  
    plt.plot(real_ori,pred_ori,'.')
    plt.plot([0,180],[0,180],'k')
    plt.plot([0,180],[90,90],'k')
    plt.plot([90,90],[0,180],'k')

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('decoded orientation')
#        plt.legend(legendlabs)
    else:
        plt.xticks([]);plt.yticks([])
   
    plt.axis('square')
    
    plt.suptitle('Circular regression predictions\nSF=%.2f, noise=%.2f' % (sf_vals[sf],noise_levels[nn]))
             
    xx=xx+1

#%% Run circular regression model across SF- plot error

plt.close('all')

nn=2

#sf = 2
ww2 = 0
layers2plot = np.arange(0,nLayers,1)
tt=0

#c = np.zeros([nLayers,nPhase])
c_all = np.zeros([nLayers,1])


plt.figure()
xx=1

ylims =[-10,10]

for ww1 in layers2plot:
    
    w = allw[ww1][ww2]
    
    inds = np.where(np.all([noiselist==nn,typelist==tt],axis=0))[0]
    
    phase = phaselist[inds]
    un_phase = np.unique(phase)
    
    all_dat = w[inds,:]
    real_ori = orilist[inds]
    pred_ori = np.zeros(np.shape(real_ori))
    
    for pp in un_phase:

        trninds = np.where(phase!=pp)[0]
        tstinds = np.where(phase==pp)[0]
        
#        predlabs, corr = classifiers.circ_regression(all_dat[tstinds,:],real_ori[tstinds], all_dat[tstinds,:], real_ori[tstinds])
        predlabs, corr = classifiers.circ_regression(all_dat[trninds,:],real_ori[trninds], all_dat[tstinds,:], real_ori[tstinds])
#        c[ww1,pp] = corr
        pred_ori[tstinds] = predlabs
        
    # make sure to convert to radians here!!    
    corr_all = classifiers.circ_corr_coef(real_ori/180*2*np.pi,pred_ori/180*2*np.pi)
    c_all[ww1,0] = corr_all
    
    
    # get prediction error. there are two difference values to choose from - take the smallest one.
    errlist_raw = np.min([np.abs(pred_ori-real_ori), 180-np.abs(pred_ori-real_ori)], axis=0);

    # now decide on the sign of this error. 
    # 1. If the angle is obtuse (>90) and the target is > the response, then the error is in the clockwise
    # direction. 
    # 2. If the angle is obtuse and the target is < the response, then
    # the error is in the counter-clockwise direction. 
    # 3. If the angle is <90 and the target is > the response, then the error is
    # counter-clockwise. 
    # 4. If the angle is <90 and the target is <the response then the error is clockwise. 
    obtuse_inds = np.abs(pred_ori-real_ori)>90;
    inds_flip = np.where(np.logical_or( np.logical_and(obtuse_inds, real_ori<pred_ori), np.logical_and(~obtuse_inds, real_ori>pred_ori)))[0]
    errlist_raw[inds_flip] = -errlist_raw[inds_flip];
  
    un = np.unique(real_ori)
    err_mean = np.zeros(np.shape(un))
    err_std = np.zeros(np.shape(un))
    for uu in range(np.size(un)):
        
        inds =np.where(real_ori==un[uu])[0]
        
        err_mean[uu] = np.mean(errlist_raw[inds])
        err_std[uu] = np.std(errlist_raw[inds])
        
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
  
    plt.ylim(ylims)
#    plt.plot(real_ori,errlist_raw,'.')
    plt.errorbar(un,err_mean, err_std, elinewidth=0.5)
    plt.plot([0,180],[0,0],'k')
#    plt.plot([0,180],[90,90],'k')
    plt.plot([90,90],ylims,'k')

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating (deg)')
        plt.ylabel('signed error (deg)')
#        plt.legend(legendlabs)
    else:
        plt.xticks([]);plt.yticks([])
   
#    plt.axis('square')
    
    plt.suptitle('%s\nCircular regression predictions\nAll SF, noise=%.2f' % (model_str,noise_levels[nn]))
             
    xx=xx+1
 
                       
#%% Run circular regression model within SF, overlay SF - plot error

plt.close('all')


nn = 0
sf2plot = [0,1,2]
ww2 = 0
layers2plot = np.arange(0,nLayers,1)
tt=0

#c = np.zeros([nLayers,nPhase])
c_all = np.zeros([nLayers,1])


plt.figure()
xx=1

ylims =[-10,10]

legendlabs = ['sf=%.2f'%sf_vals[sf] for sf in range(np.size(sf2plot))]

for ww1 in layers2plot:
    
    w = allw[ww1][ww2]
    
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    lh =[]
    #% start sf loop
    for sf in sf2plot:
            
        inds = np.where(np.all([sflist==sf, noiselist==nn,typelist==tt],axis=0))[0]
        
        phase = phaselist[inds]
        un_phase = np.unique(phase)
        
        all_dat = w[inds,:]
        real_ori = orilist[inds]
        pred_ori = np.zeros(np.shape(real_ori))
        
        for pp in un_phase:
    
            trninds = np.where(phase!=pp)[0]
            tstinds = np.where(phase==pp)[0]
            
            predlabs, corr = classifiers.circ_regression(all_dat[trninds,:],real_ori[trninds], all_dat[tstinds,:], real_ori[tstinds])
    #        c[ww1,pp] = corr
            pred_ori[tstinds] = predlabs
            
        # make sure to convert to radians here!!    
        corr_all = classifiers.circ_corr_coef(real_ori/180*2*np.pi,pred_ori/180*2*np.pi)
        c_all[ww1,0] = corr_all
        
        
        # get prediction error. there are two difference values to choose from - take the smallest one.
        errlist_raw = np.min([np.abs(pred_ori-real_ori), 180-np.abs(pred_ori-real_ori)], axis=0);
    
        # now decide on the sign of this error. 
        # 1. If the angle is obtuse (>90) and the target is > the response, then the error is in the clockwise
        # direction. 
        # 2. If the angle is obtuse and the target is < the response, then
        # the error is in the counter-clockwise direction. 
        # 3. If the angle is <90 and the target is > the response, then the error is
        # counter-clockwise. 
        # 4. If the angle is <90 and the target is <the response then the error is clockwise. 
        obtuse_inds = np.abs(pred_ori-real_ori)>90;
        inds_flip = np.where(np.logical_or( np.logical_and(obtuse_inds, real_ori<pred_ori), np.logical_and(~obtuse_inds, real_ori>pred_ori)))[0]
        errlist_raw[inds_flip] = -errlist_raw[inds_flip];
      
        un = np.unique(real_ori)
        err_mean = np.zeros(np.shape(un))
        err_std = np.zeros(np.shape(un))
        
        for uu in range(np.size(un)):
            
            inds =np.where(real_ori==un[uu])[0]
            
            err_mean[uu] = np.mean(errlist_raw[inds])
            err_std[uu] = np.std(errlist_raw[inds])
            
        lh.append(plt.errorbar(un,err_mean, err_std, elinewidth=0.5))
    
    #% end sf loop
    
    plt.ylim(ylims)
#    plt.plot(real_ori,errlist_raw,'.')
   
    plt.plot([0,180],[0,0],'k')
#    plt.plot([0,180],[90,90],'k')
    plt.plot([90,90],ylims,'k')

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating (deg)')
        plt.ylabel('signed error (deg)')
        plt.figlegend(lh,legendlabs)
    else:
        plt.xticks([]);plt.yticks([])
   
    #    plt.axis('square')
        
                
    xx=xx+1

plt.suptitle('%s\nCircular regression predictions\nTrn/test within SF, noise=%.2f' % (model_str,noise_levels[nn]))
         
                       
#%% Run circular regression model within noise, overlay noise levels - plot error

plt.close('all')

noise2plot = [0,1,2]
sf = 0
ww2 = 0
layers2plot = np.arange(0,nLayers,1)
tt=0

#c = np.zeros([nLayers,nPhase])
c_all = np.zeros([nLayers,1])


plt.figure()
xx=1

ylims =[-10,10]

legendlabs = ['noise=%.2f'%noise_levels[nn] for nn in range(np.size(noise2plot))]

for ww1 in layers2plot:
    
    w = allw[ww1][ww2]
    
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    lh =[]
    #% start loop
    for nn in noise2plot:
            
        inds = np.where(np.all([sflist==sf, noiselist==nn,typelist==tt],axis=0))[0]
        
        phase = phaselist[inds]
        un_phase = np.unique(phase)
        
        all_dat = w[inds,:]
        real_ori = orilist[inds]
        pred_ori = np.zeros(np.shape(real_ori))
        
        for pp in un_phase:
    
            trninds = np.where(phase!=pp)[0]
            tstinds = np.where(phase==pp)[0]
            
            predlabs, corr = classifiers.circ_regression(all_dat[trninds,:],real_ori[trninds], all_dat[tstinds,:], real_ori[tstinds])
    #        c[ww1,pp] = corr
            pred_ori[tstinds] = predlabs
            
        # make sure to convert to radians here!!    
        corr_all = classifiers.circ_corr_coef(real_ori/180*2*np.pi,pred_ori/180*2*np.pi)
        c_all[ww1,0] = corr_all
        
        
        # get prediction error. there are two difference values to choose from - take the smallest one.
        errlist_raw = np.min([np.abs(pred_ori-real_ori), 180-np.abs(pred_ori-real_ori)], axis=0);
    
        # now decide on the sign of this error. 
        # 1. If the angle is obtuse (>90) and the target is > the response, then the error is in the clockwise
        # direction. 
        # 2. If the angle is obtuse and the target is < the response, then
        # the error is in the counter-clockwise direction. 
        # 3. If the angle is <90 and the target is > the response, then the error is
        # counter-clockwise. 
        # 4. If the angle is <90 and the target is <the response then the error is clockwise. 
        obtuse_inds = np.abs(pred_ori-real_ori)>90;
        inds_flip = np.where(np.logical_or( np.logical_and(obtuse_inds, real_ori<pred_ori), np.logical_and(~obtuse_inds, real_ori>pred_ori)))[0]
        errlist_raw[inds_flip] = -errlist_raw[inds_flip];
      
        un = np.unique(real_ori)
        err_mean = np.zeros(np.shape(un))
        err_std = np.zeros(np.shape(un))
        
        for uu in range(np.size(un)):
            
            inds =np.where(real_ori==un[uu])[0]
            
            err_mean[uu] = np.mean(errlist_raw[inds])
            err_std[uu] = np.std(errlist_raw[inds])
            
        lh.append(plt.errorbar(un,err_mean, err_std, elinewidth=0.5))
    
    #% end loop
    
    plt.ylim(ylims)
#    plt.plot(real_ori,errlist_raw,'.')
   
    plt.plot([0,180],[0,0],'k')
#    plt.plot([0,180],[90,90],'k')
    plt.plot([90,90],ylims,'k')

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating (deg)')
        plt.ylabel('signed error (deg)')
        plt.figlegend(lh,legendlabs)
    else:
        plt.xticks([]);plt.yticks([])
   
    #    plt.axis('square')
        
                
    xx=xx+1

plt.suptitle('%s\nCircular regression predictions\nTrn/test within noise level, SF=%.2f' % (model_str,sf_vals[sf]))
                                
#%% Run circular regression model on low noise, overlay noise levels - plot error

plt.close('all')

noise_trn = 0

noise2plot = [0,1,2]
sf = 0
ww2 = 0
layers2plot = np.arange(0,,1)
tt=0

#c = np.zeros([nLayers,nPhase])
c_all = np.zeros([nLayers,1])


plt.figure()
xx=1

ylims =[-10,10]

legendlabs = ['train noise %.2f, test noise=%.2f'%(noise_levels[noise_trn], noise_levels[nn]) for nn in range(np.size(noise2plot))]

for ww1 in layers2plot:
    
    w = allw[ww1][ww2]
    
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    lh =[]
    #% start loop
    for nn in noise2plot:
            
        # first, separate the data by noise level so we can train/test across noise
        trninds1 = np.where(np.all([sflist==sf,noiselist==noise_trn,typelist==tt],axis=0))[0]
        tstinds1 = np.where(np.all([sflist==sf,noiselist==nn,typelist==tt],axis=0))[0]
        
        trnphase = phaselist[trninds1]
        tstphase = phaselist[tstinds1]
        un_phase = np.unique(trnphase)
        
        trndat = w[trninds1,:]
        tstdat = w[tstinds1,:]
        
        real_ori = orilist[tstinds1]
        pred_ori = np.zeros(np.shape(real_ori))
        
        trnori = orilist[trninds1]
        
        for pp in un_phase:
    
            # now we also want to cross-train w/r/t instance of each orientation (phase)
            trninds2 = np.where(trnphase!=pp)[0]
            tstinds2 = np.where(tstphase==pp)[0]
            
            predlabs, corr = classifiers.circ_regression(trndat[trninds2,:],trnori[trninds2], tstdat[tstinds2,:], real_ori[tstinds2])
    #        c[ww1,pp] = corr
            pred_ori[tstinds2] = predlabs
            
        # make sure to convert to radians here!!    
        corr_all = classifiers.circ_corr_coef(real_ori/180*2*np.pi,pred_ori/180*2*np.pi)
        c_all[ww1,0] = corr_all
        
        
        # get prediction error. there are two difference values to choose from - take the smallest one.
        errlist_raw = np.min([np.abs(pred_ori-real_ori), 180-np.abs(pred_ori-real_ori)], axis=0);
    
        # now decide on the sign of this error. 
        # 1. If the angle is obtuse (>90) and the target is > the response, then the error is in the clockwise
        # direction. 
        # 2. If the angle is obtuse and the target is < the response, then
        # the error is in the counter-clockwise direction. 
        # 3. If the angle is <90 and the target is > the response, then the error is
        # counter-clockwise. 
        # 4. If the angle is <90 and the target is <the response then the error is clockwise. 
        obtuse_inds = np.abs(pred_ori-real_ori)>90;
        inds_flip = np.where(np.logical_or( np.logical_and(obtuse_inds, real_ori<pred_ori), np.logical_and(~obtuse_inds, real_ori>pred_ori)))[0]
        errlist_raw[inds_flip] = -errlist_raw[inds_flip];
      
        un = np.unique(real_ori)
        err_mean = np.zeros(np.shape(un))
        err_std = np.zeros(np.shape(un))
        
        for uu in range(np.size(un)):
            
            inds =np.where(real_ori==un[uu])[0]
            
            err_mean[uu] = np.mean(errlist_raw[inds])
            err_std[uu] = np.std(errlist_raw[inds])
            
        lh.append(plt.errorbar(un,err_mean, err_std, elinewidth=0.5))
    
    #% end loop
    
    plt.ylim(ylims)
#    plt.plot(real_ori,errlist_raw,'.')
   
    plt.plot([0,180],[0,0],'k')
#    plt.plot([0,180],[90,90],'k')
    plt.plot([90,90],ylims,'k')

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating (deg)')
        plt.ylabel('signed error (deg)')
        plt.figlegend(lh,legendlabs)
    else:
        plt.xticks([]);plt.yticks([])
   
    #    plt.axis('square')
        
                
    xx=xx+1

plt.suptitle('%s\nCircular regression predictions\nTrain on noise=%.2f, SF=%.2f' % (model_str,noise_levels[noise_trn],sf_vals[sf]))
         
#%% Run circular regression model within noise, across all SF, overlay noise levels - plot error

plt.close('all')

noise2plot = [0,1,2]
#sf = 2
ww2 = 0
layers2plot = np.arange(0,nLayers,1)
tt=0

#c = np.zeros([nLayers,nPhase])
c_all = np.zeros([nLayers,1])


plt.figure()
xx=1

ylims =[-20,20]

legendlabs = ['noise=%.2f'%noise_levels[nn] for nn in range(np.size(noise2plot))]

for ww1 in layers2plot:
    
    w = allw[ww1][ww2]
    
    plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    lh =[]
    #% start loop
    for nn in noise2plot:
            
        inds = np.where(np.all([noiselist==nn,typelist==tt],axis=0))[0]
        
        phase = phaselist[inds]
        un_phase = np.unique(phase)
        
        all_dat = w[inds,:]
        real_ori = orilist[inds]
        pred_ori = np.zeros(np.shape(real_ori))
        
        for pp in un_phase:
    
            trninds = np.where(phase!=pp)[0]
            tstinds = np.where(phase==pp)[0]
            
            predlabs, corr = classifiers.circ_regression(all_dat[trninds,:],real_ori[trninds], all_dat[tstinds,:], real_ori[tstinds])
    #        c[ww1,pp] = corr
            pred_ori[tstinds] = predlabs
            
        # make sure to convert to radians here!!    
        corr_all = classifiers.circ_corr_coef(real_ori/180*2*np.pi,pred_ori/180*2*np.pi)
        c_all[ww1,0] = corr_all
        
        
        # get prediction error. there are two difference values to choose from - take the smallest one.
        errlist_raw = np.min([np.abs(pred_ori-real_ori), 180-np.abs(pred_ori-real_ori)], axis=0);
    
        # now decide on the sign of this error. 
        # 1. If the angle is obtuse (>90) and the target is > the response, then the error is in the clockwise
        # direction. 
        # 2. If the angle is obtuse and the target is < the response, then
        # the error is in the counter-clockwise direction. 
        # 3. If the angle is <90 and the target is > the response, then the error is
        # counter-clockwise. 
        # 4. If the angle is <90 and the target is <the response then the error is clockwise. 
        obtuse_inds = np.abs(pred_ori-real_ori)>90;
        inds_flip = np.where(np.logical_or( np.logical_and(obtuse_inds, real_ori<pred_ori), np.logical_and(~obtuse_inds, real_ori>pred_ori)))[0]
        errlist_raw[inds_flip] = -errlist_raw[inds_flip];
      
        un = np.unique(real_ori)
        err_mean = np.zeros(np.shape(un))
        err_std = np.zeros(np.shape(un))
        
        for uu in range(np.size(un)):
            
            inds =np.where(real_ori==un[uu])[0]
            
            err_mean[uu] = np.mean(errlist_raw[inds])
            err_std[uu] = np.std(errlist_raw[inds])
            
        lh.append(plt.errorbar(un,err_mean, err_std, elinewidth=0.5))
    
    #% end loop
    
    plt.ylim(ylims)
#    plt.plot(real_ori,errlist_raw,'.')
   
    plt.plot([0,180],[0,0],'k')
#    plt.plot([0,180],[90,90],'k')
    plt.plot([90,90],ylims,'k')

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating (deg)')
        plt.ylabel('signed error (deg)')
        plt.figlegend(lh,legendlabs)
    else:
        plt.xticks([]);plt.yticks([])
   
    #    plt.axis('square')
        
                
    xx=xx+1

plt.suptitle('%s\nCircular regression predictions\nTrn/test within noise level, All SF' % (model_str))
         