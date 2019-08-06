#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

#from matplotlib import cm
#from colorspacious import cspace_converter
#from collections import OrderedDict

#%% get the data ready to go...then can run any below cells independently.

model_str = 'inception_oriTst1'
do_recons = 1

root = '/usr/local/serenceslab/maggie/biasCNN/';
import os
os.chdir(os.path.join(root, 'code'))

import load_activations
#allw, all_labs, info = load_activations.load_activ_nasnet_oriTst0()
allw, all_labs, info = load_activations.load_activ(model_str)

# Isolate just the first timepoint, this means we'll need to save fewer recons
tmp = []
for ll in range (info['nLayers']):
    tmp.append([allw[ll][0]])
allw = tmp
info['nTimePts'] = 1
info['timepoint_labels'] = [info['timepoint_labels'][0]]
    

save_path = os.path.join(root, 'SNC_out', model_str)

#%%
# make all the reconstructions (only have to do this once)
if do_recons:
    
    layers2run = [0,1,2]
    
    import get_SNC_est
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    nUnits2Use = 100
    
    _ = get_SNC_est.run_withinsf_acrossphase(allw, layers2run, info, nUnits2Use, savename = os.path.join(save_path, 'SNC_est_withinsf_acrossphase'))
   
    _ = get_SNC_est.run_withinnoise_acrossphase(allw, layers2run, info, nUnits2Use, savename = os.path.join(save_path, 'SNC_est_withinnoise_acrossphase'))
    
    _ = get_SNC_est.run_withinnoise_acrosssf(allw, layers2run, info, nUnits2Use, savename = os.path.join(save_path, 'SNC_est_withinnoise_acrosssf'))
    
    _ = get_SNC_est.run_acrossnoise_acrossphase(allw, layers2run, info, 0, nUnits2Use, savename = os.path.join(save_path, 'SNC_est_trnzeronoise_acrossphase'))
    
    _ = get_SNC_est.run_acrossnoise_acrossphase(allw, layers2run, info, info['nNoiseLevels']-1, nUnits2Use, savename = os.path.join(save_path, 'SNC_est_trnmaxnoise_acrossphase'))

#%%
# extract some fields that will help us plot the recons in an organized way
center_deg=90
xx=info['xx']

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
#%% plot recons trained within a single SF, type, and noise level
layers2plot = [0,1,2]

SNC_all = [];

for ll in layers2plot:
           
    s = np.load(os.path.join(save_path, 'SNC_est_withinsf_acrossphase_' + info['layer_labels'][ll] + '.npy'))
    SNC_all.append(s)
        
plt.close('all')

tt=0
timepts2plot = [0]
noiselevels2plot =[0,1,2]
ylims = [0,180]
xlims = [0,180]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        # make one figure for all noise levels and SF
        plt.figure()
        ii=0  
            
        for sf in range(nSF):
            for nn in noiselevels2plot:
                ii=ii+1
                plt.subplot(nSF,np.size(noiselevels2plot),ii)
    
                # find my data
                inds = np.where(np.logical_and(np.logical_and(typelist==tt, sflist==sf), noiselist==nn))[0]
                                       
                real = info['orilist'][inds]
                est = SNC_all[ww1][0][ww2][inds]
                unc = SNC_all[ww1][1][ww2][inds]
       
        
                h, =plt.plot(real,est,'o')
   
                plt.ylim(ylims)
                plt.xlim(xlims)
                plt.plot([center_deg,center_deg], ylims)
                plt.plot(xlims, [center_deg,center_deg])
                plt.plot(xlims,ylims)
                
                plt.axis('equal')
                plt.axis('square')

                plt.title('noise=%.2f, SF=%.2f' % (info['noise_levels'][nn], sf_vals[sf]))
                if sf==nSF-1:
                    plt.xlabel('Actual Orientation (deg)')
                    plt.ylabel('Decoded Orientation (deg)')
                else:
                    plt.tick_params(axis='x', bottom=False,labelbottom = False)
   
        plt.suptitle('SNC predictions, trn/test within stimulus type and SF. \nWeights from %s - %s' % (info['layer_labels'][ww1], info['timepoint_labels'][ww2]))

#%% train the IEM across all SF and types, within noise level, train test across phase.
layers2plot = [0,1,2]

SNC_all = [];

for ll in layers2plot:
           
    s = np.load(os.path.join(save_path, 'SNC_est_withinnoise_acrosssf_' + info['layer_labels'][ll] + '.npy'))
    SNC_all.append(s)

plt.close('all')

timepts2plot = [0]
noiselevels2plot = [0,1,2]

ylims = [-1,1]

for nn in noiselevels2plot:
    ii=0;
    plt.figure()
    for ww1 in layers2plot:
       
        ii=ii+1

        myinds = np.where(noiselist==nn)[0]

        average_recons = IEM.shift_and_average(SNC_all_withinnoise[ww1][ww2][myinds,:],orilist[myinds],center_deg);
          
        plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
        plt.plot(xx,average_recons)
        plt.ylim(ylims)
        plt.plot([center_deg,center_deg], ylims,c=[0,0,0])
        plt.title('%s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
        if ww1==np.max(layers2plot)-2:
            plt.xlabel('Orientation Channel (deg)')
            plt.ylabel('Channel Activation Weight')
        else:
            plt.tick_params(axis='x', bottom=False,labelbottom = False)
            
        if (np.size(timepts2plot)>1 and ww2==np.max(timepts2plot)):
            
            plt.tick_params(axis='y', left=False,labelleft = False)
            plt.ylabel(None)
        
    plt.suptitle('Average reconstructions, leave one phase out\nnoise==%.2f' % noise_levels[nn])
    
#%% train the IEM across SF within noise level, overlay noise levels
        
SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
#SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))
plt.close('all')

layers2plot = np.arange(0,19,1)
timepts2plot = [0]
noiselevels2plot = [0,1,2]

ylims = [-1,1]

plt.figure()
legendlabs = []
lh = []

for nn in noiselevels2plot:
    ii=0;
    legendlabs.append('noise=%.2f' % noise_levels[nn])
    for ww1 in layers2plot:
       
        ii=ii+1

        myinds = np.where(noiselist==nn)[0]

        average_recons = IEM.shift_and_average(SNC_all_withinnoise[ww1][ww2][myinds,:],orilist[myinds],center_deg);
          
        ax=plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
        h, =plt.plot(xx,average_recons)
        if ww1==np.max(layers2plot):
            lh.append(h)
            if nn==np.max(noiselevels2plot):
                ax.legend(lh,legendlabs)
            
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
        
plt.suptitle('Average reconstructions, leave one phase out, collapsed over SF')

#%% train the IEM across phase, and across noise levels


layers2plot = [0]

trnNoise = 0

SNC_all = [];

for ll in layers2plot:
    
    if trnNoise==0:        
        s = np.load(os.path.join(save_path, 'SNC_est_trnzeronoise_acrossphase_' + info['layer_labels'][ll] + '.npy'))
        SNC_all.append(s)
    elif trnNoise==nNoiseLevels-1:
        s = np.load(os.path.join(save_path, 'recons_trnmaxnoise_acrossphase_' + info['layer_labels'][ll] + '.npy'))
        SNC_all.append(s)
    else:
        raise ValueError('check the value of trnNoise')
        
plt.close('all')

timepts2plot = [0]
ww2 = 0
#tstNoise = [0,1,2]
ylims = [0,180]
xlims = [0,180]
plt.figure()
#legendlabs = []
#lh = []
ii=0;
for nn2 in range(nNoiseLevels):
    ii=ii+1
    ax=plt.subplot(nNoiseLevels,1,ii)
#    legendlabs.append('test noise=%.2f' % noise_levels[nn2])
    for ww1 in range(np.size(layers2plot)):
       
        

        tstinds1 = np.where(noiselist==nn2)[0]
     
        real = info['orilist'][tstinds1]
        est = SNC_all[ww1][0][ww2][tstinds1]
        unc = SNC_all[ww1][1][ww2][tstinds1]
           
        
        h, =plt.plot(real,est,'o')
#        if ww1==np.size(layers2plot)-1:
#            lh.append(h)
#            if nn2==nNoiseLevels-1:
#                ax.legend(lh,legendlabs)
            
        plt.ylim(ylims)
        plt.xlim(xlims)
        plt.plot([center_deg,center_deg], ylims)
        plt.plot(xlims, [center_deg,center_deg])
        plt.plot(xlims,ylims)
        plt.title('Noise=%.2f, %s - %s' % (noise_levels[nn2], layer_labels[layers2plot[ww1]], timepoint_labels[ww2]))
#        if ww1==np.size(layers2plot)-2:
#            plt.xlabel('Orientation Channel (deg)')
#            plt.ylabel('Channel Activation Weight')
#        else:
#            plt.tick_params(axis='x', bottom=False,labelbottom = False)
            
#        if (np.size(timepts2plot)>1 and ww2==np.max(timepts2plot)):
            
#        plt.tick_params(axis='y', left=False,labelleft = False)
#        plt.ylabel(None)
        ax.axis('equal')    
plt.suptitle('Average reconstructions, leave one phase out\ntrain noise=%.2f' %(noise_levels[trnNoise]))

#%% train within noise across phase, separate the test set according to sf

SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))

#SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))

plt.close('all')

layers2plot = np.arange(0,19,1)
timepts2plot = [0]
nn = 0
tt = 0;
ww2 = 0;
ylims = [-1,2]

plt.figure()
legendlabs = []
for sf in range(nSF):
    legendlabs.append('sf=%.2f' %sf_vals[sf])
lh = []
ii=0;

for ww1 in layers2plot:
    
    ii=ii+1
       
    for sf in range(nSF):

        # average recons within just this spatial frequency
        inds = np.where(np.logical_and(np.logical_and(typelist==tt, sflist==sf), noiselist==nn))[0]  
          
        average_recons = IEM.shift_and_average(SNC_all_withinnoise[ww1][ww2][inds,:],orilist[inds,:],center_deg);
   
        ax=plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
        h, =plt.plot(xx,average_recons)
        if ww1==np.max(layers2plot):
            lh.append(h)
            if sf==nSF-1:
                ax.legend(lh,legendlabs)
            
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
    
plt.suptitle('Average reconstructions, leave one phase out\ntrain/test noise=%.2f' %(nn))

#%% Plot bias curves from each area, trn/test within noise level, w error bars

SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))

#SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))

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
   
  
    myinds = np.where(noiselist==nn)[0]
    orilist_now = orilist[myinds]

    pred_labels = xx[np.argmax(SNC_all_withinnoise[ww1][ww2][myinds,:], axis=1)]
    un = np.unique(orilist_now)
    avg_pred = np.zeros(np.shape(un))
    std_pred = np.zeros(np.shape(un))
   
    for uu in range(len(un)):
        avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(orilist_now==un[uu])[0]], high=180,low=0)
        std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(orilist_now==un[uu])[0]], high=180,low=0)

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
    plt.errorbar(un,avg_bias,std_pred,elinewidth=0.5)
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

#%% Plot bias curves from each area, trn/test within noise level
# plotting each spatial frequency separately  

#SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))
plt.close('all')

layers2plot = np.arange(0,nLayers,1)

ww2=0
nn=0
plt.figure()
legendlabs = []
for sf in range(nSF):
    legendlabs.append('sf=%.2f' %sf_vals[sf])
lh = []

ylims = [-25,25]

ii=0;
for ww1 in layers2plot:
       
    ii=ii+1
    ax = plt.subplot(np.ceil(len(layers2plot)/4),np.min([len(layers2plot),4]),ii)
    
    for sf in range(nSF):
        
        myinds = np.where(np.logical_and(noiselist==nn, sflist==sf))[0]
        
        pred_labels = xx[np.argmax(SNC_all_withinnoise[ww1][ww2][myinds], axis=1)]
        orilist_now = orilist[myinds]
        
        un = np.unique(orilist_now)
          
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
   
        for uu in range(len(un)):

            theseinds = np.where(orilist_now==un[uu])[0]
            avg_pred[uu] = scipy.stats.circmean(pred_labels[theseinds], high=180,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[theseinds], high=180,low=0)

         # calculate bias
        # first correct values that wrapped around
        avg_pred_corr = avg_pred
        indslow = np.where(np.logical_and(un<45, avg_pred>135))
        indshigh = np.where(np.logical_and(un>135, avg_pred<45))
        print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
        avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
        avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
        
        avg_bias = avg_pred_corr - un
    
        h, =plt.plot(un,avg_bias)
               
        if ww1==np.max(layers2plot):
            lh.append(h)
            if sf==nSF-1:
                 ax.legend(lh,legendlabs)
       

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
    
plt.suptitle('Reconstruction bias, train/test across phase\ntrain/test noise=%.2f' % noise_levels[nn])

#%% overlay the bias curves from each layer, compare them      
               
SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
plt.close('all')

layers2plot = [0,3,6,9,12]
timepts2plot = np.arange(0,1)
noiselevels2plot = [0]

legend_labs = [];

ylims = [-10,10]

for nn in noiselevels2plot:
    
    plt.figure()
    for ww1 in layers2plot:
        for ww2 in timepts2plot:
            legend_labs.append(layer_labels[ww1])

            myinds = np.where(noiselist==nn)[0]
        
            pred_labels = xx[np.argmax(SNC_all_withinnoise[ww1][ww2][myinds], axis=1)]
            orilist_now = orilist[myinds]
            un = np.unique(orilist_now)
            avg_pred = np.zeros(np.shape(un))
            std_pred = np.zeros(np.shape(un))
           
            for uu in range(len(un)):
                avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(orilist_now==un[uu])[0]], high=179,low=0)
                std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(orilist_now==un[uu])[0]], high=179,low=0)
               

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
    plt.title('Reconstruction bias, train/test across phase\nnoise=%.2f' % (noise_levels[nn]))
 
    plt.xlabel('Actual Orientation (deg)')
    plt.ylabel('Bias (deg)')
    plt.plot([0,180],[0,0],'k-')
    plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
    plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
    plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')

    plt.legend(legend_labs,bbox_to_anchor=(0.8,1))
   
#%% trn/test within noise level, overlay noise levels (plotting bias)
       
SNC_all_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
plt.close('all')

layers2plot = np.arange(0,nLayers,1)

ww2=0
nn=0
plt.figure()

legendlabs = []
for nn in range(nNoiseLevels):
    legendlabs.append('noise=%.2f' %noise_levels[nn])
lh = []
ii=0;

ylims = [-10,10]

ii=0;
for ww1 in layers2plot:
       
    ii=ii+1
    ax = plt.subplot(np.ceil(len(layers2plot)/4),np.min([len(layers2plot),4]),ii)
   
    for nn in range(nNoiseLevels):
        
        myinds = np.where(noiselist==nn)[0]
    
        pred_labels = xx[np.argmax(SNC_all_withinnoise[ww1][ww2][myinds,:], axis=1)]
        orilist_now = orilist[myinds]
        
        un = np.unique(orilist_now)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(orilist_now==un[uu])[0]], high=180,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(orilist_now==un[uu])[0]], high=180,low=0)
    
         # calculate bias
        # first correct values that wrapped around
        avg_pred_corr = avg_pred
        indslow = np.where(np.logical_and(un<45, avg_pred>135))
        indshigh = np.where(np.logical_and(un>135, avg_pred<45))
        print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
        avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
        avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
        
        avg_bias = avg_pred_corr - un
        
        h, = plt.plot(un,avg_bias)
#        plt.errorbar(un,avg_bias,std_pred,elinewidth=0.5)
    #    plt.plot(un,avg_bias,'k-')
        plt.title('%s' % (layer_labels[ww1]))
        
        if ww1==np.max(layers2plot)-2:
            plt.xlabel('Actual Orientation (deg)')
            plt.ylabel('Bias (deg)')
        else:
            plt.tick_params(axis='y', left=False,labelleft = False)
            plt.tick_params(axis='x', bottom=False,labelbottom = False)

        if ww1==np.max(layers2plot):
            lh.append(h)
            if nn==nNoiseLevels-1:
                ax.legend(lh,legendlabs)
 
        plt.xlim([0,180])
        plt.ylim(ylims)
        plt.plot([0,180],[0,0],'k-')
        plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
        plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
        plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
        
plt.suptitle('Reconstruction bias, train/test across phase\nwithin noise levels')

#%% trn/test across noise levels, plot bias
    
trnNoise = 0
if trnNoise==0:        
    SNC_all = np.load(os.path.join(save_path, 'recons_trnzeronoise_acrossphase.npy'))
elif trnNoise==nNoiseLevels-1:
    SNC_all = np.load(os.path.join(save_path, 'recons_trnmaxnoise_acrossphase.npy'))
else:
    raise ValueError('check the value of trnNoise')
   

plt.close('all')

layers2plot = range(nLayers)
ww2 = 0

ylims=[-10,10]


legendlabs = []
for nn in range(nNoiseLevels):
    legendlabs.append(' test noise=%.2f' %noise_levels[nn])
lh = []
ii=0;


for ww1 in layers2plot:
   
    ii=ii+1
    ax = plt.subplot(np.ceil(len(layers2plot)/4),np.min([len(layers2plot),4]),ii)
   
    for nn2 in range(nNoiseLevels):
       
        tstinds1 = np.where(noiselist==nn2)[0]
    
        pred_labels = xx[np.argmax(SNC_all[ww1][ww2][tstinds1,:], axis=1)]
        orilist_now = orilist[tstinds1]

        un = np.unique(orilist_now)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(orilist_now==un[uu])[0]], high=179,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(orilist_now==un[uu])[0]], high=179,low=0)

        # calculate bias
        # first correct values that wrapped around
        avg_pred_corr = avg_pred
        indslow = np.where(np.logical_and(un<45, avg_pred>135))
        indshigh = np.where(np.logical_and(un>135, avg_pred<45))
        print('layer %d: correcting %d + %d values for wraparound' % (ww1,len(indslow),len(indshigh)))
        avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
        avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
        
        avg_bias = avg_pred_corr - un
        
        h, = plt.plot(un,avg_bias)
#        plt.errorbar(un,avg_bias,std_pred,elinewidth=0.5)
    #    plt.plot(un,avg_bias,'k-')
        plt.title('%s' % (layer_labels[ww1]))
        
        if ww1==np.max(layers2plot)-2:
            plt.xlabel('Actual Orientation (deg)')
            plt.ylabel('Bias (deg)')
        else:
            plt.tick_params(axis='y', left=False,labelleft = False)
            plt.tick_params(axis='x', bottom=False,labelbottom = False)

        if ww1==np.max(layers2plot):
            lh.append(h)
            if nn2==nNoiseLevels-1:
                ax.legend(lh,legendlabs)
 
        plt.xlim([0,180])
        plt.ylim(ylims)
        plt.plot([0,180],[0,0],'k-')
        plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
        plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
        plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
        
plt.suptitle('Reconstruction bias, train/test across phase\nTRAINING noise = %.2f' %noise_levels[trnNoise])
