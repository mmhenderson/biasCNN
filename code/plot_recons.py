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
import IEM
#from matplotlib import cm
#from colorspacious import cspace_converter
#from collections import OrderedDict

#%% get the data ready to go...then can run any below cells independently.

model_str = 'inception_oriTst0'
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
    

save_path = os.path.join(root, 'recons', model_str)
# make all the reconstructions (only have to do this once)
if do_recons:
    
    import get_recons

    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    nUnits2Use = 100
    
    _ = get_recons.run_withinsf_acrossphase(allw, info, nUnits2Use, savename = os.path.join(save_path, 'recons_withinsf_acrossphase'))
   
    _ = get_recons.run_withinnoise_acrossphase(allw, info, nUnits2Use, savename = os.path.join(save_path, 'recons_withinnoise_acrossphase'))
    
    _ = get_recons.run_withinnoise_acrosssf(allw, info, nUnits2Use, savename = os.path.join(save_path, 'recons_withinnoise_acrosssf'))
    
    _ = get_recons.run_acrossnoise_acrossphase(allw, info, 0, nUnits2Use, savename = os.path.join(save_path, 'recons_trnzeronoise_acrossphase'))
    
    _ = get_recons.run_acrossnoise_acrossphase(allw, info, info['nNoiseLevels']-1, nUnits2Use, savename = os.path.join(save_path, 'recons_trnmaxnoise_acrossphase'))


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

chan_resp_withinsf = np.load(os.path.join(save_path, 'recons_withinsf_acrossphase.npy'))

plt.close('all')

layers2plot = [2]
timepts2plot = [0]
noiselevels2plot =[0]
ylims = [-1,1]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            
            # make one figure for all types and SF
            plt.figure()
            ii=0      
            for sf in range(nSF):
                 for tt in range(nType):
                    ii=ii+1
                    plt.subplot(nSF,nType,ii)
     
                    # find my data
                    inds = np.where(np.logical_and(np.logical_and(typelist==tt, sflist==sf), noiselist==nn))[0]
                   
                    average_recons = IEM.shift_and_average(chan_resp_withinsf[ww1][ww2][inds,:],orilist[inds],center_deg);
                    
                    plt.plot(xx,average_recons)
                    plt.ylim(ylims)
                    plt.plot([center_deg,center_deg], ylims)
                    plt.title('%s, SF=%.2f' % (stim_types[tt],sf_vals[sf]))
                    if sf==nSF-1:
                        plt.xlabel('Orientation Channel (deg)')
                        plt.ylabel('Channel Activation Weight')
                    else:
                        plt.tick_params(axis='x', bottom=False,labelbottom = False)
   
            plt.suptitle('Average reconstruction, trn/test within stimulus type and SF. \nWeights from %s - %s, noise=%.2f' % (info['layer_labels'][ww1], info['timepoint_labels'][ww2], info['noise_levels'][nn]))

#%% train the IEM across all SF and types, within noise level, train test across phase.
 
#chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))
plt.close('all')

layers2plot = np.arange(0,19,1)
timepts2plot = [0]
noiselevels2plot = [0]

ylims = [-1,1]

for nn in noiselevels2plot:
    ii=0;
    plt.figure()
    for ww1 in layers2plot:
       
        ii=ii+1

        myinds = np.where(noiselist==nn)[0]

        average_recons = IEM.shift_and_average(chan_resp_withinnoise[ww1][ww2][myinds,:],orilist[myinds],center_deg);
          
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
        
chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
#chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))
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

        average_recons = IEM.shift_and_average(chan_resp_withinnoise[ww1][ww2][myinds,:],orilist[myinds],center_deg);
          
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

trnNoise = 0
if trnNoise==0:        
    chan_resp = np.load(os.path.join(save_path, 'recons_trnzeronoise_acrossphase.npy'))
elif trnNoise==nNoiseLevels-1:
    chan_resp = np.load(os.path.join(save_path, 'recons_trnmaxnoise_acrossphase.npy'))
else:
    raise ValueError('check the value of trnNoise')
    
plt.close('all')

layers2plot = np.arange(0,19,1)
timepts2plot = [0]
#tstNoise = [0,1,2]
ylims = [-1,1]

plt.figure()
legendlabs = []
lh = []

for nn2 in range(nNoiseLevels):
    ii=0;
    legendlabs.append('test noise=%.2f' % noise_levels[nn2])
    for ww1 in layers2plot:
       
        ii=ii+1

        tstinds1 = np.where(noiselist==nn2)[0]
     
        average_recons = IEM.shift_and_average(chan_resp[ww1][ww2][tstinds1],orilist[tstinds1],center_deg);
          
        ax=plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
        h, =plt.plot(xx,average_recons)
        if ww1==np.max(layers2plot):
            lh.append(h)
            if nn2==nNoiseLevels-1:
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
            
plt.suptitle('Average reconstructions, leave one phase out\ntrain noise=%.2f' %(noise_levels[trnNoise]))

#%% train within noise across phase, separate the test set according to sf

chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))

#chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))

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
          
        average_recons = IEM.shift_and_average(chan_resp_withinnoise[ww1][ww2][inds,:],orilist[inds,:],center_deg);
   
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

chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))

#chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))

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

    pred_labels = xx[np.argmax(chan_resp_withinnoise[ww1][ww2][myinds,:], axis=1)]
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

#chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrosssf.npy'))
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
        
        pred_labels = xx[np.argmax(chan_resp_withinnoise[ww1][ww2][myinds], axis=1)]
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
               
chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
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
        
            pred_labels = xx[np.argmax(chan_resp_withinnoise[ww1][ww2][myinds], axis=1)]
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
       
chan_resp_withinnoise = np.load(os.path.join(save_path, 'recons_withinnoise_acrossphase.npy'))
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
    
        pred_labels = xx[np.argmax(chan_resp_withinnoise[ww1][ww2][myinds,:], axis=1)]
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
    chan_resp = np.load(os.path.join(save_path, 'recons_trnzeronoise_acrossphase.npy'))
elif trnNoise==nNoiseLevels-1:
    chan_resp = np.load(os.path.join(save_path, 'recons_trnmaxnoise_acrossphase.npy'))
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
    
        pred_labels = xx[np.argmax(chan_resp[ww1][ww2][tstinds1,:], axis=1)]
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
