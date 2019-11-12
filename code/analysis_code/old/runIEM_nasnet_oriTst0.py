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
#%% run the IEM - within each stimulus type and sf separately. This gives really clean recons everywhere

plt.close('all')

layers2plot = np.arange(0,nLayers,6)
#layers2plot = np.arange(0,1)
timepts2plot = np.arange(0,1)
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
                    inds = np.where(np.logical_and(typelist==tt, sflist==sf))[0]
                           
                    ori_labs = actual_labels[inds]
#                    ori_labs_shuff = ori_labs
#                    np.random.shuffle(ori_labs_shuff)
                    center_deg=90
                    n_chans=9
                    n_folds = 10
                    
                    alldat = allw[ww1][nn][ww2][inds,0:nVox2Use]
    
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
#%% train the IEM across stim type (gauss versus full-field) and across SF. Gives negative baseline shift.
        
plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,1)
#layers2plot = np.arange(0,nLayers,6)
timepts2plot = np.arange(0,1)
#layers2plot = [14]
noiselevels2plot = [0]
ylims = [-3,-1]
nVox2Use = 100

for nn in noiselevels2plot:
    ii=0;
    plt.figure()
    for ww1 in layers2plot:
#        for ww2 in timepts2plot:
        
        ii=ii+1
        plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
#        plt.subplot(np.size(layers2plot),np.size(timepts2plot),ii)
    
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
#        n_folds = 10
        
#        alldat = allw[ww1][ww2]
        alldat = allw[ww1][nn][ww2][:,0:nVox2Use]
        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
#        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,10,9)
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist,sflist), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
            
            average_recons = IEM.shift_and_average(chan_resp_all[tstinds,:],ori_labs[tstinds],center_deg);
                
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
    
#%% train/test across spatial frequency only, but within a stim type. Better
           
plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,1)
#layers2plot = np.arange(0,nLayers,6)
timepts2plot = np.arange(0,1)
#layers2plot = [14]
noiselevels2plot = [0]
ylims = [-1,1]
nVox2Use = 100
tt=1

for nn in noiselevels2plot:
    ii=0;
    plt.figure()
    for ww1 in layers2plot:
#        for ww2 in timepts2plot:
        
        ii=ii+1
#        plt.subplot(np.size(layers2plot),np.size(timepts2plot),ii)
    
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
#        n_folds = 10
        
#        alldat = allw[ww1][ww2]
        
        myinds = np.where(typelist==tt)[0]
        
        alldat = allw[ww1][nn][ww2][myinds,0:nVox2Use]
        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
        ori_labs_now = ori_labs[myinds]
#        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,10,9)
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist[myinds],sflist[myinds]), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs_now[trninds],alldat[tstinds,:],n_chans)
            
        average_recons = IEM.shift_and_average(chan_resp_all,ori_labs_now,center_deg);
          
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
        
    plt.suptitle('Average reconstructions, leave one spatial freq out, \n%s, noise=%.2f' % (stim_types[tt],noise_levels[nn]))

#%% train/test across spatial frequency only, within a stim type, now doing this within multiple noise levels and overlaying. 
           
plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,1)
#layers2plot = np.arange(0,nLayers,6)
timepts2plot = np.arange(0,1)
#layers2plot = [14]
noiselevels2plot = [0,1,2,3,4]
ylims = [-1,1]
nVox2Use = 100
tt=0

plt.figure()
legendlabs = []
lh = []

for nn in noiselevels2plot:
    ii=0;
    legendlabs.append('noise=%.2f' % noise_levels[nn])
    for ww1 in layers2plot:
#        for ww2 in timepts2plot:
        
        ii=ii+1
#        plt.subplot(np.size(layers2plot),np.size(timepts2plot),ii)
    
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
#        n_folds = 10
        
#        alldat = allw[ww1][ww2]
        
        myinds = np.where(typelist==tt)[0]
        
        alldat = allw[ww1][nn][ww2][myinds,0:nVox2Use]
#        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
        ori_labs_now = ori_labs[myinds]
#        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,10,9)
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist[myinds],sflist[myinds]), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs_now[trninds],alldat[tstinds,:],n_chans)
            
        average_recons = IEM.shift_and_average(chan_resp_all,ori_labs_now,center_deg);
          
           
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
        
    plt.suptitle('Average reconstructions, leave one spatial freq out, \n%s' % (stim_types[tt]))
 
    
#%% train the IEM across SF and across noise level.
        
plt.close('all')

layers2plot = np.arange(0,19,1)
timepts2plot = [0]

nn1=0
tstNoise = [0,1,2,3,4]
tt=1
ww2=0

ylims = [-1,1]

nVox2Use = 100

plt.figure()
legendlabs = []
lh = []

for nn2 in tstNoise:
    ii=0;
    legendlabs.append('test noise=%.2f' % noise_levels[nn2])
    for ww1 in layers2plot:
       
        ii=ii+1
    
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
        
        # get just the one stim type of interest
        myinds = np.where(typelist==tt)[0]
        orilabs_here = ori_labs[myinds]
        
        # first separate out the training/testing noise levels
        alldat_trn = allw[ww1][nn1][ww2][myinds,0:nVox2Use]
        
        alldat_tst = allw[ww1][nn2][ww2][myinds,0:nVox2Use]
 
        chan_resp_all = np.zeros([np.shape(alldat_tst)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist[myinds],sflist[myinds]), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
    #            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat_trn[trninds,:],orilabs_here[trninds],alldat_tst[tstinds,:],n_chans)
            
        average_recons = IEM.shift_and_average(chan_resp_all,orilabs_here,center_deg);
          
        ax=plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
        h, =plt.plot(xx,average_recons)
        if ww1==np.max(layers2plot):
            lh.append(h)
            if nn2==np.max(tstNoise):
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
            
plt.suptitle('Average reconstructions, leave one spatial freq out\n%s, train noise=%.2f' %(stim_types[tt], noise_levels[nn1]))
    
#%% train/test across stim type, but within spatial freq. Within just one noise level per plot.
           
plt.close('all')
#plt.figure()
layers2plot = np.arange(0,nLayers,1)
#layers2plot = np.arange(0,nLayers,6)
timepts2plot = np.arange(0,1)
#layers2plot = [14]
noiselevels2plot = [0]
ylims = [-1,1]
nVox2Use = 100
bb=0

ww2=0;

for nn in noiselevels2plot:
    ii=0;
    plt.figure()
    for ww1 in layers2plot:
#        for ww2 in timepts2plot:
        
        ii=ii+1
#        plt.subplot(np.size(layers2plot),np.size(timepts2plot),ii)
    
        ori_labs = actual_labels
        center_deg=90
        n_chans=9
#        n_folds = 10
        
#        alldat = allw[ww1][ww2]
        
        myinds = np.where(sflist==bb)[0]
        
        alldat = allw[ww1][0][ww2][myinds,0:nVox2Use]
        alldat = alldat - np.tile(np.expand_dims(np.mean(alldat,axis=1),1), [1,np.shape(alldat)[1]])
        ori_labs_now = ori_labs[myinds]
#        chan_resp_all = IEM.run_crossval_IEM(alldat,ori_labs,10,9)
#         cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist[myinds],sflist[myinds]), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
#            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs_now[trninds],alldat[tstinds,:],n_chans)
            
            
        inds1 = np.where(whichCV==0)[0]
        inds2 = np.where(whichCV==1)[0]
        
        average_recons1 = IEM.shift_and_average(chan_resp_all[inds1],ori_labs_now[inds1],center_deg);
        average_recons2 = IEM.shift_and_average(chan_resp_all[inds2],ori_labs_now[inds2],center_deg);
          
       
        plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
        plt.plot(xx,average_recons1)
        plt.plot(xx,average_recons2)
        plt.ylim(ylims)
        plt.plot([center_deg,center_deg], ylims)
        plt.title('%s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
        if ww1==np.max(layers2plot)-2:
            plt.xlabel('Orientation Channel (deg)')
            plt.ylabel('Channel Activation Weight')
        else:
            plt.tick_params(axis='x', bottom=False,labelbottom = False)
            
        if (np.size(timepts2plot)>1 and ww1==np.max(timepts2plot)):
            
            plt.tick_params(axis='y', left=False,labelleft = False)
            plt.ylabel(None)
        
    plt.suptitle('Average reconstructions, leave one stim type out \nsf=%.2f, noise=%.2f' % (sf_vals[bb],noise_levels[nn]))
#%% train across SF, separate the test set according to sf

plt.close('all')

layers2plot = np.arange(0,nLayers,1)
#layers2plot = np.arange(0,1)
#layers2plot = [0]
timepts2plot = np.arange(0,1)

tt=1
ww2=0
nn=0

ylims = [-1.5,1.5]

nVox2Use = 100

plt.figure()
ii=0
legendlabs = [];
lh =[];

for ww1 in layers2plot:
       
    ii=ii+1
    ax=plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
    plt.title('%s' % layer_labels[ww1])
    
    ori_labs = actual_labels
    center_deg=90
    n_chans=9
#        n_folds = 20
    
    inds = np.where(typelist==tt)[0]

    ori_labs = actual_labels[inds]

    # run the IEM across all stims as the training set
    alldat = allw[ww1][0][ww2][inds,0:nVox2Use]
    
    # cross-validate, leaving one SF out at a time
    chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
    un,whichCV = np.unique(np.concatenate((typelist[inds],sflist[inds]), axis=1), return_inverse=True, axis=0)

    for cv in range(np.size(np.unique(whichCV))):
        trninds = np.where(whichCV!=cv)[0]
        tstinds = np.where(whichCV==cv)[0]
   
        chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
        
        average_recons = IEM.shift_and_average(chan_resp_all[tstinds,:],ori_labs[tstinds],center_deg);

        h, =plt.plot(xx,average_recons)
        if ww1==np.max(layers2plot):
            lh.append(h)
            legendlabs.append('SF=%.2f' % sf_vals[cv])
            if cv==nSF-1:
                ax.legend(lh,legendlabs)
                        
    plt.ylim(ylims)
    plt.plot([center_deg,center_deg], ylims)
    plt.title('%s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
    if ww1==np.max(layers2plot)-2:
        plt.xlabel('Orientation Channel (deg)')
        plt.ylabel('Channel Activation Weight')
    else:
        plt.tick_params(axis='x', bottom=False,labelbottom = False)
        
    if (np.size(timepts2plot)>1 and ww1==np.max(timepts2plot)):
        
        plt.tick_params(axis='y', left=False,labelleft = False)
        plt.ylabel(None)
        
    
    plt.suptitle('Train across SF\n%s, noise=%.2f' % (stim_types[tt], noise_levels[nn]))
 
#%% train within SF, plot overlaid

plt.close('all')

layers2plot = np.arange(0,nLayers,1)
#layers2plot = np.arange(0,1)
#layers2plot = [0]
timepts2plot = np.arange(0,1)

tt=1
ww2=0
nn=4

ylims = [-1.5,1.5]

nVox2Use = 100

plt.figure()
ii=0
legendlabs = [];
lh =[];

for ww1 in layers2plot:
       
    ii=ii+1
    ax=plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
    plt.title('%s' % layer_labels[ww1])
    
    ori_labs = actual_labels
    center_deg=90
    n_chans=9
#        n_folds = 20
    for bb in range(nSF):
            
        inds = np.where(np.logical_and(typelist==tt, sflist==bb))[0]
        
        ori_labs = actual_labels[inds]
    
        # run the IEM across all stims as the training set
        alldat = allw[ww1][0][ww2][inds,0:nVox2Use]
        
        # cross-validate, leaving one phase out at a time
        # this amounts to a random (but not random) 25% of the data
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(phaselist[inds], return_inverse=True, axis=0)
    
        for cv in range(np.size(np.unique(whichCV))):
            
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
       
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs[trninds],alldat[tstinds,:],n_chans)
            
        average_recons = IEM.shift_and_average(chan_resp_all,ori_labs,center_deg);
    
        h, = plt.plot(xx,average_recons)
        if ww1==np.max(layers2plot):
            lh.append(h)
            legendlabs.append('SF=%.2f' % sf_vals[bb])
            if bb==nSF-1:
                ax.legend(lh,legendlabs)
                            
    plt.ylim(ylims)
    plt.plot([center_deg,center_deg], ylims)
    plt.title('%s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
    if ww1==np.max(layers2plot)-2:
        plt.xlabel('Orientation Channel (deg)')
        plt.ylabel('Channel Activation Weight')
    else:
        plt.tick_params(axis='x', bottom=False,labelbottom = False)
        
    if (np.size(timepts2plot)>1 and ww1==np.max(timepts2plot)):
        
        plt.tick_params(axis='y', left=False,labelleft = False)
        plt.ylabel(None)
        
    
    plt.suptitle('Train within SF\n%s, noise=%.2f' % (stim_types[tt], noise_levels[nn]))
            
#%% separate the recons according to cardinal-ness

plt.close('all')

layers2plot = np.arange(0,nLayers,1)

tt=0
ww2=0
nn=0

ylims = [-1.5,1.5]

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

plt.figure()
ii=0
legendlabs = [];
lh =[];


for ww1 in layers2plot:
    
    ii=ii+1
    ax=plt.subplot(np.ceil(len(layers2plot)/4),4,ii)
    plt.title('%s' % layer_labels[ww1])
    
    inds = np.where(typelist==tt)[0]
    
    alldat = allw[ww1][nn][ww2][inds,:]
    ori_labs_here = ori_labs[inds]
    dir_to_oblique_here = dir_to_oblique[inds]
    bin_labs_here = bin_labs[inds]
    
    # cross-validate, leaving one SF out at a time
    chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
    un,whichCV = np.unique(np.concatenate((typelist[inds],sflist[inds]), axis=1), return_inverse=True, axis=0)
    for cv in range(np.size(np.unique(whichCV))):
        trninds = np.where(whichCV!=cv)[0]
        tstinds = np.where(whichCV==cv)[0]

        chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs_here[trninds],alldat[tstinds,:],n_chans)
     
    for bb in range(nBins):

        inds = np.where(bin_labs_here==bb)[0]
    
        average_recons = IEM.shift_flip_and_average(chan_resp_all[inds,:],ori_labs_here[inds,:],dir_to_oblique_here[inds,:],center_deg);

        h, = plt.plot(xx,average_recons)
        if ww1==np.max(layers2plot):
            lh.append(h)
            
            if bb==nBins-1:
                ax.legend(lh,bin_labels)
                
    plt.ylim(ylims)
    plt.plot([center_deg,center_deg], ylims,'k')
    plt.title('%s - %s' % (layer_labels[ww1], timepoint_labels[ww2]))
    if ww1==np.max(layers2plot)-2:
        plt.xlabel('Orientation Channel (deg)')
        plt.ylabel('Channel Activation Weight')
    else:
        plt.tick_params(axis='x', bottom=False,labelbottom = False)
        
    if (np.size(timepts2plot)>1 and ww1==np.max(timepts2plot)):
        
        plt.tick_params(axis='y', left=False,labelleft = False)
        plt.ylabel(None)
    
plt.suptitle('Flipped so CW is away from cardinal, trn/test across SF\n%s, noise=%.2f' %  (stim_types[tt], noise_levels[nn]))

#%% Plot bias curves from each area, trn/test all stims
        
plt.close('all')

layers2plot = np.arange(0,nLayers,1)

ww2=0
nn=0
tt=1
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
    
    inds = np.where(typelist==tt)[0]
    
    alldat = allw[ww1][nn][ww2][inds,:]
    ori_labs_here = ori_labs[inds]
    
    # cross-validate, leaving one SF out at a time
    chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
    un,whichCV = np.unique(np.concatenate((typelist[inds],sflist[inds]), axis=1), return_inverse=True, axis=0)
    for cv in range(np.size(np.unique(whichCV))):
        trninds = np.where(whichCV!=cv)[0]
        tstinds = np.where(whichCV==cv)[0]

        chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],ori_labs_here[trninds],alldat[tstinds,:],n_chans)

    # now find the peak of each reconstruction and get a distribution of these predictions at each point in ori space.
    pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
       
    un = np.unique(ori_labs)
    avg_pred = np.zeros(np.shape(un))
    std_pred = np.zeros(np.shape(un))
   
    for uu in range(len(un)):
        avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(ori_labs_here==un[uu])[0]], high=180,low=0)
        std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(ori_labs_here==un[uu])[0]], high=180,low=0)

    # Now calculate bias.
    # first correct values that wrapped around. for instance, the mean at 0 might be 178.5 and we would want 1.5.
    avg_pred_corr = avg_pred
   
    indslow = np.where(np.logical_and(un<45, avg_pred>135))
    indshigh = np.where(np.logical_and(un>135, avg_pred<45))
    print('layer %d: correcting %d + %d values for wraparound' % (ww1,np.size(indslow),np.size(indshigh)))
    avg_pred_corr[indslow] = avg_pred_corr[indslow] -180
    avg_pred_corr[indshigh] = avg_pred_corr[indshigh] +180
    
    avg_bias = avg_pred_corr - un
    
#    plt.plot(un,avg_bias)
#    plt.figure()
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