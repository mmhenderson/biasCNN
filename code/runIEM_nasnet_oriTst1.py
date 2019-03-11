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
%run load_data_nasnet_oriTst1

nVox2Use = 100
center_deg=90
n_chans=9

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

#%% run the IEM - within each stimulus type and sf separately

plt.close('all')

layers2plot = [2]
timepts2plot = [0]
noiselevels2plot =[0]

xx=np.arange(0,180,1)

ylims = [-1,1]

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

#%% train the IEM across SF within noise level, plot one noise level at a time
        
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

        ori_labs = actual_labels
       
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
        
    plt.suptitle('Average reconstructions, leave one spatial freq out\nnoise==%.2f' % noise_levels[nn])
    
#%% train the IEM across SF within noise level, overlay noise levels
        
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

        ori_labs = actual_labels
       
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
        
plt.suptitle('Average reconstructions, leave one spatial freq out')

#%% train the IEM across SF, across noise levels
        
plt.close('all')

layers2plot = np.arange(0,19,1)
timepts2plot = [0]

nn1=2
tstNoise = [0,1,2]

ylims = [-1,1]

plt.figure()
legendlabs = []
lh = []

for nn2 in tstNoise:
    ii=0;
    legendlabs.append('test noise=%.2f' % noise_levels[nn2])
    for ww1 in layers2plot:
       
        ii=ii+1
    
        ori_labs = actual_labels
       
        # first separate out the training/testing noise levels
        trninds1 = np.where(noiselist==nn1)[0]
    
        alldat_trn = allw[ww1][0][ww2][trninds1,0:nVox2Use]
        typelist_trn = typelist[trninds1]
        sflist_trn = sflist[trninds1]
        orilabs_trn = ori_labs[trninds1]
        
        tstinds1 = np.where(noiselist==nn2)[0]
    
        alldat_tst = allw[ww1][0][ww2][tstinds1,0:nVox2Use]
        typelist_tst = typelist[tstinds1]
        sflist_tst = sflist[tstinds1]
        orilabs_tst = ori_labs[tstinds1]
        
        # the other labels on these partitions of the data should be identical
        assert np.array_equal(typelist_trn, typelist_tst) & np.array_equal(sflist_trn, sflist_tst) & np.array_equal(orilabs_trn, orilabs_tst)
    
        chan_resp_all = np.zeros([np.shape(alldat_tst)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist_trn,sflist_trn), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds2 = np.where(whichCV!=cv)[0]
            tstinds2 = np.where(whichCV==cv)[0]
    #            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds2,:] = IEM.get_recons(alldat_trn[trninds2,:],orilabs_trn[trninds2],alldat_tst[tstinds2,:],n_chans)
            
        average_recons = IEM.shift_and_average(chan_resp_all,orilabs_tst,center_deg);
          
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
            
plt.suptitle('Average reconstructions, leave one spatial freq out\ntrain noise=%.2f' %(noise_levels[nn1]))

#%% train all stimuli, separate the test set according to type/sf

plt.close('all')


layers2plot = np.arange(0,19,1)
timepts2plot = [0]

nn1 = 0
tt=0;
ww2=0;
ylims = [-1,2]

plt.figure()
legendlabs = []
for sf in range(nSF):
    legendlabs.append('sf=%.2f' %sf_vals[sf])
lh = []
ii=0;

for ww1 in layers2plot:
    
    ii=ii+1
      
    ori_labs = actual_labels
   
    inds1 =  np.where(noiselist==nn1)[0] 
    # run the IEM across all stims as the training set
    alldat = allw[ww1][0][ww2][inds1,0:nVox2Use]
    
    typelist_thisnoise = typelist[inds1]
    sflist_thisnoise = sflist[inds1]
    orilabs_thisnoise = ori_labs[inds1]
    
    # cross-validate, leaving one stimulus type and SF out at a time
    chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
    un,whichCV = np.unique(np.concatenate((typelist_thisnoise,sflist_thisnoise), axis=1), return_inverse=True, axis=0)

    for cv in range(np.size(np.unique(whichCV))):
        trninds = np.where(whichCV!=cv)[0]
        tstinds = np.where(whichCV==cv)[0]
   
        chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],orilabs_thisnoise[trninds],alldat[tstinds,:],n_chans)
     
    for sf in range(nSF):
        
        # average recons within just this spatial frequency
        inds = np.where(np.logical_and(typelist_thisnoise==tt, sflist_thisnoise==sf))[0]            
        average_recons = IEM.shift_and_average(chan_resp_all[inds,:],orilabs_thisnoise[inds,:],center_deg);
   
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
    
plt.suptitle('Average reconstructions, leave one spatial freq out\ntrain/test noise=%.2f' %(nn1))

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
   
  
    myinds = np.where(noiselist==nn)[0]

    alldat = allw[ww1][0][ww2][myinds,0:nVox2Use]
    typelist_now = typelist[myinds]
    sflist_now = sflist[myinds]    
    orilabs_now = ori_labs[myinds]

    # cross-validate, leaving one stimulus type and SF out at a time
    chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
    un,whichCV = np.unique(np.concatenate((typelist_now,sflist_now), axis=1), return_inverse=True, axis=0)

    for cv in range(np.size(np.unique(whichCV))):
        trninds = np.where(whichCV!=cv)[0]
        tstinds = np.where(whichCV==cv)[0]
   
        chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],orilabs_now[trninds],alldat[tstinds,:],n_chans)
     
    pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
    un = np.unique(orilabs_now)
    avg_pred = np.zeros(np.shape(un))
    std_pred = np.zeros(np.shape(un))
   
    for uu in range(len(un)):
        avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(orilabs_now==un[uu])[0]], high=180,low=0)
        std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(orilabs_now==un[uu])[0]], high=180,low=0)

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
         
            myinds = np.where(noiselist==nn)[0]
        
            alldat = allw[ww1][0][ww2][myinds,0:nVox2Use]
            typelist_now = typelist[myinds]
            sflist_now = sflist[myinds]    
            orilabs_now = ori_labs[myinds]
        
            # cross-validate, leaving one stimulus type and SF out at a time
            chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
            un,whichCV = np.unique(np.concatenate((typelist_now,sflist_now), axis=1), return_inverse=True, axis=0)
        
            for cv in range(np.size(np.unique(whichCV))):
                trninds = np.where(whichCV!=cv)[0]
                tstinds = np.where(whichCV==cv)[0]
           
                chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],orilabs_now[trninds],alldat[tstinds,:],n_chans)
                
            
            pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
            un = np.unique(orilabs_now)
            avg_pred = np.zeros(np.shape(un))
            std_pred = np.zeros(np.shape(un))
           
            for uu in range(len(un)):
                avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(orilabs_now==un[uu])[0]], high=179,low=0)
                std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(orilabs_now==un[uu])[0]], high=179,low=0)
               

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

        
#%% trn/test within noise level, overlay noise levels (plotting bias)
        
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
    
        alldat = allw[ww1][0][ww2][myinds,0:nVox2Use]
        typelist_now = typelist[myinds]
        sflist_now = sflist[myinds]    
        orilabs_now = ori_labs[myinds]
    
        # cross-validate, leaving one stimulus type and SF out at a time
        chan_resp_all = np.zeros([np.shape(alldat)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist_now,sflist_now), axis=1), return_inverse=True, axis=0)
    
        for cv in range(np.size(np.unique(whichCV))):
            trninds = np.where(whichCV!=cv)[0]
            tstinds = np.where(whichCV==cv)[0]
       
            chan_resp_all[tstinds,:] = IEM.get_recons(alldat[trninds,:],orilabs_now[trninds],alldat[tstinds,:],n_chans)
         
        pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
        un = np.unique(orilabs_now)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(orilabs_now==un[uu])[0]], high=180,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(orilabs_now==un[uu])[0]], high=180,low=0)
    
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
            if nn2==np.max(tstNoise):
                ax.legend(lh,legendlabs)
 
        plt.xlim([0,180])
        plt.ylim(ylims)
        plt.plot([0,180],[0,0],'k-')
        plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
        plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
        plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
        
plt.suptitle('Reconstruction bias, train/test across SF\nwithin noise levels')

#%% trn/test across noise levels, plot bias
    
plt.close('all')

layers2plot = range(nLayers)
ww2 = 0

nn1=0
tstNoise = [0,1,2]

ylims=[-10,10]


legendlabs = []
for nn in range(nNoiseLevels):
    legendlabs.append(' test noise=%.2f' %noise_levels[nn])
lh = []
ii=0;


for ww1 in layers2plot:
   
    ii=ii+1
    ax = plt.subplot(np.ceil(len(layers2plot)/4),np.min([len(layers2plot),4]),ii)
   
    for nn2 in tstNoise:
 
        # first separate out the training/testing noise levels
        trninds1 = np.where(noiselist==nn1)[0]
    
        alldat_trn = allw[ww1][0][ww2][trninds1,0:nVox2Use]
        typelist_trn = typelist[trninds1]
        sflist_trn = sflist[trninds1]
        orilabs_trn = ori_labs[trninds1]
        
        tstinds1 = np.where(noiselist==nn2)[0]
    
        alldat_tst = allw[ww1][0][ww2][tstinds1,0:nVox2Use]
        typelist_tst = typelist[tstinds1]
        sflist_tst = sflist[tstinds1]
        orilabs_tst = ori_labs[tstinds1]
        
        # the other labels on these partitions of the data should be identical
        assert np.array_equal(typelist_trn, typelist_tst) & np.array_equal(sflist_trn, sflist_tst) & np.array_equal(orilabs_trn, orilabs_tst)
    
        chan_resp_all = np.zeros([np.shape(alldat_tst)[0], 180])
        un,whichCV = np.unique(np.concatenate((typelist_trn,sflist_trn), axis=1), return_inverse=True, axis=0)
        for cv in range(np.size(np.unique(whichCV))):
            trninds2 = np.where(whichCV!=cv)[0]
            tstinds2 = np.where(whichCV==cv)[0]
    #            print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        
            chan_resp_all[tstinds2,:] = IEM.get_recons(alldat_trn[trninds2,:],orilabs_trn[trninds2],alldat_tst[tstinds2,:],n_chans)
            
#        average_recons = IEM.shift_and_average(chan_resp_all,orilabs_tst,center_deg);
          

        pred_labels = xx[np.argmax(chan_resp_all, axis=1)]
        un = np.unique(ori_labs)
        avg_pred = np.zeros(np.shape(un))
        std_pred = np.zeros(np.shape(un))
       
        for uu in range(len(un)):
            avg_pred[uu] = scipy.stats.circmean(pred_labels[np.where(orilabs_tst==un[uu])[0]], high=179,low=0)
            std_pred[uu] = scipy.stats.circstd(pred_labels[np.where(orilabs_tst==un[uu])[0]], high=179,low=0)

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
            if nn2==np.max(tstNoise):
                ax.legend(lh,legendlabs)
 
        plt.xlim([0,180])
        plt.ylim(ylims)
        plt.plot([0,180],[0,0],'k-')
        plt.plot([center_deg, center_deg],plt.get(plt.gca(),'ylim'),'k-')
        plt.plot([45,45],plt.get(plt.gca(),'ylim'),'k--')
        plt.plot([135,135],plt.get(plt.gca(),'ylim'),'k--')
        
plt.suptitle('Reconstruction bias, train/test across SF\nTRAINING noise = %.2f' %noise_levels[nn1])
