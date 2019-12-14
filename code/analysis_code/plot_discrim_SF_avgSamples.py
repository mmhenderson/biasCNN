#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt

import os

import numpy as np

from copy import deepcopy

from scipy import stats 

from matplotlib import cm

cols_sf = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = cols_sf[np.arange(2,8,1),:,:]

#%% get the data ready to go...then can run any below cells independently.
root = '/usr/local/serenceslab/maggie/biasCNN/';

#dataset = 'SpatFreqGratings'
model='vgg16'
#training_str='scratch_imagenet_rot_45'
training_str='pretrained'
param_str='params1'

# first, searching for all folders from the same model, evaluated on different datasets (the sets are very similar but have different noise instantiation)
dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str))
good = [ii for ii in range(np.size(dirs)) if 'SpatFreqGratings' in dirs[ii]]

model_name_2plot = model + '_' + training_str + '_' + param_str + '_avg_samples'


os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures')

import load_activations

import classifiers_custom as classifiers    

allw = []
for ii in good:
  
  # also searching for all evaluations at different timepts (nearby, all are around the optimal point)
  dirs2 = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dirs[ii]))
  good2 = [jj for jj in range(np.size(dirs2)) if 'reduced' in dirs2[jj]]

  for jj in good2:
    
    
    ckpt_num= dirs2[jj].split('_')[2][5:]
    this_allw, all_labs, info = load_activations.load_activ(model, dirs[ii], training_str, param_str, ckpt_num)
    allw.append(this_allw)
    if ii==good[0] & jj==good2[0]:
      info_orig = info
    else:
      np.testing.assert_equal(info_orig, info)
    # extract some fields that will help us process the data
    orilist = info['orilist']
    phaselist=  info['phaselist']
    sflist = info['sflist']
    typelist = info['typelist']
    noiselist = info['noiselist']
    exlist = info['exlist']
    contrastlist = info['contrastlist']
    
    nLayers = info['nLayers']
    nPhase = info['nPhase']
    nSF = info['nSF']
    nType = info['nType']
    nTimePts = info['nTimePts']
    nNoiseLevels = info['nNoiseLevels']
    nEx = info['nEx']
    nContrastLevels = info['nContrastLevels']
    
    layer_labels = info['layer_labels']
    timepoint_labels = info['timepoint_labels']
    noise_levels = info['noise_levels']    
    stim_types = info['stim_types']
    phase_vals = info['phase_vals']
    contrast_levels = info['contrast_levels']      
    sf_vals = info['sf_vals']
    
    assert nLayers == info['nLayers']
    assert nPhase == info['nPhase']
    assert nSF == info['nSF']
    assert nType == info['nType']
    assert nTimePts == info['nTimePts']
    assert nNoiseLevels == info['nNoiseLevels']
    assert nEx == info['nEx']
    assert nContrastLevels == info['nContrastLevels']

# how many total evaluations of the network do we have here?  
nSamples = np.shape(allw)[0]
#%% plot discriminability across 0-360 space, overlay spatial frequencies

assert phase_vals==[0,180]
  
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')
plt.figure()
xx=1

layers2plot = np.arange(0,nLayers,1)
sf2plot = np.arange(0,6,1)
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

cc=0
nn=0
 
for ww1 in layers2plot:

    for sf in range(np.size(sf2plot)):
      
        all_disc = []
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):

          w = allw[kk][ww1][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
         
          all_disc.append(disc)
          
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        # average over samples to get what we will plot
        disc = np.mean(all_disc,0)
        
        plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2],order='F'), axis=1),color=cols_sf[sf,0,:])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\nContrast %.2f Noise %.2f' % (model_name_2plot,contrast_levels[cc],noise_levels[nn]))
             
    xx=xx+1
#%% plot discriminability across 0-360 space, overlay samples (one spatial freq only)

assert phase_vals==[0,180]
  
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')
plt.figure()
xx=1


layers2plot = np.arange(0,nLayers,1)

#legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

sf=2
cc=0
nn=0
 
for ww1 in layers2plot:

    all_disc = []
    # looping here over "samples", going to get discriminability function for each, then average to smooth it.
    for kk in range(nSamples):

      w = allw[kk][ww1][0]

      inds = np.where(np.all([sflist==sf,contrastlist==cc,noiselist==nn],axis=0))[0]

      ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
     
      all_disc.append(disc)
        
      plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)

      # plot individual sample
      plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2],order='F'), axis=1),color=cols_sf[sf,0,:])

#      plt.plot(ori_axis[0:180],np.mean(np.reshape(disc,[180,2],order'F'), axis=1),color=cols_sf[sf,0,:])
    
    # average over samples to get the mean, plot mean too
    disc = np.mean(all_disc,0)
    
    plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2],order='F'), axis=1),color='k')

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
#        plt.legend(legendlabs)
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\nContrast %.2f Noise %.2f SF %.2f cpp' % (model_name_2plot,contrast_levels[cc],noise_levels[nn],sf_vals[sf]))
             
    xx=xx+1
  
#%% plot HORIZONTAL anisotropy in 0-360 space, overlay SF
assert phase_vals==[0,180]

cols_sf = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = cols_sf[np.arange(2,8,1),:,:]
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

#plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

b = np.arange(22.5,360,45)
a = np.arange(90,360,180)
bin_size = 6
      
for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        all_disc = []
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):

          w = allw[kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
         
          all_disc.append(disc)
        
        # average over samples to get a smoother function
        disc = np.mean(all_disc,0)
        
        # now evaluate the biases
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0] 
            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(360+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,0,:])

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nHORIZONTAL anisotropy' % (model_name_2plot))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_HorizontalAnisotropySF.pdf' % (model_name_2plot))
plt.savefig(figname, format='pdf',transparent=True)
     
#%% plot VERTICAL anisotropy in 0-360 space, overlay SF
assert phase_vals==[0,180]

cols_sf = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = cols_sf[np.arange(2,8,1),:,:]
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

b = np.arange(22.5,360,45)
a = np.arange(0,360,180)
bin_size = 6
      
for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        all_disc = []
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):

          w = allw[kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
         
          all_disc.append(disc)
        
        # average over samples to get a smoother function
        disc = np.mean(all_disc,0)
        
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0] 
            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(360+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,0,:])

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nVERTICAL anisotropy' % (model_name_2plot))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_VerticalAnisotropySF.pdf' % (model_name_2plot))
plt.savefig(figname, format='pdf',transparent=True)

    
#%% plot Cardinal (V+H) anisotropy in 0-360 space, overlay SF
assert phase_vals==[0,180]

cols_sf = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = cols_sf[np.arange(2,8,1),:,:]
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

#plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

b = np.arange(22.5,360,45)
a = np.arange(0,360,90)
bin_size = 6
      
for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        all_disc = []
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):

          w = allw[kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
         
          all_disc.append(disc)
        
        # average over samples to get a smoother function
        disc = np.mean(all_disc,0)
        
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0] 
            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(360+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,0,:])

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (model_name_2plot))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_AnisotropySF.pdf' % (model_name_2plot))
plt.savefig(figname, format='pdf',transparent=True)
#%% plot OBLIQUE (45 only) anisotropy in 0-360 space, overlay SF
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

b = np.arange(22.5,360,45)
a = np.arange(45,360,180)
bin_size = 6
      
for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        all_disc = []
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):

          w = allw[kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
         
          all_disc.append(disc)
        
        # average over samples to get a smoother function
        disc = np.mean(all_disc,0)
        
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0] 
            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(360+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,0,:])

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nOblique (45 only) anisotropy' % (model_name_2plot))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_ObliqueAnisotropySF.pdf' % (model_name_2plot))
plt.savefig(figname, format='pdf',transparent=True)
 #%% plot OBLIQUE (135 only) anisotropy in 0-360 space, overlay SF
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

b = np.arange(22.5,360,45)
a = np.arange(135,360,180)
bin_size = 6
      
for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        all_disc = []
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):

          w = allw[kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
         
          all_disc.append(disc)
        
        # average over samples to get a smoother function
        disc = np.mean(all_disc,0)
        
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0] 
            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(360+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,0,:])

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nOblique (135 only) anisotropy' % (model_name_2plot))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_ObliqueAnisotropySF.pdf' % (model_name_2plot))
plt.savefig(figname, format='pdf',transparent=True)

#%% plot OBLIQUE (45 + 135) anisotropy in 0-360 space, overlay SF
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

#plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

b = np.arange(22.5,360,45)
a = np.arange(45,360,90)
bin_size = 6
      
for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    aniso_vals = np.zeros([4,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        all_disc = []
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):

          w = allw[kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
         
          all_disc.append(disc)
        
        # average over samples to get a smoother function
        disc = np.mean(all_disc,0)
        
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0] 
            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(360+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,0,:])

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nOblique anisotropy' % (model_name_2plot))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_ObliqueAnisotropySF.pdf' % (model_name_2plot))
plt.savefig(figname, format='pdf',transparent=True)
     
#%% plot Mean Discrim in 0-360 space, overlay SF
assert phase_vals==[0,180]

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

#plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in range(np.size(layers2plot)):
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
     
        vals[ww1] = np.mean(disc)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,0,:])

#ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+1]

#plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.title('%s\nMean Discriminability' % (model_name_2plot))
plt.ylabel('d''')
#plt.xlabel('Layer number')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

figname = os.path.join(figfolder, 'SpatFreq','%s_MeanDiscSF.pdf' % (model_name_2plot))
plt.savefig(figname, format='pdf',transparent=True)
 
