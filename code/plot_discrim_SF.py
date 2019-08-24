#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt

import classifiers    

import numpy as np

from copy import deepcopy

from matplotlib import cm

cols1 = np.reshape(cm.tab20b(np.arange(0,20,1)), [5, 4, 4])
cols2 = np.reshape(cm.tab20c(np.arange(0,20,1)), [5, 4, 4])
cols_all = np.concatenate((cols1,cols2),axis=0)
cols_all = cols_all[[4, 6, 2, 7, 5, 0],:,:]
#%% get the data ready to go...then can run any below cells independently.

model_str = 'vgg16_oriTst12'
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
exlist = info['exlist']
contrastlist = info['contrastlist']

layer_labels = info['layer_labels']
sf_vals = info['sf_vals']
noise_levels = info['noise_levels']
timepoint_labels = info['timepoint_labels']
stim_types = info['stim_types']
phase_vals = info['phase_vals']
contrast_levels = info['contrast_levels']

nLayers = info['nLayers']
nPhase = info['nPhase']
nSF = info['nSF']
nType = info['nType']
nTimePts = info['nTimePts']
nNoiseLevels = info['nNoiseLevels']
nEx = info['nEx']
nContrastLevels = info['nContrastLevels']

#%% plot discriminability across 0-360 space, overlay spatial frequencies
    
assert 'oriTst11' in model_str
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')
plt.figure()
xx=1

layers2plot = np.arange(0,nLayers,1)
sf2plot = np.arange(0,6,1)
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

for ww1 in layers2plot:

    w = allw[ww1][0]
    
    for sf in range(np.size(sf2plot)):
        
        inds = np.where(sflist==sf2plot[sf])[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
       
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2]), axis=1))

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
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations' % (model_str))
             
    xx=xx+1
  
#%%  Plot the standardized euclidean distance, in 0-360 space
# Overlay SF
# plot just one layer, save a figure
assert 'oriTst11' in model_str
 
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')

from matplotlib import cm


layers2plot = [5]
sf2plot = np.arange(0,6,1)
#sf2plot = [3]
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

colors = cm.tab20b(np.linspace(0,1,np.size(sf2plot)))
#colors = [cm.Greens(0.5),cm.Purples(0.5),cm.Reds(0.5),cm.Blues(0.5),cm.PuRd(0.5),cm.Oranges(0.5)]

for ww1 in layers2plot:
    plt.figure()
    
    w = allw[ww1][0]
    
    for sf in range(np.size(sf2plot)):
        
        inds = np.where(sflist==sf2plot[sf])[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
       
        plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2]), axis=1),color = cols_all[sf,1,:])

    plt.title('Layer %d of %d\n%s' % (ww1+1,nLayers,layer_labels[ww1]))
    
    if ww1==layers2plot[-1]:
        plt.xlabel('Grating Orientation')
        plt.ylabel('Discriminability')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,181,45))
        plt.yticks(np.linspace(round(plt.gca().get_ylim()[0],-2),round(plt.gca().get_ylim()[1],-2), 3))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color=[0.8,0.8,0.8,1])
   
    plt.xlim([0,180])
    
#    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations' % (model_str))
    
    figname = os.path.join(figfolder, 'SpatFreq','%s_%s_discrim.pdf' % (model_str,layer_labels[ww1]))
    plt.savefig(figname, format='pdf',transparent=True)
     

#%% plot Cardinal anisotropy in 0-360 space, overlay SF
assert 'oriTst11' in model_str
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
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
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
       
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0] 
#            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(360+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_all[sf,1,:])

ylims = [-.2,1]
xlims = [-1, np.size(layers2plot)+1]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_AnisotropySF.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)
     
#%% plot Mean Discrim in 0-360 space, overlay SF
assert 'oriTst11' in model_str
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
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
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_all[sf,1,:])

#ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+1]

#plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.title('%s\nMean Discriminability' % (model_str))
plt.ylabel('d''')
#plt.xlabel('Layer number')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

figname = os.path.join(figfolder, 'SpatFreq','%s_MeanDiscSF.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)
 
#%%  Plot the standardized euclidean distance, in 0-180 space
# Overlay spatial frequencies 
plt.close('all')
plt.figure()
xx=1

layers2plot = np.arange(0,nLayers,1)
sf2plot = np.arange(0,3,1)
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

for ww1 in layers2plot:

    w = allw[ww1][0]
    
    for sf in range(np.size(sf2plot)):
        
        inds = np.where(sflist==sf2plot[sf])[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist[inds])
       
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(ori_axis,disc)

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,360,45))
    else:
        plt.xticks([])
   
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations' % (model_str))
             
    xx=xx+1



#%% plot Cardinal anisotropy, overlay SF
      
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]
b = np.arange(22.5,180,45)
a = np.arange(0,180,90)
bin_size = 4
     
for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    aniso_vals = np.zeros([2,np.size(layers2plot)])
    
    for ww1 in range(np.size(layers2plot)):
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist[inds1])
       
        # take the bins of interest to get amplitude       
        baseline_discrim = [];
        for ii in range(np.size(b)):        
            inds = np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(180+b[ii]))<bin_size/2))[0] 
#            assert np.size(inds)==bin_size-1
            baseline_discrim.append(disc[inds])
            
        for ii in range(np.size(a)):       
            inds = np.where(np.logical_or(np.abs(ori_axis-a[ii])<bin_size/2, np.abs(ori_axis-(180+a[ii]))<bin_size/2))[0]
            assert np.size(inds)==bin_size
            aniso_vals[ii,ww1] = (np.mean(disc[inds]) - np.mean(baseline_discrim))/(np.mean(disc[inds]) + np.mean(baseline_discrim))
      
    vals = np.mean(aniso_vals,0)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(legendlabs)
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

#%% plot Mean discrim overlay SF
      
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

 
for sf in sf2plot:
    
    inds1 = np.where(sflist==sf2plot[sf])[0]
        
    vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in range(np.size(layers2plot)):
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist[inds1])
           
        vals[ww1] = np.mean(disc)
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals)

ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+2]

plt.legend(legendlabs)
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nMean Orientation Discriminability' % (model_str))
plt.ylabel('Normalized Euclidean Distance')
plt.xlabel('Layer number')


#%% Plot a dissimilarity matrix based on the standardized euclidean distance, within one SF
 
plt.close('all')
nn=0
ww2 = 0;
ww1 = 15;
tt = 0
sf2plot = [0,1,2]
    
for sf in sf2plot:
    
    w = allw[ww1][ww2]
        
    myinds_bool = np.all([typelist==tt, noiselist==nn, sflist==sf],axis=0)
    un,ia = np.unique(orilist, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==orilist)
    
    disc = np.zeros([180,180])
    
    for ii in np.arange(0,np.size(un)):
        
        # find all gratings with this label
        inds1 = np.where(np.logical_and(orilist==un[ii], myinds_bool))[0]    
           
        for jj in np.arange(ii+1, np.size(un)):
            
            # now all gratings with other label
            inds2 = np.where(np.logical_and(orilist==un[jj], myinds_bool))[0]    
        
            dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])
    
            disc[ii,jj]=  dist
            disc[jj,ii] = dist
      
    plt.figure()
    plt.pcolormesh(disc)
    plt.colorbar()
             
    plt.title('Standardized Euclidean distance, sf=%.2f, noise=%.2f\n%s' % (sf_vals[sf],noise_levels[nn],layer_labels[ww1]))
        
    plt.xlabel('orientation 1')
    plt.ylabel('orientation 2')
    
    plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
    plt.yticks(np.arange(0,180,45),np.arange(0,180,45))
            
#%% Plot a dissimilarity matrix based on the standardized euclidean distance, comparing different SFs
 
plt.close('all')
nn=0
ww2 = 0;
ww1 = 15
tt = 0

w = allw[ww1][ww2]
    
myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
un,ia = np.unique(orilist, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==orilist)

disc = np.zeros([180*nSF,180*nSF])

sf_tiled = np.repeat(np.arange(0,nSF,1),180)
ori_tiled = np.tile(np.arange(0,180,1),[1,nSF])

for ii1 in range(180*nSF):
    
     # find all gratings with this label
     inds1 = np.where(np.all([sflist==sf_tiled[ii1], orilist==ori_tiled[0,ii1], myinds_bool], axis=0))[0]    
             
     for ii2 in np.arange(ii1+1, 180*nSF):
        
        # now all gratings with other label
        inds2 = np.where(np.all([sflist==sf_tiled[ii2], orilist==ori_tiled[0,ii2], myinds_bool], axis=0))[0]    
  
        dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])

        disc[ii1,ii2]=  dist
     
        disc[ii2,ii1] = dist
 
plt.figure()
plt.pcolormesh(disc)
plt.colorbar()
         
plt.title('Standardized Euclidean distance, noise=%.2f\n%s' % (noise_levels[nn],layer_labels[ww1]))
    
plt.xlabel('orientation 1')
plt.ylabel('orientation 2')

plt.xticks(np.arange(0,180*nSF,45),np.tile(np.arange(0,180,45), [1,nSF])[0])
plt.yticks(np.arange(0,180*nSF,45),np.tile(np.arange(0,180,45), [1,nSF])[0])
        
for sf in range(nSF):
    plt.plot([sf*180,sf*180], [0,nSF*180],'k')
    plt.plot([0,nSF*180],[sf*180,sf*180],'k')

#%% Plot a dissimilarity matrix based on the standardized euclidean distance, across all SF
 
plt.close('all')
nn=0
ww2 = 0;
ww1 = 14
tt = 0

w = allw[ww1][ww2]
    
myinds_bool = np.all([typelist==tt, noiselist==nn],axis=0)
un,ia = np.unique(orilist, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==orilist)

disc = np.zeros([180,180])

for ii in np.arange(0,np.size(un)):
    
    # find all gratings with this label
    inds1 = np.where(np.logical_and(orilist==un[ii], myinds_bool))[0]    
       
    for jj in np.arange(ii+1, np.size(un)):
        
        # now all gratings with other label
        inds2 = np.where(np.logical_and(orilist==un[jj], myinds_bool))[0]    
    
        dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])

        disc[ii,jj]=  dist
        disc[jj,ii] = dist
  
plt.figure()
plt.pcolormesh(disc)
plt.colorbar()
         
plt.title('Standardized Euclidean distance, across all SF, noise=%.2f\n%s' % (noise_levels[nn],layer_labels[ww1]))
    
plt.xlabel('orientation 1')
plt.ylabel('orientation 2')

plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
plt.yticks(np.arange(0,180,45),np.arange(0,180,45))
        