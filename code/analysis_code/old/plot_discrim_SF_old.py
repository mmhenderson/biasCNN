#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt


import numpy as np

from copy import deepcopy

from scipy import stats 

from matplotlib import cm

cols_sf = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = cols_sf[np.arange(2,8,1),:,:]

#%% get the data ready to go...then can run any below cells independently.

model_str = ['vgg16_oriTst11']

model_name_2plot = 'VGG-16-PRETRAINED'

root = '/usr/local/serenceslab/maggie/biasCNN/';

import os
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures')


import load_activations_old

import classifiers_custom as classifiers    


for mm in range(np.size(model_str)):
    
    this_allw, this_all_labs, this_info = load_activations_old.load_activ(model_str[mm])

    if mm==0:
        
        allw = this_allw
        
        # extract some fields that will help us process the data
        orilist = this_info['orilist']
        phaselist=  this_info['phaselist']
        sflist = this_info['sflist']
        typelist = this_info['typelist']
        noiselist = this_info['noiselist']
        exlist = this_info['exlist']
        contrastlist = this_info['contrastlist']
        
        nLayers = this_info['nLayers']
        nPhase = this_info['nPhase']
        nSF = this_info['nSF']
        nType = this_info['nType']
        nTimePts = this_info['nTimePts']
        nNoiseLevels = this_info['nNoiseLevels']
        nEx = this_info['nEx']
        nContrastLevels = this_info['nContrastLevels']
        
        layer_labels = this_info['layer_labels']
        timepoint_labels = this_info['timepoint_labels']
        noise_levels = this_info['noise_levels']    
        stim_types = this_info['stim_types']
        phase_vals = this_info['phase_vals']
        contrast_levels = this_info['contrast_levels']
        
        sf_vals = this_info['sf_vals']
    
        
    else:
        
        for ll in range(nLayers):
            for tt in range(nTimePts):
                allw[ll][tt] = np.concatenate((allw[ll][tt], this_allw[ll][tt]))
                
        orilist = np.concatenate((orilist,this_info['orilist']))
        phaselist=  np.concatenate((phaselist,this_info['phaselist']))
        sflist = np.concatenate((sflist, this_info['sflist']+np.max(sflist)+1))
        typelist = np.concatenate((typelist,this_info['typelist']))
        noiselist = np.concatenate((noiselist, this_info['noiselist']))
        exlist = np.concatenate((exlist,this_info['exlist']))
        contrastlist = np.concatenate((contrastlist, this_info['contrastlist']))
        
        
        assert nLayers == this_info['nLayers']
        assert nPhase == this_info['nPhase']
        assert nSF == this_info['nSF']
        assert nType == this_info['nType']
        assert nTimePts == this_info['nTimePts']
        assert nNoiseLevels == this_info['nNoiseLevels']
        assert nEx == this_info['nEx']
        assert nContrastLevels == this_info['nContrastLevels']
        
        sf_vals = np.concatenate((sf_vals,this_info['sf_vals']))
        
nSF=np.size(np.unique(sflist))

if not np.all(np.char.find(model_str,'oriTst12')==-1):
    sf_tmp = deepcopy(sflist)
    sf_vals_tmp = deepcopy(sf_vals)
    
    sflist[sf_tmp==1] = 4
    sflist[sf_tmp==3] = 1
    sflist[sf_tmp==4] = 3
    
    sf_vals[1] = sf_vals_tmp[4]
    sf_vals[3] = sf_vals_tmp[1]
    sf_vals[4] = sf_vals_tmp[3]
    
assert np.all(sf_vals==np.sort(sf_vals,0))
#%% plot discriminability across 0-360 space, overlay spatial frequencies
    
#assert 'oriTst11' in model_str
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

    w = allw[ww1][0]
    
    for sf in range(np.size(sf2plot)):
        
        inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
       
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
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
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\nContrast %.2f Noise %.2f' % (model_str,contrast_levels[cc],noise_levels[nn]))
             
    xx=xx+1
  
#%%  Plot the standardized euclidean distance, in 0-360 space
# Overlay SF
# plot just one layer, save a figure
assert phase_vals==[0,180]
 
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')

from matplotlib import cm


layers2plot = [12]
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
       
        plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2],order='F'), axis=1),color = cols_sf[sf,0,:])

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
#    plt.savefig(figname, format='pdf',transparent=True)
     
#%%  Plot the standardized euclidean distance, in 0-360 space
# plot subplots
    
assert phase_vals==[0,180]
 
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[12,5]

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')

from matplotlib import cm


layers2plot = [12]
sf2plot = np.arange(0,6,1)
#sf2plot = [3]
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

colors = cm.tab20b(np.linspace(0,1,np.size(sf2plot)))
#colors = [cm.Greens(0.5),cm.Purples(0.5),cm.Reds(0.5),cm.Blues(0.5),cm.PuRd(0.5),cm.Oranges(0.5)]

for ww1 in layers2plot:
    plt.figure()
    
    w = allw[ww1][0]
    
    for sf in range(np.size(sf2plot)):
        
        plt.subplot(2,3,sf+1)
        
        inds = np.where(sflist==sf2plot[sf])[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
       
        plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2],order='F'), axis=1),color = 'k')

        plt.title('%.2f cpp' % (sf_vals[sf2plot[sf]]/10))
        
        if sf>2:
            plt.xlabel('Grating Orientation')
            plt.xticks(np.arange(0,181,45))
        else:
            plt.xticks(np.arange(0,181,45),[])
        if sf==0 or sf==3:
            plt.ylabel('Discriminability (a.u.)')

        plt.yticks([])
      
        for ll in np.arange(0,181,45):
            plt.axvline(ll,color=[0.8,0.8,0.8,1])
       
        plt.xlim([0,180])
    

    figname = os.path.join(figfolder, 'SpatFreq','%s_%s_discrim_subplots.pdf' % (model_str,layer_labels[ww1]))
    plt.savefig(figname, format='pdf',transparent=True)

#%% plot HORIZONTAL anisotropy in 0-360 space, overlay SF
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
a = np.arange(90,360,180)
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
plt.title('%s\nHORIZONTAL anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_HorizontalAnisotropySF.pdf' % (model_str))
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
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
       
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
plt.title('%s\nVERTICAL anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_VerticalAnisotropySF.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)

    
#%% plot Cardinal (V+H) anisotropy in 0-360 space, overlay SF
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
plt.title('%s\nCardinal anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_AnisotropySF.pdf' % (model_str))
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
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
       
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
plt.title('%s\nOblique (45 only) anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_ObliqueAnisotropySF.pdf' % (model_str))
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
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
       
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
plt.title('%s\nOblique (135 only) anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_ObliqueAnisotropySF.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)

#%% plot OBLIQUE (45 + 135) anisotropy in 0-360 space, overlay SF
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
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
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
       
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
plt.title('%s\nOblique anisotropy' % (model_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'SpatFreq','%s_ObliqueAnisotropySF.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)
     
#%% plot Mean Discrim in 0-360 space, overlay SF
assert phase_vals==[0,180]

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
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,0,:])

#ylims = [-1,1]
xlims = [-1, np.size(layers2plot)+1]

#plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.title('%s\nMean Discriminability' % (model_str))
plt.ylabel('d''')
#plt.xlabel('Layer number')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

figname = os.path.join(figfolder, 'SpatFreq','%s_MeanDiscSF.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)
 

#%% plot Cardinal Versus Mean discriminability (scatter)
assert phase_vals==[0,180]
 
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
    discrim_vals = np.zeros([np.size(layers2plot),1])
    
    for ww1 in range(np.size(layers2plot)):
               
        w = allw[layers2plot[ww1]][0]
               
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
       
        discrim_vals[ww1] = np.mean(disc)
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
      
    aniso_vals = np.mean(aniso_vals,0)
 
    plt.plot(discrim_vals,aniso_vals,'.',color = cols_sf[sf,0,:])

#ylims = [-.2,1]
#xlims = [-1, np.size(layers2plot)+1]

plt.legend(['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
#plt.xlim(xlims)
#plt.ylim(ylims)
#plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (model_str))
plt.ylabel('Anisotropy')
plt.xlabel('Mean discrim')

figname = os.path.join(figfolder, 'SpatFreq','%s_Scatter.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)

#%% plot mean discrim versus SF, overlay layers

assert phase_vals==[0,180]

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [21]
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['%s'%layer_labels[ll] for ll in layers2plot]
layer_colors = cm.plasma(np.linspace(0,1,nLayers))

for ww1 in range(np.size(layers2plot)):
           
    w = allw[layers2plot[ww1]][0]
    
    vals = np.zeros([np.size(sf2plot),1])
         
    for sf in range(np.size(sf2plot)):
    
        inds1 = np.where(sflist==sf2plot[sf])[0]
                         
        ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
     
        vals[sf] = np.mean(disc)
 
    vals = stats.zscore(vals)
    plt.plot(sf_vals[sf2plot],vals,color=layer_colors[layers2plot[ww1],:])

#ylims = [-1,1]
#xlims = [-1, np.size(layers2plot)+1]

plt.figlegend(legendlabs,loc='right')

plt.title('%s\nMean Discriminability' % (model_str))
plt.ylabel('z-score')
plt.xlabel('Spatial Frequency')
plt.xscale('log')
#plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

#figname = os.path.join(figfolder, 'SpatFreq','%s_MeanDiscSF.pdf' % (model_str))
#plt.savefig(figname, format='pdf',transparent=True)
 
#%% plot mean discrim versus SF, overlay layers

assert phase_vals==[0,180]

orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
#layers2plot = [21]
sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['%s'%layer_labels[ll] for ll in layers2plot]
layer_colors = cm.plasma(np.linspace(0,1,nLayers))

for ww1 in range(np.size(layers2plot)):
           
    w = allw[layers2plot[ww1]][0]
    
    vals = np.zeros([np.size(sf2plot),1])
         
    for sf in range(np.size(sf2plot)):
    
        inds1 = np.where(sflist==sf2plot[sf])[0]
        activ_patterns = w[inds1,:]
       
        vals[sf] = np.mean(activ_patterns.flatten())
 
    vals = stats.zscore(vals)
    plt.plot(sf_vals[sf2plot],vals,color=layer_colors[layers2plot[ww1],:])

#ylims = [-1,1]
#xlims = [-1, np.size(layers2plot)+1]

plt.figlegend(legendlabs,loc='right')
plt.title('%s\nMean Signal' % (model_str))
plt.ylabel('z-score')
plt.xlabel('Spatial Frequency')
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
        
        inds = np.where(np.all([sflist==sf2plot[sf]],axis=0))[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist[inds])
       
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(ori_axis,disc)

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,np.max(ori_axis),45))
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
        