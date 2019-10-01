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

#%% get the data ready to go...then can run any below cells independently.

model_name_2plot = 'VGG-16'

root = '/usr/local/serenceslab/maggie/biasCNN/';

import os
os.chdir(os.path.join(root, 'code'))
figfolder = os.path.join(root, 'figures')

import load_activations
#allw, all_labs, info = load_activations.load_activ_nasnet_oriTst0()
nSF = 6
nContrastLevels = 6

cols1 = np.reshape(cm.tab20b(np.arange(0,20,1)), [5, 4, 4])
cols2 = np.reshape(cm.tab20c(np.arange(0,20,1)), [5, 4, 4])
cols_all = np.concatenate((cols1,cols2),axis=0)
cols_all = cols_all[[4, 6, 2, 7, 5, 0],:,:]
cols_all_interp = np.zeros([nSF,nContrastLevels,4])
for ii in range(nSF):
    for jj in range(4):
        cols_all_interp[ii,:,jj] = np.interp([0,1,2,3,4,5],[0,2,4,6],cols_all[ii,:,jj])
        
cols_sf_noise = cols_all_interp
#%% load multiple datasets or just one

model_str = ['vgg16_oriTst12', 'vgg16_oriTst12a','vgg16_oriTst12b']

for mm in range(np.size(model_str)):
    
    this_allw, this_all_labs, this_info = load_activations.load_activ(model_str[mm])

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
        
nSF = np.size(np.unique(sflist))

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
#%%  Plot Discriminability Function, overlay noise levels, single SF
# subplots for all layers
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')
plt.figure()
xx=1
sf=3

layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

for ww1 in layers2plot:

    w = allw[ww1][0]
    
    for nn in range(np.size(noise2plot)):
        
        inds = np.where(np.all([sflist==sf, noiselist==noise2plot[nn]],axis=0))[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
       
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2],order='F'), axis=1),color=cols_sf_noise[sf,nn,:])

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
#        plt.plot([ll,ll],plt.gca().get_ylim(),'k')
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\nSF=%.2f' % (model_str, sf_vals[sf]))
             
    xx=xx+1
      
#%%  Plot Discriminability Function, overlay noise levels, single SF
# Overlay noise levels
# plot just one layer and save the figure
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[8,5]

plt.figure()

assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')

sf=4

layers2plot = [6]
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

cols_noise = cm.Greys(np.linspace(1,0,5))
#colors = colors[[2,3,4],:]

for ww1 in layers2plot:

    plt.figure()
    
    w = allw[ww1][0]
    
    for nn in range(np.size(noise2plot)):
        
        inds = np.where(np.all([sflist==sf, noiselist==noise2plot[nn]],axis=0))[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
       
        plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2],order='F'), axis=1),color=cols_noise[nn,:])

    plt.title('%s\nSF=%.2fcpd' % (layer_labels[ww1], sf_vals[sf]))

    if ww1==layers2plot[-1]:
        plt.xlabel('Stimulus Orientation')
        plt.ylabel('Discriminability (a.u.)')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
       plt.axvline(ll,color=[0.8,0.8,0.8,1])

    plt.xlim([0,180])
    
#    xx=xx+1

    figname = os.path.join(figfolder, 'Noise','VGG16_%s_SF=%.2f_discrim.pdf' % (model_str,layer_labels[ww1],sf_vals[sf]))
    plt.savefig(figname, format='pdf',transparent=True)

#%% plot Cardinal anisotropy, overlay Noise levels
# plotting one SF at a time.

cols_noise = cm.Greys(np.linspace(1,0,5))
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
     
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

b = np.arange(22.5,360,45)
a = np.arange(0,360,90)
bin_size = 4
sf=4

for nn in noise2plot:
    
    inds1 = np.where(np.all([sflist==sf, noiselist==noise2plot[nn]],axis=0))[0]
           
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
 
    plt.plot(np.arange(0,np.size(layers2plot),1),vals,color=cols_noise[nn,:])

ylims = [-0.2,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(legendlabs)
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy\nSF=%.2f' % (model_str, sf_vals[sf]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

figname = os.path.join(figfolder, 'Noise','VGG16_noise_SF%.2f_aniso.pdf' % (sf_vals[sf]))
plt.savefig(figname, format='pdf',transparent=True)

 #%% plot overall discriminability, overlay noise levels
# just one  SF
cols_noise = cm.Greys(np.linspace(1,0,5))
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
     
plt.close('all')
plt.figure()
#xx=0
layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
sf2plot = [4]

legendlabs =[]
#for sf in sf2plot: 
for nn in noise2plot:           
    legendlabs.append('Noise=%.2f'%(noise_levels[nn]))
    
b = np.arange(22.5,360,45)
a = np.arange(0,360,90)
bin_size = 4

for sf in range(np.size(sf2plot)):
    
#    xx=xx+1
#    plt.subplot(np.size(sf2plot)/2,2,xx)
    
    for nn in noise2plot:
    
        inds1 = np.where(np.all([sflist==sf2plot[sf], noiselist==noise2plot[nn]],axis=0))[0]
#        print(np.size(inds1))
        vals = np.zeros([np.size(layers2plot),1])
        
        for ww1 in range(np.size(layers2plot)):
                   
            w = allw[layers2plot[ww1]][0]
                   
            ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])

            vals[ww1] = np.mean(disc)
     
        plt.plot(np.arange(0,np.size(layers2plot),1),vals,color=cols_noise[nn,:])

    ylims = [-0.2,1]
    xlims = [-1, np.size(layers2plot)]
    
#    plt.legend(legendlabs)
    plt.plot(xlims, [0,0], 'k')
    plt.xlim(xlims)
#    plt.ylim(ylims)
    plt.title('SF=%.2fcpd'%sf_vals[sf2plot[sf]])
    if sf==np.size(sf2plot)-1:
        plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
    else:
        plt.xticks(np.arange(0,np.size(layers2plot),1),[])
    plt.ylabel('d-prime')
    
#plt.suptitle('%s\nCardinal anisotropy' % (model_str))

figname = os.path.join(figfolder, 'Noise','VGG16_noise_SF%.2f_meandiscrim.pdf' % (sf_vals[sf2plot[sf]]))
plt.savefig(figname, format='pdf',transparent=True)
 
#%% plot overall discriminability, overlay noise levels
# separate subplot for each SF

assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
     
plt.close('all')
plt.figure()
xx=0
layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
sf2plot = np.arange(0,6,1)

legendlabs =[]
#for sf in sf2plot: 
for nn in noise2plot:           
    legendlabs.append('Noise=%.2f'%(noise_levels[nn]))
    
b = np.arange(22.5,360,45)
a = np.arange(0,360,90)
bin_size = 4

for sf in range(np.size(sf2plot)):
    
    xx=xx+1
    plt.subplot(np.size(sf2plot)/2,2,xx)
    
    for nn in noise2plot:
    
        inds1 = np.where(np.all([sflist==sf2plot[sf], noiselist==noise2plot[nn]],axis=0))[0]
#        print(np.size(inds1))
        vals = np.zeros([np.size(layers2plot),1])
        
        for ww1 in range(np.size(layers2plot)):
                   
            w = allw[layers2plot[ww1]][0]
                   
            ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])

            vals[ww1] = np.mean(disc)
     
        plt.plot(np.arange(0,np.size(layers2plot),1),vals,color=cols_sf_noise[sf,nn,:])

    ylims = [-0.2,1]
    xlims = [-1, np.size(layers2plot)]
    
#    plt.legend(legendlabs)
    plt.plot(xlims, [0,0], 'k')
    plt.xlim(xlims)
#    plt.ylim(ylims)
    plt.title('SF=%.2fcpd'%sf_vals[sf2plot[sf]])
    if sf==np.size(sf2plot)-1:
        plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
    else:
        plt.xticks(np.arange(0,np.size(layers2plot),1),[])
    plt.ylabel('d-prime')
     
#%% plot anisotropy, overlay noise levels
# separate subplot for each SF


plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[15,5]

assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
     
plt.close('all')
plt.figure()
xx=0
layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
sf2plot = np.arange(0,6,1)

legendlabs =[]
#for sf in sf2plot: 
for nn in noise2plot:           
    legendlabs.append('Noise=%.2f'%(noise_levels[nn]))
    
b = np.arange(22.5,360,45)
a = np.arange(0,360,90)
bin_size = 6

for sf in range(np.size(sf2plot)):
    
    xx=xx+1
    plt.subplot(2, np.size(sf2plot)/2,xx)
    
    for nn in noise2plot:
    
        inds1 = np.where(np.all([sflist==sf2plot[sf], noiselist==noise2plot[nn]],axis=0))[0]
#        print(np.size(inds1))
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
     
        plt.plot(np.arange(0,np.size(layers2plot),1),vals,color=cols_sf_noise[sf,nn,:])

    ylims = [-0.2,1]
    xlims = [-1, np.size(layers2plot)]
    
#    plt.legend(legendlabs)
    plt.plot(xlims, [0,0], 'k')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.title('SF=%.2fcpd'%sf_vals[sf2plot[sf]])
    if sf==np.size(sf2plot)-1:
        plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
    else:
        plt.xticks(np.arange(0,np.size(layers2plot),1),[])
    plt.ylabel('Anisotropy')

#%%  Plot the standardized euclidean distance, in 0-180 space
# Overlay noise levels
plt.close('all')
plt.figure()
xx=1

layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

for ww1 in layers2plot:

    w = allw[ww1][0]
    
    for nn in range(np.size(noise2plot)):
        
        inds = np.where(noiselist==noise2plot[nn])[0]
        
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



#%% plot Cardinal anisotropy, overlay Noise levels
      
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

b = np.arange(22.5,180,45)
a = np.arange(0,180,90)
bin_size = 4

for nn in noise2plot:
    
    inds1 = np.where(noiselist==noise2plot[nn])[0]
           
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

#%% plot Mean discrim overlay noise levels
      
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
noise2plot = np.arange(0,3,1)
legendlabs = ['noise=%.2f'%(noise_levels[nn]) for nn in noise2plot]

      
for nn in noise2plot:
    
    inds1 = np.where(noiselist==noise2plot[nn])[0]
             
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
