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


root = '/usr/local/serenceslab/maggie/biasCNN/';
nSF = 6
nContrastLevels = 6

cols1 = np.reshape(cm.tab20b(np.arange(0,20,1)), [5, 4, 4])
cols2 = np.reshape(cm.tab20c(np.arange(0,20,1)), [5, 4, 4])
cols_all = np.concatenate((cols1,cols2),axis=0)
cols_all = cols_all[[4, 6, 2, 7, 5, 0],:,:]
cols_all_interp = np.zeros([nSF,nContrastLevels,4])
for ii in range(nSF):
    for jj in range(4):
        cols_all_interp[ii,:,jj] = np.interp([5,4,3,2,1,0],[0,2,4,6],cols_all[ii,:,jj])
        
cols_all = cols_all_interp

import os
os.chdir(os.path.join(root, 'code'))
figfolder = os.path.join(root, 'figures')

import load_activations
#%% load multiple datasets or just one

model_str = ['vgg16_oriTst13a', 'vgg16_oriTst13b','vgg16_oriTst13c','vgg16_oriTst13d', 'vgg16_oriTst13e','vgg16_oriTst13f']
model_name_2plot = 'VGG-16'

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
        
nSF=np.size(np.unique(sflist))

#%% plot discriminability across 0-360 space, overlay contrast levels
    
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
 
plt.close('all')
plt.figure()
xx=1

layers2plot = np.arange(0,nLayers,1)
contrast2plot = np.arange(0,6,1)
legendlabs = ['contrast=%.2f'%(contrast_levels[cc]) for cc in contrast2plot]

colors = cm.Greens(np.linspace(0.2, 1, np.size(contrast2plot)))

for ww1 in layers2plot:

    w = allw[ww1][0]
    
    for cc in range(np.size(contrast2plot)):
        
        inds = np.where(contrastlist==contrast2plot[cc])[0]
        
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
       
        plt.subplot(np.ceil(len(layers2plot)/4), 4, xx)
    
        plt.plot(ori_axis[0:180],np.mean(np.reshape(disc, [180,2]), axis=1),color=colors[cc,:])

    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('Grating orientation')
        plt.ylabel('d''')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
   
   
    plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations' % (model_str))
             
    xx=xx+1

#%% plot anisotropy, overlay contrast levels
# separate subplot for each SF
    
    
cols1 = np.reshape(cm.tab20b(np.arange(0,20,1)), [5, 4, 4])
cols2 = np.reshape(cm.tab20c(np.arange(0,20,1)), [5, 4, 4])
cols_all = np.concatenate((cols1,cols2),axis=0)
cols_all = cols_all[[4, 6, 2, 7, 5, 0],:,:]
cols_all_interp = np.zeros([nSF,nContrastLevels,4])
for ii in range(nSF):
    for jj in range(4):
        cols_all_interp[ii,:,jj] = np.interp([5,4,3,2,1,0],[0,2,4,6],cols_all[ii,:,jj])
        
cols_all = cols_all_interp


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
contrast2plot = np.arange(1,6,2)
sf2plot = [0,1,2,3,4,5]

legendlabs =[]
#for sf in sf2plot: 
for cc in contrast2plot:           
    legendlabs.append('Contrast=%.2f'%(contrast_levels[cc]))

b = np.arange(22.5,360,45)
a = np.arange(0,360,90)
bin_size = 6


for sf in range(np.size(sf2plot)):
    
    xx=xx+1
    plt.subplot(2, np.size(sf2plot)/2,xx)
    
    for cc in range(np.size(contrast2plot)):
    
        inds1 = np.where(np.all([sflist==sf2plot[sf], contrastlist==contrast2plot[cc]],axis=0))[0]
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
     
        plt.plot(np.arange(0,np.size(layers2plot),1),vals,color=cols_all[sf,contrast2plot[cc],:])

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
    
#plt.suptitle('%s\nCardinal anisotropy' % (model_str))

figname = os.path.join(figfolder, 'Contrast','VGG16_contrast_vs_SF_subplots_anisotropy.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)
  
#%% plot overall discriminability, overlay contrast levels
# separate subplot for each SF
assert phase_vals==[0,180]
 
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
     
plt.close('all')
plt.figure()
xx=0
layers2plot = np.arange(0,nLayers,1)
contrast2plot = np.arange(1,6,2)
sf2plot = [0,1,2,3,4,5]

legendlabs =[]
#for sf in sf2plot: 
for cc in contrast2plot:           
    legendlabs.append('Contrast=%.2f'%(contrast_levels[cc]))

b = np.arange(22.5,360,45)
a = np.arange(0,360,90)
bin_size = 4

for sf in range(np.size(sf2plot)):
    
    xx=xx+1
    plt.subplot(np.size(sf2plot)/2,2,xx)
    
    for cc in range(np.size(contrast2plot)):
    
        inds1 = np.where(np.all([sflist==sf2plot[sf], contrastlist==contrast2plot[cc]],axis=0))[0]
#        print(np.size(inds1))
        vals = np.zeros([np.size(layers2plot),1])
        
        for ww1 in range(np.size(layers2plot)):
                   
            w = allw[layers2plot[ww1]][0]
                   
            ori_axis, disc = classifiers.get_discrim_func(w[inds1,:],orilist_adj[inds1])
           
            vals[ww1] = np.mean(disc)
     
        plt.plot(np.arange(0,np.size(layers2plot),1),vals,color=cols_all[sf,contrast2plot[cc],:])

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

figname = os.path.join(figfolder, 'Contrast','VGG16_contrast_vs_SF_subplots_discrim.pdf' % (model_str))
plt.savefig(figname, format='pdf',transparent=True)
  

#%% plot contrast response function for several layers
      
assert 'oriTst9a' in model_str

plt.close('all')

layers2plot = np.arange(0, nLayers, 1)

contrast2plot =np.arange(0,24,1)
#contrast2plot = [0,1]
ori_bins = np.linspace(0,157.5, 8)
bin_size = 4

plt.figure()
xx=0

for ww1 in range (np.size(layers2plot)):
    
    xx=xx+1
    plt.subplot(6,4,xx);    

    w = allw[layers2plot[ww1]][0]  
    
    for oo in range(np.size(ori_bins)):
          
        cr_func = np.zeros([np.size(contrast2plot),1])      
        cr_err = np.zeros([np.size(contrast2plot),1])      
    
        for cc in range(np.size(contrast2plot)):

            myinds_bool = np.all([contrastlist==contrast2plot[cc], np.logical_or(np.abs(orilist-ori_bins[oo])<bin_size/2, np.abs(orilist-(np.mod(180+ori_bins[oo],180)))<bin_size/2)], axis=0)
        
            act_vals = w[np.where(myinds_bool)[0],:]
            cr_func[cc] = np.mean(act_vals.flatten())
            cr_err[cc] = np.std(act_vals.flatten())/np.sqrt(np.size(act_vals))
        
        plt.plot(contrast_levels[contrast2plot],cr_func)
#            plt.errorbar(contrast_levels[contrast2plot],cr_func,cr_err)
            
    if ww1==np.max(layers2plot):
        plt.legend(['%.2f deg'%ori_bins[oo] for oo in range(np.size(ori_bins))])
        plt.xlabel('Stim. contrast')
    else:
        plt.xticks([])
        
    plt.ylabel('Average response')
        
    plt.plot(plt.gca().get_xlim(),[0,0],'k')
    plt.title('%s'%layer_labels[ww1])
    
plt.suptitle('%s\nContrast response function (mean response)' % (model_str))

 #%% plot contrast response function, averaging within groups of orientations    
assert 'oriTst9a' in model_str

plt.close('all')

layers2plot = np.arange(0, nLayers, 1)

contrast2plot =np.arange(0,24,1)
#contrast2plot = [0,1]
ori_bins = np.linspace(0,157.5, 8)
bin_size = 4

plt.figure()
xx=0

for ww1 in range (np.size(layers2plot)):
    
    xx=xx+1
    plt.subplot(6,4,xx);    

    w = allw[layers2plot[ww1]][0]
    
    cr_func = np.zeros([np.size(contrast2plot),np.size(ori_bins)])      
    cr_err = np.zeros([np.size(contrast2plot),np.size(ori_bins)])      

    for oo in range(np.size(ori_bins)):
          
       
        for cc in range(np.size(contrast2plot)):

            myinds_bool = np.all([contrastlist==contrast2plot[cc], np.logical_or(np.abs(orilist-ori_bins[oo])<bin_size/2, np.abs(orilist-(np.mod(180+ori_bins[oo],180)))<bin_size/2)], axis=0)
        
            act_vals = w[np.where(myinds_bool)[0],:]
            cr_func[cc,oo] = np.mean(act_vals.flatten())
            cr_err[cc,oo] = np.std(act_vals.flatten())/np.sqrt(np.size(act_vals))
        
    card_func = np.mean(cr_func[:,[0,4]], axis=1)
    oblique_func = np.mean(cr_func[:,[2, 6]], axis=1)
    middle_func = np.mean(cr_func[:,[1,3,5,7]], axis=1)
    
    plt.plot(contrast_levels[contrast2plot],card_func)
    plt.plot(contrast_levels[contrast2plot],oblique_func)
    plt.plot(contrast_levels[contrast2plot],middle_func)
#            plt.errorbar(contrast_levels[contrast2plot],cr_func,cr_err)
            
    if ww1==np.max(layers2plot):
        plt.legend(['cardinals','obliques','middles'])
        plt.xlabel('Stim. contrast')
    else:
        plt.xticks([])
        
    plt.ylabel('Average response')
        
    plt.plot(plt.gca().get_xlim(),[0,0],'k')
    plt.title('%s'%layer_labels[ww1])
    
plt.suptitle('%s\nContrast response function (mean response)' % (model_str))