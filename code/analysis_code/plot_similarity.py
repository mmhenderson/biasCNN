#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn import decomposition 
from matplotlib import cm

   
#%% get the data ready to go...then can run any below cells independently.
#model_str = ['vgg16_oriTst13a', 'vgg16_oriTst13b','vgg16_oriTst13c','vgg16_oriTst13d', 'vgg16_oriTst13e','vgg16_oriTst13f']
#model_str = ['vgg16_oriTst12', 'vgg16_oriTst12a','vgg16_oriTst12b']
#model_str = ['vgg16_oriTst11']
#model_name_2plot = 'VGG-16'
#model_str = ['vgg16_oriTst11']
model_str = ['scratch_vgg16_imagenet_rot_0_oriTst11']
#model_name_2plot = 'VGG-16'
model_name_2plot = 'VGG-16-TRAIN-ROT-0'


root = '/usr/local/serenceslab/maggie/biasCNN/';

import os
os.chdir(os.path.join(root, 'code'))
figfolder = os.path.join(root, 'figures')

import load_activations


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


            
#%% PCA , plotting pts by orientation, with spatial frequency markers

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[10,8]

plt.close('all')

#layers2plot = np.arange(0,8,1)
#layers2plot = np.arange(0,22,6)
layers2plot = [4,5,6,7]
ww2 = 0
nn=0
sf2plot = np.arange(0,6,1)
ori2plot = np.arange(0,180,1)
c_map = cm.get_cmap('plasma')
markers = ['^','+','o','p','*','3']

pc2plot = [0,1]

for ww1 in layers2plot:
   
    w = allw[ww1][ww2]

              
    plt.figure()
    for sf in range(np.size(sf2plot)):
        
#        if sf==0:
#            ori2plot = np.arange(0,180,10)
#        elif sf==1:
#            ori2plot = np.arange(0,180,5)
#        elif sf==2:
#            ori2plot = np.arange(0,180,5)
#        else:
    
        myinds = np.where(np.all([np.isin(orilist,ori2plot), exlist==0, phaselist==0, noiselist==nn, sflist==sf2plot[sf]],axis=0))[0]
     
        sc = plt.scatter(w[myinds,pc2plot[0]], w[myinds,pc2plot[1]],
                         c=orilist[myinds,0],
                         vmin = 0,vmax = 180, cmap=c_map,marker=markers[sf2plot[sf]])
             
    plt.title('Layer %d of %d\n%s' % (ww1+1,nLayers,layer_labels[ww1]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(sc,ticks=[0,45,90,135,180])
    plt.xticks([])
    plt.yticks([])
    figname = os.path.join(figfolder, 'PCA','%s_%s_allSF_PC1_vs_PC2.eps' % (model_str,layer_labels[ww1]))
    plt.savefig(figname, format='pdf',transparent=True)
       
#%% PCA , color pts by orientation - markers are spatial frequency. DONT save plots

plt.close('all')

#layers2plot = np.arange(0,8,1)
layers2plot = np.arange(0,22,6)
ww2 = 0
nn=0
sf2plot = np.arange(0,6,1)
ori2plot = np.arange(0,180,1)
c_map = cm.get_cmap('plasma')
markers = ['^','+','o','p','*','3']

pc2plot = [0,1]
cc=0

for ww1 in layers2plot:
   
    w = allw[ww1][ww2]

              
    plt.figure()
    for sf in range(np.size(sf2plot)):
        
#        if sf==0:
#            ori2plot = np.arange(0,180,10)
#        elif sf==1:
#            ori2plot = np.arange(0,180,5)
#        elif sf==2:
#            ori2plot = np.arange(0,180,5)
#        else:

        myinds = np.where(np.all([np.isin(orilist,ori2plot), exlist==0, noiselist==nn, contrastlist==cc, sflist==sf2plot[sf]],axis=0))[0]
     
        sc = plt.scatter(w[myinds,pc2plot[0]], w[myinds,pc2plot[1]],
                         c=orilist[myinds,0],
                         vmin = 0,vmax = 180, cmap=c_map,marker=markers[sf2plot[sf]])
             
    plt.title('Layer %d of %d\n%s' % (ww1+1,nLayers,layer_labels[ww1]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(sc,ticks=[0,45,90,135,180])
    plt.xticks([])
    plt.yticks([])
    
#%% Color points by orientation, markers are contrast levels
plt.close('all')

#layers2plot = np.arange(0,8,1)
layers2plot = np.arange(0,22,6)
ww2 = 0
nn=0
sf=4
ori2plot = np.arange(0,180,1)
c_map = cm.get_cmap('plasma')
markers = ['^','+','o','p','*','3']

pc2plot = [0,1]
contrast2plot =np.arange(0,6,1);

for ww1 in layers2plot:
   
    w = allw[ww1][ww2]

              
    plt.figure()
    for cc in range(np.size(contrast2plot)):
        
#        if sf==0:
#            ori2plot = np.arange(0,180,10)
#        elif sf==1:
#            ori2plot = np.arange(0,180,5)
#        elif sf==2:
#            ori2plot = np.arange(0,180,5)
#        else:

        myinds = np.where(np.all([np.isin(orilist,ori2plot), exlist==0, noiselist==nn, contrastlist==contrast2plot[cc], sflist==sf],axis=0))[0]
     
        sc = plt.scatter(w[myinds,pc2plot[0]], w[myinds,pc2plot[1]],
                         c=orilist[myinds,0],
                         vmin = 0,vmax = 180, cmap=c_map,marker=markers[contrast2plot[cc]])
             
    plt.title('Layer %d of %d\n%s' % (ww1+1,nLayers,layer_labels[ww1]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(sc,ticks=[0,45,90,135,180])
    plt.xticks([])
    plt.yticks([])
    
#%% Color points by orientation, markers are noise levels
plt.close('all')

#layers2plot = np.arange(0,8,1)
layers2plot = np.arange(0,22,6)
ww2 = 0
cc=0
sf=4
ori2plot = np.arange(0,180,1)
c_map = cm.get_cmap('plasma')
markers = ['^','+','o','p','*','3']

pc2plot = [0,1]
noise2plot =np.arange(0,3,1);

for ww1 in layers2plot:
   
    w = allw[ww1][ww2]

              
    plt.figure()
    for nn in range(np.size(noise2plot)):
        
#        if sf==0:
#            ori2plot = np.arange(0,180,10)
#        elif sf==1:
#            ori2plot = np.arange(0,180,5)
#        elif sf==2:
#            ori2plot = np.arange(0,180,5)
#        else:

        myinds = np.where(np.all([np.isin(orilist,ori2plot), exlist==0, noiselist==noise2plot[nn], contrastlist==cc, sflist==sf],axis=0))[0]
     
        sc = plt.scatter(w[myinds,pc2plot[0]], w[myinds,pc2plot[1]],
                         c=orilist[myinds,0],
                         vmin = 0,vmax = 180, cmap=c_map,marker=markers[noise2plot[nn]])
             
    plt.title('Layer %d of %d\n%s' % (ww1+1,nLayers,layer_labels[ww1]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(sc,ticks=[0,45,90,135,180])
    plt.xticks([])
    plt.yticks([])
            
#%% Plot points just around 0, Color points by their orientation
            
plt.close('all')
layers2plot= np.arange(14,15,1)

timepts2plot = np.arange(0,1)
noiselevels2plot = [0]
#ori2plot = np.arange(0,180,1);
ori2plot = [175, 176, 177,178, 179, 0, 1, 2, 3, 4]
#ori2plot = np.arange(85,95,1)
clist = cm.plasma(np.linspace(0,1,np.size(ori2plot)))

markers = ['^','+','o','x']
sf = 0
pc2plot = [0,1]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:

            w = allw[ww1][ww2]

            myinds = np.where(noiselist==nn)[0]
                
            plt.figure()
            allpts = np.zeros([np.size(ori2plot), 2])
            h = [];
            for oo in range(np.size(ori2plot)):
                
                myinds = np.where(np.all([orilist==ori2plot[oo],sflist==sf, noiselist==nn],axis=0))[0]
             
                if ori2plot[oo]==90 or ori2plot[oo]==0:
                    mark = markers[0]
                else:
                    mark = markers[3]
                sc = plt.scatter(w[myinds,pc2plot[0]], w[myinds,pc2plot[1]],
                                 c=[clist[oo,:],],marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(w[myinds,pc2plot[0]],0), np.mean(w[myinds,pc2plot[1]],0)]
                sc = plt.scatter(allpts[oo,0],allpts[oo,1],
                                 c=[clist[oo,:],],marker=mark)
            ax = plt.gca()               
            plt.title('PC %d versus %d, %s-%s, noise=%.2f, sf=%.2f' % (pc2plot[0],pc2plot[1],layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn], sf_vals[sf]))
            plt.xlabel('PC%d' %pc2plot[0])
            plt.ylabel('PC%d' % pc2plot[1])
            plt.figlegend(h, ['%d deg'%ori2plot[oo] for oo in range(np.size(ori2plot))])
            plt.plot(allpts[:,0],allpts[:,1],'k')

#%% Color points by their contrast level
            
plt.close('all')
layers2plot= np.arange(14,15,1)

timepts2plot = np.arange(0,1)
noiselevels2plot = [0]
ori2plot = np.arange(0,180,1);
#ori2plot = [175, 176, 177,178, 179, 0, 1, 2, 3, 4]
#ori2plot = np.arange(85,95,1)
clist = cm.plasma(np.linspace(0,1,nContrastLevels))

ee=0

markers = ['^','+','o','x']
sf = 0
pc2plot = [0,1]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:

            w = allw[ww1][ww2]

            myinds = np.where(noiselist==nn)[0]
                
            plt.figure()
            allpts = np.zeros([np.size(ori2plot), 2])
            h = [];
            for oo in range(np.size(ori2plot)):

                myinds = np.where(np.all([orilist==ori2plot[oo],sflist==sf, noiselist==nn],axis=0))[0]
                colors = contrast_levels[np.squeeze(contrastlist[myinds].astype('int'))]
                if ori2plot[oo]==90 or ori2plot[oo]==0:
                    mark = markers[0]
                else:
                    mark = markers[3]
                sc = plt.scatter(w[myinds,pc2plot[0]], w[myinds,pc2plot[1]],
                                 c=colors,
                                 vmin = 0,vmax=1, cmap=cm.get_cmap('plasma'),marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(w[myinds,pc2plot[0]],0), np.mean(w[myinds,pc2plot[1]],0)]
                
            ax = plt.gca()               
            plt.title('PC %d versus %d, %s-%s, noise=%.2f, sf=%.2f\nColor=Contrast level' % (pc2plot[0],pc2plot[1],layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn], sf_vals[sf]))
            plt.xlabel('PC%d' %pc2plot[0])
            plt.ylabel('PC%d' % pc2plot[1])
#            plt.figlegend(h, ori2plot)

            plt.plot(allpts[:,0],allpts[:,1],'k')
            plt.colorbar(ticks =[0,0.5, 1])

#%% Color the points by phase
            
plt.close('all')
layers2plot= np.arange(0,1,1)

timepts2plot = np.arange(0,1)
noiselevels2plot = [0]
#ori2plot = [175, 176, 177,178, 179, 0, 1, 2, 3, 4]
ori2plot = np.arange(85,95,1)
#ori2plot = np.arange(0,180,1)
clist = cm.plasma(np.linspace(0,1,nPhase))

ee=0

markers = ['^','+','o','x']
sf = 0
pc2plot = [0,1]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:

            w = allw[ww1][ww2]

            myinds = np.where(noiselist==nn)[0]
                
            plt.figure()
            allpts = np.zeros([np.size(ori2plot), 2])
            h = [];
            for oo in range(np.size(ori2plot)):

                myinds = np.where(np.all([orilist==ori2plot[oo],sflist==sf, noiselist==nn],axis=0))[0]
                colors = phase_vals[np.squeeze(phaselist[myinds].astype('int'))]
                if ori2plot[oo]==90 or ori2plot[oo]==0:
                    mark = markers[0]
                else:
                    mark = markers[3]
                sc = plt.scatter(w[myinds,pc2plot[0]], w[myinds,pc2plot[1]],
                                 c=colors,
                                 vmin = 0,vmax=360, cmap=cm.get_cmap('plasma'),marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(w[myinds,pc2plot[0]],0), np.mean(w[myinds,pc2plot[1]],0)]
                
            ax = plt.gca()               
            plt.title('PC %d versus %d, %s-%s, noise=%.2f, sf=%.2f\nColor=phase' % (pc2plot[0],pc2plot[1],layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn], sf_vals[sf]))
            plt.xlabel('PC%d' %pc2plot[0])
            plt.ylabel('PC%d' % pc2plot[1])
#            plt.figlegend(h, ori2plot)

            plt.plot(allpts[:,0],allpts[:,1],'k')
            plt.colorbar(ticks =[0,90,180,270,360])
        
#%% Color the points by spatial frequency
            
plt.close('all')
layers2plot= np.arange(0,8,2)

timepts2plot = np.arange(0,1)
noiselevels2plot = [0]
#ori2plot = [175, 176, 177,178, 179, 0, 1, 2, 3, 4]
#ori2plot = np.arange(85,95,1)
ori2plot = np.arange(0,180,1)
clist = cm.plasma(np.linspace(0,1,nPhase))

ee=0

markers = ['^','+','o','x']
sf = 0
pc2plot = [0,1]

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:

            w = allw[ww1][ww2]

            myinds = np.where(noiselist==nn)[0]
                
            plt.figure()
            allpts = np.zeros([np.size(ori2plot), 2])
            h = [];
            for oo in range(np.size(ori2plot)):
#            for oo in range(1):

                myinds = np.where(np.all([orilist==ori2plot[oo],noiselist==nn],axis=0))[0]
                colors = sf_vals[np.squeeze(sflist[myinds].astype('int'))]
                if ori2plot[oo]==90 or ori2plot[oo]==0:
                    mark = markers[0]
                else:
                    mark = markers[3]
            
                sc = plt.scatter(w[myinds,pc2plot[0]], w[myinds,pc2plot[1]],
                                 c=colors,
                                 vmin = np.min(sf_vals)-0.01,vmax=np.max(sf_vals)+0.01, cmap=cm.get_cmap('plasma'),marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(w[myinds,pc2plot[0]],0), np.mean(w[myinds,pc2plot[1]],0)]
                
            ax = plt.gca()               
            plt.title('PC %d versus %d, %s-%s, noise=%.2f\nColor=Spatial Frequency' % (pc2plot[0],pc2plot[1],layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC%d' %pc2plot[0])
            plt.ylabel('PC%d' % pc2plot[1])
#            plt.figlegend(h, ori2plot)

            plt.plot(allpts[:,0],allpts[:,1],'k')
            plt.colorbar(ticks =np.linspace(np.min(sf_vals),np.max(sf_vals), 1))
                

