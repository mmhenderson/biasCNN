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

model_str = 'vgg16_oriTst11'
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

layer_labels = np.array(info['layer_labels'])
#layer_labels_plot = np.array(info['layer_labels_plot'])
sf_vals = np.array(info['sf_vals'])
noise_levels = np.array(info['noise_levels'])
timepoint_labels = np.array(info['timepoint_labels'])
stim_types = np.array(info['stim_types'])
phase_vals = np.array(info['phase_vals'])
contrast_levels = np.array(info['contrast_levels'])

nLayers = info['nLayers']
nPhase = info['nPhase']
nSF = info['nSF']
nType = info['nType']
nTimePts = info['nTimePts']
nNoiseLevels = info['nNoiseLevels']
nEx = info['nEx']
nContrastLevels = info['nContrastLevels']


            
#%% PCA , plotting pts by orientation, with spatial frequency markers

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[10,8]

plt.close('all')

#layers2plot = np.arange(0,8,1)
layers2plot = np.arange(0,22,6)
ww2 = 0
nn=0
sf2plot = np.arange(0,6,1)

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
        ori2plot = np.arange(0,180,1)
    
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
       
#%% PCA , plotting pts by orientation - just a subset of orientations around zero
            
plt.close('all')
layers2plot= np.arange(14,15,1)

timepts2plot = np.arange(0,1)
noiselevels2plot = [0]
ori2plot = [175, 176, 177,178, 179, 0, 1, 2, 3, 4]

clist = cm.plasma(np.linspace(0,1,np.size(ori2plot)))

cc=0

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
                
                myinds = np.where(np.all([contrastlist==cc, orilist==ori2plot[oo],sflist==sf, noiselist==nn],axis=0))[0]
             
                if ori2plot[oo]==0:
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
            plt.figlegend(h, ori2plot)

            plt.plot(allpts[:,0],allpts[:,1],'k')
            
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
                

