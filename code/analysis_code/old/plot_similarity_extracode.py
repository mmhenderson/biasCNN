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


            
#%% PCA , plotting pts by orientation
plt.close('all')
#layers2plot = np.arange(0,10,1)
layers2plot = [14]
timepts2plot = np.arange(0,1)
#noiselevels2plot = [0,1,2]
noiselevels2plot = [0,1,2]
#clist = cm.plasma(np.linspace(0,1,12))
c_map = cm.get_cmap('plasma')
markers = ['^','+','o','x','^','^']
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            
#            pca = decomposition.PCA(n_components = 4)
#            
#            weights_reduced = pca.fit_transform(allw[ww1][ww2])
            
            weights_reduced = allw[ww1][ww2]
#            nBins = int(12)
#            nPerBin = int(180/nBins)
#            binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
#            
#            legend_labs = [];
            myinds = np.where(noiselist==nn)[0]
                
            plt.figure()
            for sf in range(nSF):
#                myinds = np.where(np.logical_and(phaselist==1,np.logical_and(sflist==sf, noiselist==nn)))[0]
                myinds = np.where(np.logical_and(sflist==sf, noiselist==nn))[0]
             
                sc = plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],
                                 c=orilist[myinds,0],
                                 vmin = 0,vmax = 180, cmap=c_map,marker=markers[sf])
                
#            plt.figure(1)
            ax = plt.gca()               
            plt.title('PC 2 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.colorbar(sc,ticks=[0,45,90,135,180])
            
            figname = os.path.join(figfolder, '%s_zeronoise_PC1_vs_PC2.eps' % (layer_labels[ww1]))
            plt.savefig(figname, format='eps')

#            
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

            weights_reduced = allw[ww1][ww2]

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
                sc = plt.scatter(weights_reduced[myinds,pc2plot[0]], weights_reduced[myinds,pc2plot[1]],
                                 c=[clist[oo,:],],marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(weights_reduced[myinds,pc2plot[0]],0), np.mean(weights_reduced[myinds,pc2plot[1]],0)]
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

            weights_reduced = allw[ww1][ww2]

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
                sc = plt.scatter(weights_reduced[myinds,pc2plot[0]], weights_reduced[myinds,pc2plot[1]],
                                 c=[clist[oo,:],],marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(weights_reduced[myinds,pc2plot[0]],0), np.mean(weights_reduced[myinds,pc2plot[1]],0)]
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

            weights_reduced = allw[ww1][ww2]

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
                sc = plt.scatter(weights_reduced[myinds,pc2plot[0]], weights_reduced[myinds,pc2plot[1]],
                                 c=colors,
                                 vmin = 0,vmax=1, cmap=cm.get_cmap('plasma'),marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(weights_reduced[myinds,pc2plot[0]],0), np.mean(weights_reduced[myinds,pc2plot[1]],0)]
                
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

            weights_reduced = allw[ww1][ww2]

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
                sc = plt.scatter(weights_reduced[myinds,pc2plot[0]], weights_reduced[myinds,pc2plot[1]],
                                 c=colors,
                                 vmin = 0,vmax=360, cmap=cm.get_cmap('plasma'),marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(weights_reduced[myinds,pc2plot[0]],0), np.mean(weights_reduced[myinds,pc2plot[1]],0)]
                
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

            weights_reduced = allw[ww1][ww2]

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
            
                sc = plt.scatter(weights_reduced[myinds,pc2plot[0]], weights_reduced[myinds,pc2plot[1]],
                                 c=colors,
                                 vmin = np.min(sf_vals)-0.01,vmax=np.max(sf_vals)+0.01, cmap=cm.get_cmap('plasma'),marker=mark)

                h.append(sc)
                allpts[oo,:] = [np.mean(weights_reduced[myinds,pc2plot[0]],0), np.mean(weights_reduced[myinds,pc2plot[1]],0)]
                
            ax = plt.gca()               
            plt.title('PC %d versus %d, %s-%s, noise=%.2f\nColor=Spatial Frequency' % (pc2plot[0],pc2plot[1],layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC%d' %pc2plot[0])
            plt.ylabel('PC%d' % pc2plot[1])
#            plt.figlegend(h, ori2plot)

            plt.plot(allpts[:,0],allpts[:,1],'k')
            plt.colorbar(ticks =np.linspace(np.min(sf_vals),np.max(sf_vals), 1))
                
#%% PCA , plotting pts by orientation (binned)
            
plt.close('all')
layers2plot = np.arange(6,7)
timepts2plot = np.arange(0,1)
noiselevels2plot = [0]

clist = cm.plasma(np.linspace(0,1,12))

for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
            
#            pca = decomposition.PCA(n_components = 4)
#            
#            weights_reduced = pca.fit_transform(allw[ww1][ww2])
            weights_reduced = allw[ww1][ww2]
            nBins = int(12)
            nPerBin = int(180/nBins)
            binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
            
            legend_labs = [];
            for bb in range(nBins):   
                
                myinds = np.where(np.logical_and(np.isin(orilist, binned_labs[bb,:]), noiselist==nn))[0]
                plt.figure(1)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],c=np.expand_dims(clist[bb],axis=0))
                plt.figure(2)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2],c=np.expand_dims(clist[bb],axis=0))
                plt.figure(3)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3],c=np.expand_dims(clist[bb],axis=0))
                legend_labs.append('%d - %d deg' % (orilist[myinds[0]], orilist[myinds[-1]]))
            
            plt.figure(1)
            ax = plt.gca()               
            plt.title('PC 2 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.legend(legend_labs, bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
            
            plt.figure(2)           
            ax = plt.gca()         
            plt.title('PC 3 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC3')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])
                                   
            plt.figure(3)            
            ax = plt.gca()           
            plt.title('PC4 versus 1, %s-%s, noise=%.2f' % (layer_labels[ww1], timepoint_labels[ww2], noise_levels[nn]))
            plt.xlabel('PC1')
            plt.ylabel('PC4')
            plt.legend(legend_labs,bbox_to_anchor = (1,1))
            box = ax.get_position()
            ax.set_position([box.x0 , box.y0,
                             box.width*0.80, box.height])            
#%% PCA across all noise levels, plotting pts by orientation
            
plt.close('all')
layers2plot = [2]
timepts2plot = [0]

clist = cm.plasma(np.linspace(0,1,12))

for ww1 in layers2plot:
    for ww2 in timepts2plot:
#        for ww3 in noiselevels2plot:
            
        pca = decomposition.PCA(n_components = 4)
        
        weights_reduced = pca.fit_transform(allw[ww1][ww2])
       
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = [];
        for bb in range(nBins):   
             
            myinds = np.where(np.isin(orilist, binned_labs[bb,:]))[0]
        #    print(myinds)
            plt.figure(1)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],c=np.expand_dims(clist[bb],axis=0))
            plt.figure(2)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2],c=np.expand_dims(clist[bb],axis=0))
            plt.figure(3)
            plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3],c=np.expand_dims(clist[bb],axis=0))
            
            legend_labs.append('%d - %d deg' % (orilist[myinds[0]], orilist[myinds[-1]]))
            
        plt.figure(1)
        plt.title('PC 2 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(legend_labs, bbox_to_anchor = (1,1))
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.80, box.height])
        
        plt.figure(2)        
        ax = plt.gca()
        plt.title('PC 3 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        plt.legend(legend_labs,bbox_to_anchor = (1,1))
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.80, box.height])
            
        plt.figure(3)
        ax = plt.gca()
        plt.title('PC4 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC4')
        plt.legend(legend_labs,bbox_to_anchor = (1,1))
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.80, box.height])
        

#%% PCA across all noise levels, plotting pts by SF/noise level
            
plt.close('all')
layers2plot = [2]
timepts2plot = [0]
tt=0

csteps=8
my_purples = np.expand_dims(cm.Purples(np.linspace(1,0,csteps)),2)
my_greens = np.expand_dims(cm.Greens(np.linspace(1,0,csteps)),2)
my_oranges = np.expand_dims(cm.Oranges(np.linspace(1,0,csteps)),2)
my_blues = np.expand_dims(cm.Blues(np.linspace(1,0,csteps)),2)

clist = np.concatenate((my_purples, my_greens, my_oranges, my_blues),2)



for ww1 in layers2plot:
    for ww2 in timepts2plot:
#        for ww3 in noiselevels2plot:
            
        pca = decomposition.PCA(n_components = 4)
        
        weights_reduced = pca.fit_transform(allw[ww1][ww2])
       
        nBins = int(12)
        nPerBin = int(180/nBins)
        binned_labs = np.reshape(np.arange(0,180,1), [nBins,nPerBin])
        
        legend_labs = [];
        for bb in range(4):
            for nn in range(nNoiseLevels):
             
                myinds = np.where(np.all([sflist==bb, typelist==tt, noiselist==nn],0))[0]
            #    print(myinds)
                plt.figure(1)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,1],c=np.expand_dims(clist[nn,:,bb],axis=0))
                plt.figure(2)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,2],c=np.expand_dims(clist[nn,:,bb],axis=0))
                plt.figure(3)
                plt.scatter(weights_reduced[myinds,0], weights_reduced[myinds,3],c=np.expand_dims(clist[nn,:,bb],axis=0))
                
                legend_labs.append('SF=%.2f,noise=%.2f' % (sf_vals[bb], noise_levels[nn]))
                
        plt.figure(1)
        plt.title('PC 2 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(legend_labs, bbox_to_anchor = (1,1))
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.70, box.height])
        
        plt.figure(2)        
        ax = plt.gca()
        plt.title('PC 3 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC3')
        plt.legend(legend_labs,bbox_to_anchor = (1,1))
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.70, box.height])
            
        plt.figure(3)
        ax = plt.gca()
        plt.title('PC4 versus 1, %s-%s' % (layer_labels[ww1], timepoint_labels[ww2]))
        plt.xlabel('PC1')
        plt.ylabel('PC4')
        plt.legend(legend_labs,bbox_to_anchor = (1,1))
        box = ax.get_position()
        ax.set_position([box.x0 , box.y0,
                         box.width*0.70, box.height])
        

#%% CORR MATRIX, each spatial freq and type separately
plt.close('all')
tick_spacing = 45

sf2plot = [3]
type2plot = [0]
layers2plot = [13]
timepts2plot = [0]
noiselevels2plot = [0,1,2]

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
     
            corrmat = np.corrcoef(allw[ww1][ww2])
           
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds = np.where(np.logical_and(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)), noiselist==nn))[0]
                    plt.figure()
                    plt.pcolormesh(corrmat[myinds,:][:,myinds])
                    plt.title('Correlations for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                     
                    plt.colorbar()
                    plt.xticks(np.arange(0,180*nPhase+1, nPhase*tick_spacing),np.arange(0,180+1,tick_spacing))
                    plt.yticks(np.arange(0,180*nPhase+1, nPhase*tick_spacing),np.arange(0,180+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')



#%% DIST MATRIX, each spatial freq and type separately
plt.set_cmap('plasma')
plt.close('all')
tick_spacing = 45

sf2plot = [3]
type2plot = [0]
layers2plot = [13]
timepts2plot = [0]
noiselevels2plot = [0,1,2]

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
 
            
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds = np.where(np.logical_and(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)), noiselist==nn))[0]
                    distmat = scipy.spatial.distance.pdist(allw[ww1][ww2][myinds,:], 'euclidean')
                    distmat = scipy.spatial.distance.squareform(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
#                    plt.clim([0,4000])
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,180*nPhase+1, nPhase*tick_spacing),np.arange(0,180+1,tick_spacing))
                    plt.yticks(np.arange(0,180*nPhase+1, nPhase*tick_spacing),np.arange(0,180+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


#%% An alternative way of plotting the discriminability (averaging in bins)
plt.set_cmap('plasma')
plt.close('all')
tick_spacing = 45

sf2plot = [3]
type2plot = [0]
layers2plot = [13]
timepts2plot = [0]
noiselevels2plot = [0,1,2]

un,ia = np.unique(orilist, return_inverse=True)
assert np.all(np.expand_dims(ia,1)==orilist)

plt.close('all')
for ww1 in layers2plot:
    for ww2 in timepts2plot:
        for nn in noiselevels2plot:
 
            
            for bb in sf2plot:
                for tt in type2plot:
                
                    myinds_bool = np.logical_and(np.logical_and(np.isin(sflist, bb),np.isin(typelist, tt)), noiselist==nn)
                    
                    distmat = np.zeros([np.size(un),np.size(un)])
                    for uu1 in np.arange(0,np.size(un)):
                        for uu2 in np.arange(uu1,np.size(un)):
                            
                            inds1 = np.where(np.logical_and(orilist==un[uu1], myinds_bool))[0]    
                            inds2 = np.where(np.logical_and(orilist==un[uu2], myinds_bool))[0]    
    
                            vals = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(np.concatenate((allw[ww1][ww2][inds1,:],  allw[ww1][ww2][inds2,:]),0)))      
            
                            
                            
                            distmat[uu1,uu2] = np.mean(vals)
                            distmat[uu2,uu1] = np.mean(vals)
                    
#                    distmat = scipy.spatial.distance.pdist(allw[ww1][nn][ww2][myinds,:], 'euclidean')
#                    distmat = scipy.spatial.distance.squareform(distmat)
            
                    
                    plt.figure()
                    plt.pcolormesh(distmat)
                    plt.clim([0,500])
                    plt.title('Euc. distances for %s layer, noise %.2f\n %s - %s, SF=%.2f' % (layer_labels[ww1],noise_levels[nn],timepoint_labels[ww2],stim_types[tt],sf_vals[bb]))
                      
                    plt.colorbar()
                    plt.xticks(np.arange(0,180, tick_spacing),np.arange(0,180+1,tick_spacing))
                    plt.yticks(np.arange(0,180, tick_spacing),np.arange(0,180+1,tick_spacing))
        
                    plt.xlabel('Grating 1 deg')
                    plt.ylabel('Grating 2 deg')


