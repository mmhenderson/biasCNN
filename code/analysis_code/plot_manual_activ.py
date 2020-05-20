#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:07:15 2020

@author: mmhender
"""

import numpy as np
import matplotlib.pyplot as plt
#import conv_ops
import os
#from PIL import Image
#import scipy
#from scipy import spatial
#from sklearn import decomposition
import load_activations
from copy import deepcopy
import classifiers_custom as classifiers
from matplotlib import cm
import matplotlib.lines as mlines


root ='/mnt/neurocube/local/serenceslab/maggie/biasCNN/'

#%% first get all the information about this image set
image_set = 'CosGratings'
seed = 298484  #random initializations of the weights
model = 'vgg16_simul'
training_str = 'random_normal_weights_%s'%seed
param_str = 'params1'
ckpt_num='00000'

reduced_folder = os.path.join(root,'activations',model,training_str,param_str,image_set,'eval_at_ckpt-0_reduced')
if not os.path.isdir(reduced_folder):
  os.makedirs(reduced_folder)

info = load_activations.get_info('vgg16', image_set)


# now only using the first 19 layers
layer_labels = ['conv1_1','conv1_2','pool1',
 'conv2_1','conv2_2','pool2',
 'conv3_1','conv3_2','conv3_3','pool3',
 'conv4_1','conv4_2','conv4_3','pool4',
 'conv5_1','conv5_2','conv5_3','pool5',
 'fc6']
 
nLayers = np.size(layer_labels)

# information abot the stimuli
exlist = info['exlist'] 
Orients = np.arange(0,180,1)
nOrients = np.size(Orients)
sf_vals = info['sf_vals']
nSF = np.size(sf_vals)
contrast_levels = info['contrast_levels']
contrastlist =info['contrastlist'] 

typelist = info['typelist'] 
orilist = info['orilist']
sflist = info['sflist']
phaselist = info['phaselist']

# treat the orientation space as a 0-360 space since we have to go around the 180 space twice to account for phase.      
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180
ori_axis = np.arange(0.5, 360,1)

nIms= np.size(orilist)

# how many batches to do this in?
num_batches = 96
batch_size = int(nIms/num_batches)
which_batch = np.repeat(range(num_batches),batch_size)

#%% LOAD THE ACTIVATION PATTERNS FOR THESE IMS 

weight_path = reduced_folder
n_comp = []
allw = []   
allvarexpl = []
for ll in range(np.size(layer_labels)):

    file = os.path.join(weight_path, 'allStimsReducedWts_%s.npy' % layer_labels[ll])
    if os.path.isfile(file):
      w1 = np.load(file)
    else:
      print('missing file for layer %s\n'%layer_labels[ll])
      w1 = []
    w2 = []

    allw.append([w1,w2])
    n_comp.append(np.shape(w1)[1])
      
#%% get discriminability curve for this image set and layer
discrim = np.zeros([nLayers, nSF, 360])
for ll in range(nLayers):
  
  w = allw[ll][0]
    
  for sf in range(nSF):
          
      inds = np.where(sflist==sf)[0]
    
      ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
        
      discrim[ll,sf,:] = np.squeeze(disc)


#%% plot average discriminability curves, overlay spatial frequencies
      
# make a big color map - nSF x nNetworks x RBGA
cols_sf_1 = np.moveaxis(np.expand_dims(cm.Greys(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_2 = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_3 = np.moveaxis(np.expand_dims(cm.OrRd(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_4 = np.moveaxis(np.expand_dims(cm.YlGn(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = np.concatenate((cols_sf_1[np.arange(2,8,1),:,:],cols_sf_2[np.arange(2,8,1),:,:],cols_sf_3[np.arange(2,8,1),:,:],cols_sf_4[np.arange(2,8,1),:,:]),axis=1)

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()
nSamples=1

layers2plot = np.arange(0,nLayers,1)
#sf2plot = np.arange(0,6,1)
sf2plot =[5]
#sf2plot=[0]
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for sf in sf2plot:
      
        all_disc= np.squeeze(discrim[ww1,sf,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
    
        # average over samples to get what we will plot
        # first reshape to [nSamples x 180 x 2], folding the 360 axis in half        
        disc = np.reshape(all_disc,[nSamples,180,2],order='F')
        # now treat the two halves of the axis like an additional sample (now there are nSamples x 2 things to average over)
        disc = np.reshape(np.moveaxis(disc,2,1),[nSamples*2,180])
        
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)

        plt.errorbar(ori_axis[0:180],meandisc,errdisc,color=cols_sf[sf,1,:])
    
    # finish up this subplot    
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
        
# finish up the entire plot   
plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations' % (training_str))

#%% plot cardinal anisotropy

b = np.arange(22.5,360,90)  # baseline
t = np.arange(67.5,360,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
c = np.arange(0,360,90) # cardinals
o = np.arange(45,360,90)  # obliques
bin_size = 6

baseline_inds = []
for ii in range(np.size(b)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(360+b[ii]))<bin_size/2))[0])
  baseline_inds=np.append(baseline_inds,inds)
baseline_inds = np.uint64(baseline_inds)
            
card_inds = []
for ii in range(np.size(c)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-c[ii])<bin_size/2, np.abs(ori_axis-(360+c[ii]))<bin_size/2))[0])
  card_inds=np.append(card_inds,inds)
card_inds = np.uint64(card_inds)
   
obl_inds = []
for ii in range(np.size(o)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-o[ii])<bin_size/2, np.abs(ori_axis-(360+o[ii]))<bin_size/2))[0])
  obl_inds=np.append(obl_inds,inds)
obl_inds = np.uint64(obl_inds)
 
twent_inds = []
for ii in range(np.size(t)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-t[ii])<bin_size/2, np.abs(ori_axis-(360+t[ii]))<bin_size/2))[0])
  twent_inds=np.append(twent_inds,inds)
twent_inds = np.uint64(twent_inds)
 

tr=0
plt.rcParams.update({'font.size': 16})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(111)
layers2plot = np.arange(0,nLayers,1)

sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]
     
handles = []   

for sf in sf2plot:
  
    # matrix to store anisotropy index for each layer    
    aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
    
    # loop over network layers
    for ww1 in range(np.size(layers2plot)):
     
      # looping here over "samples"
      for kk in range(nSamples):

        disc = np.squeeze(discrim[layers2plot[ww1],sf,:])
        
        # take the bins of interest to get anisotropy
        base_discrim=  disc[baseline_inds]
        peak_discrim = disc[card_inds]
        
        # final value for this layer: difference divided by sum 
        aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
    # put the line for this spatial frequency onto the plot      
    vals = np.mean(aniso_vals,0)
    errvals = np.std(aniso_vals,0)
    myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,1,:])
    ax.add_line(myline)   
    handles.append(myline)
    plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[sf,1,:])

# finish up the whole plot    
ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]

#plt.legend(handles,['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (training_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)

 #%% plot oblique anisotropy

tr=0
plt.rcParams.update({'font.size': 16})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(111)
layers2plot = np.arange(0,nLayers,1)

sf2plot = [0,1,2,3,4,5] # spat freq
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]
     
handles = []   

for sf in sf2plot:
  
    # matrix to store anisotropy index for each layer    
    aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
    
    # loop over network layers
    for ww1 in range(np.size(layers2plot)):
     
      # looping here over "samples"
      for kk in range(nSamples):

        disc = np.squeeze(discrim[layers2plot[ww1],sf,:])
        
        # take the bins of interest to get anisotropy
        base_discrim=  disc[baseline_inds]
        peak_discrim = disc[obl_inds]
        
        # final value for this layer: difference divided by sum 
        aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
    # put the line for this spatial frequency onto the plot      
    vals = np.mean(aniso_vals,0)
    errvals = np.std(aniso_vals,0)
    myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[sf,1,:])
    ax.add_line(myline)   
    handles.append(myline)
    plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[sf,1,:])

# finish up the whole plot    
ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]

#plt.legend(handles,['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nOblique anisotropy' % (training_str))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)
