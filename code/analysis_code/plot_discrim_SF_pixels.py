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
from matplotlib import cm
import matplotlib.lines as mlines
import load_activations

# make a big color map - nSF x nNetworks x RBGA
cols_sf_1 = np.moveaxis(np.expand_dims(cm.Greys(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_2 = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_3 = np.moveaxis(np.expand_dims(cm.OrRd(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_4 = np.moveaxis(np.expand_dims(cm.YlGn(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = np.concatenate((cols_sf_1[np.arange(2,8,1),:,:],cols_sf_2[np.arange(2,8,1),:,:],cols_sf_3[np.arange(2,8,1),:,:],cols_sf_4[np.arange(2,8,1),:,:]),axis=1)

#%% get the data ready to go...then can run any below cells independently.
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures')

dataset_all = 'FiltImsAllSFCos'
#dataset_all = 'CosGratings'
#dataset_all = 'CircGratings'
#dataset_all = 'SpatFreqGratings'
#dataset_all = 'SquareGratings'
nSamples = 5
model='pixel'

legend_strs=['pixels']
sf_vals = np.logspace(np.log10(0.02),np.log10(.4),6)*140/224

all_discrim = []
all_discrim5 = []
  
# different versions of the evaluation image set (samples)
for kk in range(nSamples):
  
  if 'FiltIms' in dataset_all:
    dataset = '%s_rand%d'%(dataset_all,kk+1)
  else:
    if kk==0:
      dataset = dataset_all
    else:
      dataset = '%s%d'%(dataset_all,kk)
        

  d1, info = load_activations.load_discrim(model,dataset)
  d2, info = load_activations.load_discrim_5degsteps(model,dataset)

  if kk==0:
    nLayers = info['nLayers']
    nSF = info['nSF']
    sfrange=np.arange(0,nSF,1)
    nOri = np.shape(d1)[2]
    discrim = np.zeros([nSamples, nLayers, nSF, nOri])
    discrim5 = np.zeros([nSamples, nLayers, nSF, nOri])
    
  discrim[kk,:,:,:] = d1
  discrim5[kk,:,:,:] = d2
   
all_discrim.append(discrim)
all_discrim5.append(discrim5)
#%% more parameters of interest 
layer_labels = info['layer_labels']
nOri = info['nOri']
ori_axis = np.arange(0.5, nOri,1)

#% define the orientation bins of interest
# will use these below to calculate anisotropy index
#b = np.arange(22.5,nOri,45)
b = np.arange(22.5,nOri,90)  # baseline
t = np.arange(67.5,nOri,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
c = np.arange(0,nOri,90) # cardinals
o = np.arange(45,nOri,90)  # obliques
bin_size = 6

baseline_inds = []
for ii in range(np.size(b)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-b[ii])<bin_size/2, np.abs(ori_axis-(nOri+b[ii]))<bin_size/2))[0])
  baseline_inds=np.append(baseline_inds,inds)
baseline_inds = np.uint64(baseline_inds)
            
card_inds = []
for ii in range(np.size(c)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-c[ii])<bin_size/2, np.abs(ori_axis-(nOri+c[ii]))<bin_size/2))[0])
  card_inds=np.append(card_inds,inds)
card_inds = np.uint64(card_inds)
   
obl_inds = []
for ii in range(np.size(o)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-o[ii])<bin_size/2, np.abs(ori_axis-(nOri+o[ii]))<bin_size/2))[0])
  obl_inds=np.append(obl_inds,inds)
obl_inds = np.uint64(obl_inds)
 
twent_inds = []
for ii in range(np.size(t)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-t[ii])<bin_size/2, np.abs(ori_axis-(nOri+t[ii]))<bin_size/2))[0])
  twent_inds=np.append(twent_inds,inds)
twent_inds = np.uint64(twent_inds)
 
#%% visualize the bins 
plt.figure();
plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),baseline_inds))
plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),card_inds))
plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),obl_inds))
plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),twent_inds))
plt.legend(['baseline','cardinals','obliques','22'])
plt.title('bins for getting anisotropy index')


#%% plot cardinal versus baseline, one training scheme at a time
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

        disc = np.squeeze(all_discrim[tr][kk,layers2plot[ww1],sf,:])
        
        # take the bins of interest to get anisotropy
        base_discrim=  disc[baseline_inds]
        peak_discrim = disc[card_inds]
        
        # final value for this layer: difference divided by sum 
        aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
    # put the line for this spatial frequency onto the plot      
    vals = np.mean(aniso_vals,0)
    errvals = np.std(aniso_vals,0)
    
    plt.scatter(sf_vals[sf],vals,color=cols_sf[sf,1,:])
#    myline = mlines.Line2D(sf_vals[sf],vals,color = cols_sf[sf,1,:])
#    ax.add_line(myline)   
#    handles.append(myline)
#    plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[sf,1,:])

# finish up the whole plot    
ylims = [-1,1]
xlims = [0, 0.3]

#plt.legend(handles,['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
#plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Spatial Frequency')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)
plt.title('Cardinal anisotropy in pixel representation')

#%% plot average discriminability curves, overlay spatial frequencies
tr=0
ww1=0
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

#layers2plot = np.arange(0,nLayers,1)

#sf2plot=[0]
if 'FiltIms' in dataset_all:
  sf2plot = [0]
  legendlabs = ['Broadband SF']
else:
  sf2plot = np.arange(0,6,1)
  legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

# loop over layers, making a subplot for each
#for ww1 in layers2plot:

    # loop over SF, making a line for each
for sf in range(np.size(sf2plot)):
  
    all_disc= np.squeeze(all_discrim[tr][:,ww1,sf,:])
            
    disc = np.reshape(all_disc,[nSamples,180],order='F')
       
    
    # get mean and std, plot errorbars.
    meandisc = np.mean(disc,0)
    errdisc = np.std(disc,0)
    if 'FiltIms' in dataset_all:
      plt.errorbar(ori_axis[0:180],meandisc,errdisc,color=cols_sf[3,0,:])
    else:
      plt.errorbar(ori_axis[0:180],meandisc,errdisc,color=cols_sf[sf,1,:])

# finish up this subplot    
#plt.title('%s' % (layer_labels[ww1]))
#if ww1==layers2plot[-1]:
plt.xlabel('actual orientation of grating')
plt.ylabel('discriminability (std. euc dist)')
plt.legend(legendlabs)
plt.xticks(np.arange(0,181,45))
#else:
#    plt.xticks([])
for ll in np.arange(0,181,45):
    plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.title('%s\n%s\nDiscriminability (1 deg steps) between pairs of orientations' % (dataset_all,legend_strs[tr]))
  
#%% plot average 5-step-out discriminability curves, overlay spatial frequencies
tr=0
ww1=0
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

#layers2plot = np.arange(0,nLayers,1)

#sf2plot=[0]
if 'FiltIms' in dataset_all:
  sf2plot = [0]
  legendlabs = ['Broadband SF']
else:
  sf2plot = np.arange(0,6,1)
  legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

# loop over layers, making a subplot for each
#for ww1 in layers2plot:

    # loop over SF, making a line for each
for sf in range(np.size(sf2plot)):
  
    all_disc= np.squeeze(all_discrim5[tr][:,ww1,sf,:])
            
    disc = np.reshape(all_disc,[nSamples,180],order='F')
       
    
    # get mean and std, plot errorbars.
    meandisc = np.mean(disc,0)
    errdisc = np.std(disc,0)
    if 'FiltIms' in dataset_all:
      plt.errorbar(ori_axis[0:180],meandisc,errdisc,color=cols_sf[3,0,:])
    else:
      plt.errorbar(ori_axis[0:180],meandisc,errdisc,color=cols_sf[sf,1,:])

# finish up this subplot    
#plt.title('%s' % (layer_labels[ww1]))
#if ww1==layers2plot[-1]:
plt.xlabel('actual orientation of grating')
plt.ylabel('discriminability (std. euc dist)')
plt.legend(legendlabs)
plt.xticks(np.arange(0,181,45))
#else:
#    plt.xticks([])
for ll in np.arange(0,181,45):
    plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.title('%s\n%s\nDiscriminability (5-deg steps) between pairs of orientations' % (dataset_all,legend_strs[tr]))

#%% plot average 5-step-out discriminability curves, overlay image sets (single SF)
tr=0
ww1=0
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

#layers2plot = np.arange(0,nLayers,1)
legendlabs = ['set %d'%(ss+1) for ss in range(nSamples)]
legendlabs = np.append(legendlabs,['mean'])
sf=0


all_disc= np.squeeze(all_discrim5[tr][:,ww1,sf,:])
        
disc = np.reshape(all_disc,[nSamples,180],order='F')
   
# get mean and std, plot errorbars.
meandisc = np.mean(disc,0)
errdisc = np.std(disc,0)

for kk in range(nSamples):
  
  plt.plot(ori_axis,disc[kk,:],color = cols_sf[sf+kk,1,:])

plt.plot(ori_axis,meandisc,color=cols_sf[5,0,:])
#plt.errorbar(ori_axis[0:180],meandisc,errdisc,color=cols_sf[5,0,:])

# finish up this subplot    
#plt.title('%s' % (layer_labels[ww1]))
#if ww1==layers2plot[-1]:
plt.xlabel('actual orientation of grating')
plt.ylabel('discriminability (std. euc dist)')
plt.legend(legendlabs)
plt.xticks(np.arange(0,181,45))
#else:
#    plt.xticks([])
for ll in np.arange(0,181,45):
    plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.title('%s\n%s\nDiscriminability (5-deg steps) between pairs of orientations' % (dataset_all,legend_strs[tr]))
    