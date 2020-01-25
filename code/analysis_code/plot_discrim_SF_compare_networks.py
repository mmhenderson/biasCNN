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


#%% get the data ready to go...then can run any below cells independently.
root = '/usr/local/serenceslab/maggie/biasCNN/';

dataset_all = 'SpatFreqGratings'
#dataset_all = 'SquareGratings'

model='vgg16'

os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures')

import load_activations

import classifiers_custom as classifiers    

param_str='params1'

#%% load activations from whatever networks/training schemes are of interest here

#training_strs=['scratch_imagenet_rot_0_square','scratch_imagenet_rot_0_square','scratch_imagenet_rot_0_square'];
#ckpt_strs=['350000','400000','450000']
#legend_strs=['trained 0 square - 350K steps','trained 0 square - 400K steps','trained 0 square - 450K steps']

training_strs=['scratch_imagenet_rot_0_square','scratch_imagenet_rot_22_square','scratch_imagenet_rot_45_square'];
ckpt_strs=['350000','350000','350000']
legend_strs=['trained 0 square - 350K steps','trained 22 square - 350K steps','trained 45 square - 350K steps']

#training_strs=['scratch_imagenet_rot_0_square','scratch_imagenet_rot_22_square','scratch_imagenet_rot_45_square'];
#ckpt_strs=['400000','400000','400000']
#legend_strs=['trained 0 square - 400K steps','trained 22 square - 400K steps','trained 45 square - 400K steps']

#training_strs=['scratch_imagenet_rot_0_square','scratch_imagenet_rot_22_square','scratch_imagenet_rot_45_square'];
#ckpt_strs=['450000','450000','450000']
#legend_strs=['trained 0 square - 450K steps','trained 22 square - 450K steps','trained 45 square - 450K steps']


nTrainingSchemes = np.size(training_strs)

bigw = []


# load activations for each training scheme
for tr in range(nTrainingSchemes):
  
  training_str = training_strs[tr]
  ckpt_str = ckpt_strs[tr]
  
  # first, searching for all folders from the same model, evaluated on different datasets (the sets are very similar but have different noise instantiation)
  dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str))
  good = [ii for ii in range(np.size(dirs)) if dataset_all in dirs[ii]]

  model_name_2plot = model + '_' + training_str + '_' + param_str + '_' + dataset_all + '_avg_samples'
  
  allw = []
  for ii in good:
    
    # also searching for all evaluations at different timepts (nearby, all are around the optimal point)
    dirs2 = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dirs[ii]))
    nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in dirs2]
    
    # compare the first two characters
    good2 = [jj for jj in range(np.size(dirs2)) if 'reduced' in dirs2[jj] and not 'sep_edges' in dirs2[jj] and ckpt_str[0:2] in nums[jj][0:2]]
  
    for jj in good2:
      

      ckpt_num= dirs2[jj].split('_')[2][5:]
      this_allw, all_labs, allvarexpl, info = load_activations.load_activ(model, dirs[ii], training_str, param_str, ckpt_num)
      allw.append(this_allw)
      if ii==good[0] and jj==good2[0]:
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
  if tr==0:
    nSamples = np.shape(allw)[0]
  else:
    assert(np.shape(allw)[0]==nSamples)

  bigw.append(allw)

cc=0
nn=0

# treat the orientation space as a 0-360 space since we have to go around the 180 space twice to account for phase.
assert phase_vals==[0,180]
orilist_adj = deepcopy(orilist)
orilist_adj[phaselist==1] = orilist_adj[phaselist==1]+180

# make a big color map - nSF x nNetworks x RBGA
cols_sf_1 = np.moveaxis(np.expand_dims(cm.Greys(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_2 = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_3 = np.moveaxis(np.expand_dims(cm.OrRd(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = np.concatenate((cols_sf_1[np.arange(2,8,1),:,:],cols_sf_2[np.arange(2,8,1),:,:],cols_sf_3[np.arange(2,8,1),:,:]),axis=1)

#%% define the orientation bins of interest
# will use these below to calculate anisotropy index

# using half steps, since discriminability is always between pairs of orientations 1 degree apart
ori_axis = np.arange(0.5, 360,1)

#b = np.arange(22.5,360,45)  # baseline

b = np.arange(22.5,360,90)  # baseline
m = np.arange(67.5,360,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
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
 
middle_inds = []
for ii in range(np.size(m)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis-m[ii])<bin_size/2, np.abs(ori_axis-(360+m[ii]))<bin_size/2))[0])
  middle_inds=np.append(middle_inds,inds)
middle_inds = np.uint64(middle_inds)
 
#%% visualize the bins 
plt.figure();
plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),baseline_inds))
plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),card_inds))
plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),obl_inds))
plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),middle_inds))
plt.legend(['baseline','cardinals','obliques','22'])
plt.title('bins for getting anisotropy index')


#%% plot Cardinal (V+H) anisotropy W single samples, overlay networks, one subplot per spatial freq
plt.rcParams.update({'font.size': 10})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf2plot=[0,1,2,3,4,5]
#sf2plot=[0]
# loop over SF, make one plot for each
for sf in sf2plot:
      
    ax=fig.add_subplot(2,3,sf+1)
    handles = []
    
    # loop over network training schemes (upright versus rot images etc)
    for tr in range(nTrainingSchemes):
      
      # matrix to store anisotropy index for each layer    
      aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
      
      # loop over network layers
      for ww1 in range(np.size(layers2plot)):
  
        disc_vals = np.zeros([nSamples, 360])    
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):
  
          w = bigw[tr][kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]
  
          # these are each 360 long (go around twice because of phase)
          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
          
          # take the bins of interest to get anisotropy
          base_discrim=  disc[baseline_inds]
          peak_discrim = disc[card_inds]
          
          # final value for this layer: difference divided by sum 
          aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
        
      # put the line for this spatial frequency onto the plot      
      vals = np.mean(aniso_vals,0)
      errvals = np.std(aniso_vals,0)
      myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,tr,:])
      ax.add_line(myline)   
      handles.append(myline)
      plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,tr,:])


    # finish up this subplot 
    ylims = [-1,1]
    xlims = [-1, np.size(layers2plot)]
    
    if sf==sf2plot[-1]:
      plt.legend(handles,legend_strs)
    
    plt.plot(xlims, [0,0], 'k')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.title('SF=%.2f cpp' % (sf_vals[sf]))
   
    if sf>2:
      plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
      plt.ylabel('Normalized Euclidean Distance difference')
    
# finish up the entire plot
plt.suptitle('Cardinals versus baseline')  
fig.set_size_inches(18,8)

#%% plot Oblique (45+135) anisotropy W single samples, overlay networks, one subplot per spatial freq
plt.rcParams.update({'font.size': 10})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf2plot=[0,1,2,3,4,5]
#sf2plot=[0]
# loop over SF, make one plot for each
for sf in sf2plot:
      
    ax=fig.add_subplot(2,3,sf+1)
    handles = []
    
    # loop over network training schemes (upright versus rot images etc)
    for tr in range(nTrainingSchemes):
      
      # matrix to store anisotropy index for each layer    
      aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
      
      # loop over network layers
      for ww1 in range(np.size(layers2plot)):
  
        disc_vals = np.zeros([nSamples, 360])    
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):
  
          w = bigw[tr][kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]
  
          # these are each 360 long (go around twice because of phase)
          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
          
          # take the bins of interest to get anisotropy
          base_discrim=  disc[baseline_inds]
          peak_discrim = disc[obl_inds]
          
          # final value for this layer: difference divided by sum 
          aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
        
      # put the line for this spatial frequency onto the plot      
      vals = np.mean(aniso_vals,0)
      errvals = np.std(aniso_vals,0)
      myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,tr,:])
      ax.add_line(myline)   
      handles.append(myline)
      plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,tr,:])


    # finish up this subplot 
    ylims = [-1,1]
    xlims = [-1, np.size(layers2plot)]
    
    if sf==sf2plot[-1]:
      plt.legend(handles,legend_strs)
    
    plt.plot(xlims, [0,0], 'k')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.title('SF=%.2f cpp' % (sf_vals[sf]))
   
    if sf>2:
      plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
      plt.ylabel('Normalized Euclidean Distance difference')
    
# finish up the entire plot
plt.suptitle('Obliques versus baseline')  
fig.set_size_inches(18,8)
                    
                    
#%% plot 22-deg anisotropy W single samples, overlay networks, one subplot per spatial freq
plt.rcParams.update({'font.size': 10})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf2plot=[0,1,2,3,4,5]
#sf2plot=[0]
# loop over SF, make one plot for each
for sf in sf2plot:
      
    ax=fig.add_subplot(2,3,sf+1)
    handles = []
    
    # loop over network training schemes (upright versus rot images etc)
    for tr in range(nTrainingSchemes):
      
      # matrix to store anisotropy index for each layer    
      aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
      
      # loop over network layers
      for ww1 in range(np.size(layers2plot)):
  
        disc_vals = np.zeros([nSamples, 360])    
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):
  
          w = bigw[tr][kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]
  
          # these are each 360 long (go around twice because of phase)
          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
          
          # take the bins of interest to get anisotropy
          base_discrim=  disc[baseline_inds]
          peak_discrim = disc[middle_inds]
          
          # final value for this layer: difference divided by sum 
          aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
        
      # put the line for this spatial frequency onto the plot      
      vals = np.mean(aniso_vals,0)
      errvals = np.std(aniso_vals,0)
      myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,tr,:])
      ax.add_line(myline)   
      handles.append(myline)
      plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,tr,:])


    # finish up this subplot 
    ylims = [-1,1]
    xlims = [-1, np.size(layers2plot)]
    
    if sf==sf2plot[-1]:
      plt.legend(handles,legend_strs)
    
    plt.plot(xlims, [0,0], 'k')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.title('SF=%.2f cpp' % (sf_vals[sf]))
   
    if sf>2:
      plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
      plt.ylabel('Normalized Euclidean Distance difference')
    
# finish up the entire plot
plt.suptitle('22 versus baseline')  
fig.set_size_inches(18,8)

#%% plot Cardinal (0+90) MINUS Oblique (45+135) anisotropy W single samples, overlay networks, one subplot per spatial freq
plt.rcParams.update({'font.size': 10})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf2plot=[0,1,2,3,4,5]
#sf2plot=[0]
# loop over SF, make one plot for each
for sf in sf2plot:
      
    ax=fig.add_subplot(2,3,sf+1)
    handles = []
    
    # loop over network training schemes (upright versus rot images etc)
    for tr in range(nTrainingSchemes):
      
      # matrix to store anisotropy index for each layer    
      aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
      
      # loop over network layers
      for ww1 in range(np.size(layers2plot)):
  
        disc_vals = np.zeros([nSamples, 360])    
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):
  
          w = bigw[tr][kk][layers2plot[ww1]][0]
    
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]
  
          # these are each 360 long (go around twice because of phase)
          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
          
          # take the bins of interest to get anisotropy
          base_discrim=  disc[obl_inds]
          peak_discrim = disc[card_inds]
          
          # final value for this layer: difference divided by sum 
          aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
        
      # put the line for this spatial frequency onto the plot      
      vals = np.mean(aniso_vals,0)
      errvals = np.std(aniso_vals,0)
      myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,tr,:])
      ax.add_line(myline)   
      handles.append(myline)
      plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,tr,:])


    # finish up this subplot 
    ylims = [-1,1]
    xlims = [-1, np.size(layers2plot)]
    
    if sf==sf2plot[-1]:
      plt.legend(handles,legend_strs)
    
    plt.plot(xlims, [0,0], 'k')
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.title('SF=%.2f cpp' % (sf_vals[sf]))
   
    if sf>2:
      plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
      plt.ylabel('Normalized Euclidean Distance difference')
    
# finish up the entire plot
plt.suptitle('Cardinals versus obliques')  
fig.set_size_inches(18,8)

#%% plot cardinal versus baseline, one training scheme at a time
tr=1
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

      disc_vals = np.zeros([nSamples, 360])    
      # looping here over "samples", going to get discriminability function for each, then average to smooth it.
      for kk in range(nSamples):

        w = bigw[tr][kk][layers2plot[ww1]][0]
  
        inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

        # these are each 360 long (go around twice because of phase)
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
        
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

plt.legend(handles,['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal anisotropy' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)

#%% plot oblique versus baseline, one training scheme at a time
tr=1
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

      disc_vals = np.zeros([nSamples, 360])    
      # looping here over "samples", going to get discriminability function for each, then average to smooth it.
      for kk in range(nSamples):

        w = bigw[tr][kk][layers2plot[ww1]][0]
  
        inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

        # these are each 360 long (go around twice because of phase)
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
        
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

plt.legend(handles,['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nOblique anisotropy' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)

#%% plot card versus obliques, one training scheme at a time
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

      disc_vals = np.zeros([nSamples, 360])    
      # looping here over "samples", going to get discriminability function for each, then average to smooth it.
      for kk in range(nSamples):

        w = bigw[tr][kk][layers2plot[ww1]][0]
  
        inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

        # these are each 360 long (go around twice because of phase)
        ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
        
        # take the bins of interest to get anisotropy
        base_discrim=  disc[obl_inds]
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

plt.legend(handles,['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinals versus obliques' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)

#%% plot average discriminability curves, overlay spatial frequencies
tr=2

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = np.arange(0,6,1)
#sf2plot=[0]
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for sf in range(np.size(sf2plot)):
      
        all_disc = []
        
        # looping here over "samples", going to get discriminability function for each, then average to smooth it.
        for kk in range(nSamples):

          w = bigw[tr][kk][ww1][0]
    
          # which stimuli are of interest here?
          inds = np.where(np.all([sflist==sf2plot[sf],contrastlist==cc,noiselist==nn],axis=0))[0]

          # these are each 360 long (go around twice because of phase)
          ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
         
          all_disc.append(disc)
          
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
plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations' % (legend_strs[tr]))
  
#%% plot discriminability curves, overlay samples (one spatial freq only)
tr=2
sf=5

plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    all_disc = []
    
    # looping here over "samples", going to get discriminability function for each, then average to smooth it.
    for kk in range(nSamples):

      w = bigw[tr][kk][ww1][0]

      # find stimuli of interest to use
      inds = np.where(np.all([sflist==sf,contrastlist==cc,noiselist==nn],axis=0))[0]

      # these are each 360 long (go around twice because of phase)
      ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
     
      all_disc.append(disc)
        
    # average over samples to get what we will plot
    # first reshape to [nSamples x 180 x 2], folding the 360 axis in half        
    disc = np.reshape(all_disc,[nSamples,180,2],order='F')
    # now treat the two halves of the axis like an additional sample (now there are nSamples x 2 things to average over)
    disc = np.reshape(np.moveaxis(disc,2,1),[nSamples*2,180])
      
    # get mean and std, plot errorbars.
    meandisc = np.mean(disc,0)
     
    plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)

    # plot individual samples
    plt.plot(ori_axis[0:180],np.transpose(disc),color=cols_sf[sf,1,:])

    # plot the mean
    plt.plot(ori_axis[0:180],meandisc,color='k')
   
    # finish up this subplot
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

# finish up the entire plot   
plt.suptitle('%s\nDiscriminability (std. euc distance) between pairs of orientations\nSF %.2f cpp' % (legend_strs[tr],sf_vals[sf]))

#%% plot discriminability curves, overlay training schemes, one layer and SF only
tr=2
sf=5
ww1=15

plt.close('all')
plt.figure()

for tr in range(nTrainingSchemes):
  
  all_disc = []
  # looping here over "samples", going to get discriminability function for each, then average to smooth it.
  for kk in range(nSamples):
  
    w = bigw[tr][kk][ww1][0]
  
    # find stimuli of interest to use
    inds = np.where(np.all([sflist==sf,contrastlist==cc,noiselist==nn],axis=0))[0]
  
    # these are each 360 long (go around twice because of phase)
    ori_axis, disc = classifiers.get_discrim_func(w[inds,:],orilist_adj[inds])
   
    all_disc.append(disc)
      
  # average over samples to get what we will plot
  # first reshape to [nSamples x 180 x 2], folding the 360 axis in half        
  disc = np.reshape(all_disc,[nSamples,180,2],order='F')
  # now treat the two halves of the axis like an additional sample (now there are nSamples x 2 things to average over)
  disc = np.reshape(np.moveaxis(disc,2,1),[nSamples*2,180])
  
  # get mean and std, plot errorbars.
  meandisc = np.mean(disc,0)
  errdisc = np.std(disc,0)

  plt.errorbar(ori_axis[0:180],meandisc,errdisc,color=cols_sf[3,tr,:])

plt.legend(legend_strs)
# finish up the plot
if ww1==layers2plot[-1]:
    plt.xlabel('actual orientation of grating')
    plt.ylabel('discriminability (std. euc dist)')
#        plt.legend(legendlabs)
    plt.xticks(np.arange(0,181,45))
else:
    plt.xticks([])
for ll in np.arange(0,181,45):
    plt.axvline(ll,color='k')
   
plt.suptitle('%s\nSF %.2f cpp' % (layer_labels[ww1],sf_vals[sf]))
