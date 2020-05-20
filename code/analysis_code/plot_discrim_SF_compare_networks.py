#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import matplotlib.pyplot as plt
import os
import numpy as np
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

nSamples =4
model='vgg16'
param_str='params1'

training_strs = ['pretrained','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos']
ckpt_strs=['00000','400000','400000','400000']
legend_strs = ['pre-trained - tested bandpass-filtered imagenet images',
               'trained 0 cos - tested bandpass-filtered imagenet images',
               'trained 22 cos - tested bandpass-filtered imagenet images',
               'trained 45 cos - tested bandpass-filtered imagenet images']
dataset_str = ['FiltImsAllSFCos']


#training_strs = ['scratch_imagenet_rot_0_stop_early','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos'];
#ckpt_strs=['00000','400000','400000','400000']
#legend_strs=['untrained - tested on filtered imagenet images',
#             'trained 0 cos - tested on filtered imagenet images',
#             'trained 22 cos - tested on filtered imagenet images',
#             'trained 45 cos - tested on filtered imagenet images']
#dataset_str = ['FiltImsAllSFCos','FiltImsAllSFCos','FiltImsAllSFCos','FiltImsAllSFCos']

#training_strs = ['scratch_imagenet_rot_22','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_22_square']
#ckpt_strs = ['400000','400000','400000']
#legend_strs = ['trained 22 circ - tested on filtered imagenet images',
#               'trained 22 cos - tested on filtered imagenet images',
#               'trained 22 square - tested on filtered imagenet images']
#dataset_str = ['FiltImsAllSFCos','FiltImsAllSFCos','FiltImsAllSFCos']


nTrainingSchemes = np.size(training_strs)
if np.size(dataset_str)==1:
  dataset_str = np.tile(dataset_str,nTrainingSchemes)
  
all_discrim = []
all_discrim5 = []
all_discrim_binned = []
all_fisher5 = []
all_fisher2 = []

sf_vals = np.logspace(np.log10(0.02),np.log10(.4),6)*140/224

# load activations for each training scheme
for tr in range(nTrainingSchemes):
  training_str = training_strs[tr]
  ckpt_num = ckpt_strs[tr]
  dataset_all = dataset_str[tr]
  
  # different versions of the evaluation image set (samples)
  for kk in range(nSamples):
  
    if kk==0 and 'FiltIms' not in dataset_all:
      dataset = dataset_all
    elif 'FiltIms' in dataset_all:
      dataset = '%s_rand%d'%(dataset_all,kk+1)
    else:
      dataset = '%s%d'%(dataset_all,kk)
        
    d1, info = load_activations.load_discrim(model,dataset,training_str,param_str,ckpt_num,part_str='all_units')
    d2, info = load_activations.load_discrim_5degsteps(model,dataset,training_str,param_str,ckpt_num,part_str='all_units')
    d3, info = load_activations.load_discrim_binned(model,dataset,training_str,param_str,ckpt_num,part_str='all_units')
    d4, info = load_activations.load_fisher5(model,dataset,training_str,param_str,ckpt_num,part_str='all_units')
    d5, info = load_activations.load_fisher2(model,dataset,training_str,param_str,ckpt_num,part_str='all_units')
   
    if kk==0:
      nLayers = info['nLayers']
      nSF = 6;
      sfrange=np.arange(0,nSF,1)
      nOri = np.shape(d1)[2]
      discrim = np.zeros([nSamples, nLayers, nSF, nOri])
      discrim5 = np.zeros([nSamples, nLayers, nSF, nOri])
      discrim_not_normalized = np.zeros([nSamples, nLayers, nSF,nOri])
      discrim_binned = np.zeros([nSamples, nLayers, nSF, nOri])
      fisher_delta5 = np.zeros([nSamples, nLayers, nSF, nOri])
      fisher_delta2 = np.zeros([nSamples, nLayers, nSF, nOri])
      
    for sfi in range(np.shape(d1)[1]):
      discrim[kk,:,sfrange[sfi],:] = d1[:,sfi,:]      
      discrim5[kk,:,sfrange[sfi],:] = d2[:,sfi,:] 
      discrim_binned[kk,:,sfrange[sfi],:] = d3[:,sfi,:] 
      fisher_delta5[kk,:,sfrange[sfi],:] = d4[:,sfi,:] 
      fisher_delta2[kk,:,sfrange[sfi],:] = d5[:,sfi,:] 
      
  all_discrim.append(discrim)
  all_discrim5.append(discrim5)
  all_discrim_binned.append(discrim_binned)
  all_fisher5.append(fisher_delta5)
  all_fisher2.append(fisher_delta2)
  
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


#%% plot Cardinal (V+H) anisotropy W single samples, overlay networks, one subplot per spatial freq
plt.rcParams.update({'font.size': 12})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf2plot=[0,1,2,3,4,5]
#sf2plot=[4]

# loop over SF, make one plot for each
for sf in sf2plot:

    if len(sf2plot)>1:
      ax=fig.add_subplot(2,3,sf+1)
    else:
      ax=fig.add_subplot(1,1,1)
    handles = []
    
    # loop over network training schemes (upright versus rot images etc)
    for tr in range(nTrainingSchemes):
      
      # matrix to store anisotropy index for each layer    
      aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
      
      # loop over network layers
      for ww1 in range(np.size(layers2plot)):

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
      if sf<4:
        plt.ylabel('Normalized Euclidean Distance difference')
    
# finish up the entire plot
plt.suptitle('Cardinals versus baseline')  
fig.set_size_inches(18,8)



#%% plot Cardinal (V+H) anisotropy from 5-deg steps, overlay networks, one subplot per spatial freq
plt.rcParams.update({'font.size': 12})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf2plot=[0,1,2,3,4,5]
#sf2plot=[4]

# loop over SF, make one plot for each
for sf in sf2plot:

    if len(sf2plot)>1:
      ax=fig.add_subplot(2,3,sf+1)
    else:
      ax=fig.add_subplot(1,1,1)
    handles = []
    
    # loop over network training schemes (upright versus rot images etc)
    for tr in range(nTrainingSchemes):
      
      # matrix to store anisotropy index for each layer    
      aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
      
      # loop over network layers
      for ww1 in range(np.size(layers2plot)):

        for kk in range(nSamples):
  
          disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
       
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
      if sf<4:
        plt.ylabel('Normalized Euclidean Distance difference')
    
# finish up the entire plot
plt.suptitle('Cardinals versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(18,8)


#%% plot Cardinal (V+H) anisotropy from 5-deg steps, broadband SF

plt.rcParams.update({'font.size': 12})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf=0

# loop over network training schemes (upright versus rot images etc)
for tr in range(nTrainingSchemes):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
   
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

plt.legend(legend_strs)

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)

plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.ylabel('Normalized Euclidean Distance difference')

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\nCardinals versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)


#%% plot Oblique (45+135) anisotropy W single samples, overlay networks, one subplot per spatial freq
plt.rcParams.update({'font.size': 12})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf2plot=[0,1,2,3,4,5]
#sf2plot=[0]
# loop over SF, make one plot for each
for sf in sf2plot:
      
    if len(sf2plot)>1:
      ax=fig.add_subplot(2,3,sf+1)
    else:
      ax=fig.add_subplot(1,1,1)
    handles = []
 

    # loop over network training schemes (upright versus rot images etc)
    for tr in range(nTrainingSchemes):
      
      # matrix to store anisotropy index for each layer    
      aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
      
      # loop over network layers
      for ww1 in range(np.size(layers2plot)):

        for kk in range(nSamples):
  
          disc = np.squeeze(all_discrim[tr][kk,layers2plot[ww1],sf,:])
       
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
      if sf<4:
        plt.ylabel('Normalized Euclidean Distance difference')
    
# finish up the entire plot
plt.suptitle('Obliques versus baseline')  
fig.set_size_inches(18,8)
                    
 #%% plot Oblique (45+135) anisotropy (5 deg steps), overlay networks, one subplot per spatial freq
plt.rcParams.update({'font.size': 12})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf2plot=[0,1,2,3,4,5]
#sf2plot=[0]
# loop over SF, make one plot for each
for sf in sf2plot:
      
    if len(sf2plot)>1:
      ax=fig.add_subplot(2,3,sf+1)
    else:
      ax=fig.add_subplot(1,1,1)
    handles = []
 

    # loop over network training schemes (upright versus rot images etc)
    for tr in range(nTrainingSchemes):
      
      # matrix to store anisotropy index for each layer    
      aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
      
      # loop over network layers
      for ww1 in range(np.size(layers2plot)):

        for kk in range(nSamples):
  
          disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
       
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
      if sf<4:
        plt.ylabel('Normalized Euclidean Distance difference')
    
# finish up the entire plot
plt.suptitle('Obliques versus baseline (5-deg steps)')  
fig.set_size_inches(18,8)
         
#%% plot Oblique anisotropy from 5-deg steps, broadband SF

plt.rcParams.update({'font.size': 12})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf=0

# loop over network training schemes (upright versus rot images etc)
for tr in range(nTrainingSchemes):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
   
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

plt.legend(legend_strs)

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)

plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.ylabel('Normalized Euclidean Distance difference')

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\nObliques versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)


         
#%% plot 22-deg anisotropy from 5-deg steps, broadband SF

plt.rcParams.update({'font.size': 12})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf=0

# loop over network training schemes (upright versus rot images etc)
for tr in range(nTrainingSchemes):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
   
      # take the bins of interest to get anisotropy
      base_discrim=  disc[baseline_inds]
      peak_discrim = disc[twent_inds]
      
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

plt.legend(legend_strs)

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)

plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.ylabel('Normalized Euclidean Distance difference')

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\n22 deg versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)


#%% plot Card-Obliques from 5-deg steps, broadband SF

plt.rcParams.update({'font.size': 12})
plt.close('all')
fig=plt.figure()
 
layers2plot = np.arange(0,nLayers,1)
sf=0

# loop over network training schemes (upright versus rot images etc)
for tr in range(nTrainingSchemes):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
   
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

plt.legend(legend_strs)

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)

plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.ylabel('Normalized Euclidean Distance difference')

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\nCard-Obliques (from 5 deg step discrim)')  
fig.set_size_inches(10,7)
            
#%% plot cardinal versus baseline, one training scheme at a time
tr=2
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
plt.title('%s\nCardinal anisotropy' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)

                    
#%% plot cardinal versus baseline (5-deg steps), one training scheme at a time
tr=3
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

        disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
        
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
plt.title('%s\nCardinal anisotropy (from 5-deg steps)' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)

#%% plot oblique versus baseline, one training scheme at a time
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
plt.title('%s\nOblique anisotropy' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)


#%% plot oblique versus baseline, one training scheme at a time (5-deg steps)
tr=3

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

        disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
        
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
plt.title('%s\nOblique anisotropy (5-deg steps)' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)



#%% plot card versus obliques, one training scheme at a time (5-deg steps)
tr=3

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

        disc = np.squeeze(all_discrim5[tr][kk,layers2plot[ww1],sf,:])
        
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

#plt.legend(handles,['sf=%.2f'%sf_vals[sf] for sf in sf2plot])
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.title('%s\nCardinal-Oblique (5-deg steps)' % (legend_strs[tr]))
plt.ylabel('Normalized Euclidean Distance difference')
plt.xlabel('Layer number')

fig.set_size_inches(10,8)
# want to move the fig panel up a bit so labels don't get cut off...
# current bbox [left, bottom, width, height]
curr_pos = np.asarray(ax.get_position().bounds)
new_pos = curr_pos + [0, 0.2, 0, -0.2]
ax.set_position(new_pos)

#%% plot average discriminability curves, overlay spatial frequencies (ONE NETWORK)
tr=0
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
#sf2plot = np.arange(0,6,1)
sf2plot =[0]
#sf2plot=[0]
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for sf in range(np.size(sf2plot)):
      
        all_disc= np.squeeze(all_discrim[tr][:,ww1,sf,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
                    
         # reshape this to [nSamples x nOri]
        disc = np.reshape(all_disc,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)

        plt.errorbar(ori_axis,meandisc,errdisc,color=cols_sf[sf,1,:])
    
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
  
#%% plot average 5-deg step discriminability curves, overlay spatial frequencies (ONE NETWORK)
tr=0
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0,1,2,3,4,5]
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for sf in range(np.size(sf2plot)):
      
        all_disc= np.squeeze(all_discrim5[tr][:,ww1,sf,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
         # reshape this to [nSamples x nOri]
        disc = np.reshape(all_disc,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)
        plt.plot(ori_axis,disc[0,:],color=cols_sf[sf,1,:])
#        plt.plot(ori_axis,disc[1,:],color=cols_sf[sf+1,1,:])
        
#        plt.errorbar(ori_axis,meandisc,errdisc,color=cols_sf[sf,1,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Normalized euclidean dist')
        plt.legend(legendlabs)
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s\nNormalized Euclidean dist (5 deg steps)' % (legend_strs[tr]))

#%% plot average 5-deg step discriminability curves, overlay samples, broadband SFs (ONE NETWORK)
tr=3
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0]

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for sf in range(np.size(sf2plot)):
      
        all_disc= np.squeeze(all_discrim5[tr][:,ww1,sf,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
         # reshape this to [nSamples x nOri]
        disc = np.reshape(all_disc,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)
        for nn in range(nSamples):
          plt.plot(ori_axis,disc[nn,:],color=cols_sf[sf+nn,1,:])
   
        plt.errorbar(ori_axis,meandisc,errdisc,color=cols_sf[3,0,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Normalized euclidean dist')
        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s\nNormalized Euclidean dist (5 deg steps)' % (legend_strs[tr]))

#%% plot average 5-deg FISHER INFORMATION curves, overlay samples, broadband SFs (ONE NETWORK)
tr=0
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf2plot = [0]

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for sf in range(np.size(sf2plot)):
      
        all_disc= np.squeeze(all_fisher5[tr][:,ww1,sf,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
         # reshape this to [nSamples x nOri]
        disc = np.reshape(all_disc,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)
        for nn in range(nSamples):
          plt.plot(ori_axis,disc[nn,:],color=cols_sf[sf+nn,1,:])
   
        plt.errorbar(ori_axis,meandisc,errdisc,color=cols_sf[3,0,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Fisher information')
        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s\nFisher Information (5 deg steps)' % (legend_strs[tr]))



#%% plot average discriminability (5-step out) curves, overlay networks (one SF)
tr2plot=[0,1]
sf =3;
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for tr in tr2plot:
      
        all_disc= np.squeeze(all_discrim5[tr][:,ww1,sf,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
    
                     
        # reshape this to [nSamples x nOri]
        disc = np.reshape(all_disc,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)

        plt.errorbar(ori_axis,meandisc,errdisc,color=cols_sf[sf,tr,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(['%s'%legend_strs[tr] for tr in tr2plot])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%0.2f cpp\nDiscriminability (5-deg-steps) between pairs of orientations' %sf_vals[sf])
  


#%% plot average discriminability (5-step out) curves, broadband SF (overlay networks)
tr2plot=[0,1,2,3]
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for tr in tr2plot:
      
        all_disc= np.squeeze(all_discrim5[tr][:,ww1,0,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
    
                     
        # reshape this to [nSamples x nOri]
        disc = np.reshape(all_disc,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)

        plt.errorbar(ori_axis,meandisc,errdisc,color=cols_sf[4,tr,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(['%s'%legend_strs[tr] for tr in tr2plot])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('Broadband SF stimuli\nDiscriminability (5-deg-steps) between pairs of orientations')
  


#%% plot average FISHER INFORMATION (5-step out) curves, broadband SF (overlay networks)
tr2plot=[0,1,2,3]
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for tr in tr2plot:
      
        all_disc= np.squeeze(all_fisher5[tr][:,ww1,0,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
    
                     
        # reshape this to [nSamples x nOri]
        disc = np.reshape(all_disc,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)

        plt.errorbar(ori_axis,meandisc,errdisc,color=cols_sf[4,tr,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Fisher Information')
        plt.legend(['%s'%legend_strs[tr] for tr in tr2plot])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('Broadband SF stimuli\nFisher information (5-deg-steps)')
  
#%% plot average discriminability curves, overlay networks (one SF)
tr2plot=[0,1,2]
sf = 4;
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
#sf2plot = np.arange(0,6,1)
sf2plot=[4]
legendlabs = ['sf=%.2f'%(sf_vals[sf]) for sf in sf2plot]

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    # loop over SF, making a line for each
    for tr in tr2plot:
      
        all_disc= np.squeeze(all_discrim[tr][:,ww1,sf,:])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
                  
        # reshape this to [nSamples x nOri]
        disc = np.reshape(all_disc,[nSamples,180],order='F')
        
        # get mean and std, plot errorbars.
        meandisc = np.mean(disc,0)
        errdisc = np.std(disc,0)

        plt.errorbar(ori_axis,meandisc,errdisc,color=cols_sf[sf,tr,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('discriminability (std. euc dist)')
        plt.legend(['%s'%legend_strs[tr] for tr in tr2plot])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%0.2f cpp\nDiscriminability (std. euc distance) between pairs of orientations' %sf_vals[sf])
  