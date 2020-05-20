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
import load_activations
import matplotlib.lines as mlines

# make a big color map - nSF x nNetworks x RBGA
cols_sf_1 = np.moveaxis(np.expand_dims(cm.Greys(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_2 = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_3 = np.moveaxis(np.expand_dims(cm.YlGn(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf_4 = np.moveaxis(np.expand_dims(cm.OrRd(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = np.concatenate((cols_sf_1[np.arange(2,8,1),:,:],cols_sf_2[np.arange(2,8,1),:,:],cols_sf_3[np.arange(2,8,1),:,:],cols_sf_4[np.arange(2,8,1),:,:]),axis=1)

# values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)
nDeltaVals = np.size(delta_vals)

#%% get the data ready to go...then can run any below cells independently.
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures','FisherInfoPop')

nSamples =4
#model='vgg16'
model='pixel'

param_str='params1'

training_strs=['pixels']
ckpt_strs=['00000']
legend_strs=['pixels - filtered images, cos window']
#dataset_str=['FiltNoiseCos']
dataset_str=['FiltIms12AllSFCos']
#dataset_str=['SquareGratings']
color_inds=[0]


#training_strs = ['scratch_imagenet_rot_0_stop_early']
#ckpt_strs=['00000']
#legend_strs = ['randomly initialized - tested cosine gratings']
#dataset_str = ['CosGratings']
#color_inds = [0]


#training_strs = ['pretrained','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos']
#ckpt_strs=['00000','400000','400000','400000']
#legend_strs = ['pre-trained - tested bandpass-filtered imagenet images',
#               'trained 0 cos - tested bandpass-filtered imagenet images',
#               'trained 22 cos - tested bandpass-filtered imagenet images',
#               'trained 45 cos - tested bandpass-filtered imagenet images']
#dataset_str = ['FiltIms2AllSFCos']
#color_inds=[1,1,2,3]

#training_strs = ['pretrained','scratch_imagenet_rot_0_stop_early']
#ckpt_strs=['00000','00000']
#legend_strs = ['pre-trained - tested bandpass-filtered imagenet images',
#               'randomly initialized - tested bandpass-filtered imagenet images']
#dataset_str = ['FiltIms2Cos']
#color_inds = [1,0]

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
  
all_fisher = []
#all_deriv_sq = []
#all_pooled_var = []

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
        
    if tr==0 and kk==0:
      info = load_activations.get_info(model,dataset)
      
    if 'pixel' in model:
      save_path = os.path.join(root,'code','fisher_info',model,dataset,'Fisher_info_pixels.npy')
    else:
      save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'Fisher_info_eval_at_ckpt_%s_all_units.npy'%(ckpt_num))
    print('loading from %s\n'%save_path)
    
    # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
    FI = np.load(save_path)
    
#    save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'Deriv_sq_eval_at_ckpt_%s_all_units.npy'%(ckpt_num))
#    print('loading from %s\n'%save_path)
#    
#    # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
#    d = np.load(save_path)
#    
#    save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'Pooled_var_eval_at_ckpt_%s_all_units.npy'%(ckpt_num))
#    print('loading from %s\n'%save_path)
    
    # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
    v = np.load(save_path)
   
    if kk==0:
      nLayers = info['nLayers']
      nSF = 6;
      sfrange=np.arange(0,nSF,1)
      nOri = np.shape(FI)[2]
      
      fisher = np.zeros([nSamples, nLayers, nSF, nOri, nDeltaVals])
#      deriv_sq = np.zeros([nSamples, nLayers, nSF, nOri, nDeltaVals])
#      pooled_var = np.zeros([nSamples, nLayers, nSF, nOri, nDeltaVals])
    
    fisher[kk,:,:,:,:] = FI;
#    deriv_sq[kk,:,:,:,:] = d;
#    pooled_var[kk,:,:,:,:] = v;

  all_fisher.append(fisher)
#  all_deriv_sq.append(deriv_sq)
#  all_pooled_var.append(pooled_var)
  
#%% more parameters of interest 
layer_labels = info['layer_labels']
nOri = info['nOri']
ori_axis = np.arange(0, nOri,1)

#% define the orientation bins of interest
# will use these below to calculate anisotropy index
#b = np.arange(22.5,nOri,45)
b = np.arange(22.5,nOri,90)  # baseline
t = np.arange(67.5,nOri,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
c = np.arange(0,nOri,90) # cardinals
o = np.arange(45,nOri,90)  # obliques
bin_size = 10

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

#%%  saving plots of single layers at a time
# plot average FISHER INFORMATION curves, overlay samples, broadband SFs (ONE NETWORK)
# one delta values at a time.
# params for figure saving
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[10,6]

tr=0
dd=3

plt.rcParams.update({'font.size': 20})
plt.close('all')

#layers2plot = np.arange(0,nLayers,1)
layers2plot = [0,6,12,18]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
sf = 0
plt.figure()
# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

    
    all_fish= np.squeeze(all_fisher[tr][:,layers2plot[ll],sf,:,dd])
            
    plt.subplot(npx,npy, ll+1)
     # reshape this to [nSamples x nOri]
    fish = np.reshape(all_fish,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    errfish = np.std(fish,0)
#    for nn in range(nSamples):
#      plt.plot(ori_axis,fish[nn,:],color=cols_sf[sf+nn,1,:])
 
    if tr==0 or tr==1 or tr==3:
      plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[2,tr,:],color=[0,0,0])
    else:
      plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[1,tr,:],color=[0,0,0])
    # finish up this subplot    
    plt.title('%s' % (layer_labels[layers2plot[ll]]))
#    if ww1==layers2plot[-1]:
    plt.xlabel('Orientation (deg)')
    
#        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
    plt.xticks(np.arange(0,181,45))
    
    for ii in np.arange(0,181,45):
        plt.axvline(ii,color='k')
        
    plt.yticks([])
    plt.ylabel('Fisher information (a.u.)')
    
    # finish up the entire plot   
    plt.title('%s\n%s' % (legend_strs[tr],layer_labels[layers2plot[ll]]))
figname = os.path.join(figfolder, '%s_FisherInfo.pdf' % (training_strs[tr]))
plt.savefig(figname, format='pdf',transparent=True)

#%% big plot of all layers together 
# plot average FISHER INFORMATION curves, overlay samples, broadband SFs (ONE NETWORK)
# one delta values at a time.

tr=0
dd=3

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()
blue_colors = cm.Blues(np.linspace(0,1,nSamples))

layers2plot = np.arange(0,nLayers,1)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
sf = 0
# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  
    all_fish= np.squeeze(all_fisher[tr][:,layers2plot[ll],sf,:,dd])
            
    plt.subplot(npx,npy, ll+1)
     # reshape this to [nSamples x nOri]
    fish = np.reshape(all_fish,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    errfish = np.std(fish,0)
#    for nn in range(nSamples):
#      plt.plot(ori_axis,fish[nn,:],color=cols_sf[sf+nn,1,:])
 
    plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[3,color_inds[tr],:],color=[0,0,0])

    # finish up this subplot    
    plt.title('%s' % (layer_labels[layers2plot[ll]]))
    if ll==np.size(layers2plot)-1:
       plt.xlabel('Orientation (deg)')
      
#       plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
       plt.xticks(np.arange(0,181,45))
    else:
       plt.yticks([])
       plt.xticks([])
    for ll in np.arange(0,181,45):
       plt.axvline(ll,color='k')
        
       
#    plt.ylabel('Fisher information (a.u.)')
    
# finish up the entire plot   
plt.suptitle('%s - %s\nFisher Information (delta=%d deg)\nBroadband SF' % (training_strs[tr],dataset_all, delta_vals[dd]))


#%% plot average FISHER INFORMATION curves, overlay SPATIAL FREQUENCIES (one network)
# one delta values at a time.

tr=0
dd=3

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
npx = np.ceil(np.sqrt(len(layers2plot)))
npy = np.ceil(len(layers2plot)/npx)
#sf2plot = np.arange(0,nSF,1);
sf2plot=[5]
# loop over layers, making a subplot for each
for ww1 in layers2plot:

  for sf in sf2plot:
  
    all_fish= np.squeeze(all_fisher[tr][:,ww1,sf,:,dd])
            
    plt.subplot(npx,npy,ww1+1)
     # reshape this to [nSamples x nOri]
    fish = np.reshape(all_fish,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    errfish = np.std(fish,0)
    plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[sf,1,:],color=[0,0,0])

#    plt.errorbar(ori_axis,meanfish,errfish,color=cols_sf[sf,1,:])
#    plt.plot(ori_axis,meanfish,color='k')
  # finish up this subplot    
  plt.title('%s' % (layer_labels[ww1]))
  if ww1==layers2plot[-1]:
      plt.xlabel('actual orientation of grating')
      plt.ylabel('Fisher information')
      plt.legend(['%.2f cpp'%sf_vals[sf] for sf in sf2plot])
      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([])
  for ll in np.arange(0,181,45):
      plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s - %s\nFisher Information (delta=%d deg)' % (training_strs[tr],dataset_all, delta_vals[dd]))

#%% plot average SQUARED DERIVATIVE (not div by variance yet) curves, overlay samples, broadband SFs (ONE NETWORK)
# one delta values at a time.

tr=1
dd=3

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf = 0
# loop over layers, making a subplot for each
for ww1 in layers2plot:

  
    all_vals= np.squeeze(all_deriv_sq[tr][:,ww1,sf,:,dd])
            
    plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
     # reshape this to [nSamples x nOri]
    vals = np.reshape(all_vals,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanvals = np.mean(vals,0)
    errvals = np.std(vals,0)
    for nn in range(nSamples):
      plt.plot(ori_axis,vals[nn,:],color=cols_sf[sf+nn,1,:])
 
    plt.errorbar(ori_axis,meanvals,errvals,color=cols_sf[3,0,:])

    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Deriv squared')
        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s\nDeriv squared (delta=%d deg)\nBroadband SF' % (legend_strs[tr], delta_vals[dd]))


#%% plot average POOLED VARIANCE curves, overlay samples, broadband SFs (ONE NETWORK)
# one delta values at a time.

tr=1
dd=3

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf = 0
# loop over layers, making a subplot for each
for ww1 in layers2plot:

  
    all_vals= np.squeeze(all_pooled_var[tr][:,ww1,sf,:,dd])
            
    plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
     # reshape this to [nSamples x nOri]
    vals = np.reshape(all_vals,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanvals = np.mean(vals,0)
    errvals = np.std(vals,0)
    for nn in range(nSamples):
      plt.plot(ori_axis,vals[nn,:],color=cols_sf[sf+nn,1,:])
 
    plt.errorbar(ori_axis,meanvals,errvals,color=cols_sf[3,0,:])

    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Response variance')
        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s\nResponse Variance (delta=%d deg)\nBroadband SF' % (legend_strs[tr], delta_vals[dd]))


#%% plot average FISHER INFORMATION curves, mean over samples, broadband SFs (ONE NETWORK)
# Overlay different delta values to compare shapes.


tr=1

dd2plot = np.arange(0,9,1)
deltacolors = cm.YlGn(np.linspace(0,1,np.size(dd2plot)))
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf = 0

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    for dd in range(np.size(dd2plot)):
      
        all_fish= np.squeeze(all_fisher[tr][:,ww1,sf,:,dd2plot[dd]])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
         # reshape this to [nSamples x nOri]
        fish = np.reshape(all_fish,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meanfish = np.mean(fish,0)
       
        plt.plot(ori_axis,meanfish, color=deltacolors[dd,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Fisher information')
        plt.legend(['Delta=%d deg'%delta_vals[dd2plot[dd]] for dd in range(np.size(dd2plot))])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s\nFisher Information\nBroadband SF' % (legend_strs[tr]))
  

#%%  saving plots of single layers at a time
# plot average FISHER INFORMATION curves, overlay NETWORKS, broadband SFs 
# one delta values at a time.
# params for figure saving
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[10,6]

tr2plot = [1,2,3]
dd=3

plt.rcParams.update({'font.size': 20})
plt.close('all')

#layers2plot = np.arange(0,nLayers,1)
layers2plot = [0,6,12,18]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
sf = 0
# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

    plt.figure()

    for tr in tr2plot:
      all_fish= np.squeeze(all_fisher[tr][:,layers2plot[ll],sf,:,dd])

      fish = np.reshape(all_fish,[nSamples,180],order='F')
     
      # get mean and std, plot errorbars.
      meanfish = np.mean(fish,0)
      errfish = np.std(fish,0)
      if tr==1 or tr==3:
        plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[2,tr,:],color=[0,0,0])
      else:
        plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[1,tr,:],color=[0,0,0])

    # finish up this subplot    
    plt.title('%s' % (layer_labels[layers2plot[ll]]))

    plt.xlabel('Orientation (deg)')
    
#        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
    plt.xticks(np.arange(0,181,45))
    
    for ii in np.arange(0,181,45):
        plt.axvline(ii,color='k')
        
    plt.yticks([])
    plt.ylabel('Fisher information (a.u.)')
    
    # finish up the entire plot   
    plt.title('%s\n%s' % (legend_strs[tr],layer_labels[layers2plot[ll]]))
    figname = os.path.join(figfolder, '%s_FisherInfo_%s.pdf' % ('CompareFI',layer_labels[layers2plot[ll]]))
    plt.savefig(figname, format='pdf',transparent=True)


#%% big plot of all layers
#plot average FISHER INFORMATION curves, overlay NETWORKS, broadband SFs
# one delta values at a time.
dd=3
tr2plot = np.arange(0,nTrainingSchemes,1)

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf = 0
handles = []
# loop over layers, making a subplot for each
for ww1 in layers2plot:

  for tr in range(nTrainingSchemes):
  
    all_fish= np.squeeze(all_fisher[tr][:,ww1,sf,:,dd])
            
    plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
     # reshape this to [nSamples x nOri]
    fish = np.reshape(all_fish,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    errfish = np.std(fish,0)
    
    plt.errorbar(ori_axis,meanfish,errfish,color=cols_sf[3,tr,:])

  # finish up this subplot    
  plt.title('%s' % (layer_labels[ww1]))
  if ww1==layers2plot[-1]:
      plt.xlabel('actual orientation of grating')
      plt.ylabel('Fisher information')
      plt.legend(legend_strs)
      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([])
  for ll in np.arange(0,181,45):
      plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('Fisher Information (delta=%d deg)\nBroadband SF' % (delta_vals[dd]))


#%% plot Cardinal (V+H) anisotropy from 5-deg steps, broadband SF
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[10,6]

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr2plot = [0,1]
# loop over network training schemes (upright versus rot images etc)
for tr in tr2plot:
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      all_fish= np.squeeze(all_fisher[tr][kk,ww1,sf,:,dd])
    
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[card_inds]
      
      # final value for this layer: difference divided by sum 
      aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
    
  # put the line for this spatial frequency onto the plot      
  vals = np.mean(aniso_vals,0)
  errvals = np.std(aniso_vals,0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,tr,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,tr,:])


#  finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

#plt.legend(['%s'%legend_strs[tr] for tr in tr2plot])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)

plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Anisotropy Index')

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\nV+H versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'Aniso_cardinals.pdf')
plt.savefig(figname, format='pdf',transparent=True)

#%% plot 45 deg anisotropy from 5-deg steps, broadband SF

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr2plot = [0,1]
# loop over network training schemes (upright versus rot images etc)
for tr in tr2plot:
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      all_fish= np.squeeze(all_fisher[tr][kk,ww1,sf,:,dd])
    
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[obl_inds]
      
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
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

#plt.legend(['%s'%legend_strs[tr] for tr in tr2plot])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)

plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Anisotropy Index')

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\n45-deg versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'Aniso_45deg.pdf')
plt.savefig(figname, format='pdf',transparent=True)
#%% plot 22 deg anisotropy from 5-deg steps, broadband SF

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr2plot = [1,2,3]
# loop over network training schemes (upright versus rot images etc)
for tr in tr2plot:
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      all_fish= np.squeeze(all_fisher[tr][kk,ww1,sf,:,dd])
    
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[twent_inds]
      
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
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

#plt.legend(['%s'%legend_strs[tr] for tr in tr2plot])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Anisotropy Index')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\n22-deg versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'Aniso_22deg.pdf')
plt.savefig(figname, format='pdf',transparent=True)

#%% plot each kind of anisotropy for one network at a time.

tr=3

peak_inds=[card_inds, twent_inds,obl_inds]

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
#tr2plot = [1,2,3]
# loop over network training schemes (upright versus rot images etc)
for pp in range(np.size(peak_inds)):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      all_fish= np.squeeze(all_fisher[tr][kk,ww1,sf,:,dd])
    
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[peak_inds[pp]]
      
      # final value for this layer: difference divided by sum 
      aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
    
  # put the line for this spatial frequency onto the plot      
  vals = np.mean(aniso_vals,0)
  errvals = np.std(aniso_vals,0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,pp+1,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,pp+1,:])


# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline'])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Anisotropy Index')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Each type of anisotropy\n%s'%training_strs[tr])  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'AnisoEachType_%s.pdf'%training_strs[tr])
plt.savefig(figname, format='pdf',transparent=True)