#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the tuning parameters of individual units in each network - after fitting 
curves to the orientation tuning functions.
Before this, run get_orient_tuning.py and analyze_orient_tuning_jitter.py to 
compute the tuning curves and fit their parameters.

"""

import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
import load_activations
from copy import deepcopy
import analyze_orient_tuning_jitter

von_mises_deg = analyze_orient_tuning_jitter.von_mises_deg
get_fwhm = analyze_orient_tuning_jitter.get_fwhm
get_r2 = analyze_orient_tuning_jitter.get_r2

#%% paths
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root,'figures','UnitTuning')

#%% define which network to load - uncomment one of these lines

#training_strs = ['scratch_imagenet_rot_0_stop_early']   # a randomly initialized, un-trained model
#training_strs = ['scratch_imagenet_rot_0_cos']  # model trained on upright images
#training_strs = ['scratch_imagenet_rot_22_cos']   # model trained on 22 deg rot iamges
training_strs = ['scratch_imagenet_rot_0_cos']   # model trained on 45 deg rot images
#training_strs = ['pretrained']   # a pre-trained model 

#%% define other basic parameters
nImageSets = 1
model='vgg16'
param_str='params1'
dataset_str=['CosGratings']

if 'pretrained' in training_strs[0]:
  init_nums=[0]
  ckpt_strs=['0']
  # which color to use - [0,1,2,3,4] are for [random, trained upright, trained 22 rot, trained 45 rot, pretrained]
  color_ind=4
elif 'stop_early' in training_strs[0]:
  init_nums=[0]
  ckpt_strs=['0']
  color_ind=0
else:
  init_nums=[0]
  ckpt_strs=['400000']  
  if '0' in training_strs[0]:
    color_ind=1
  elif '22' in training_strs[0]:
    color_ind=2
  elif '45' in training_strs[0]:
    color_ind=3
    
nInits = np.size(init_nums)
param_strs=[]
for ii in range(len(init_nums)):    
  if init_nums[ii]>0:
    param_strs.append(param_str+'_init%d'%init_nums[ii])
  else:
    param_strs.append(param_str)

nTrainingSchemes = 1

# when identifying well-fit units, what criteria to use?
r2_cutoff = 0.4;

sf_labels=['0.01 cpp', '0.02 cpp', '0.04 cpp','0.08 cpp','0.14 cpp','0.25 cpp']
nSF=6
#%% loop to load all the data (orientation tuning fit parameters for all units)
tr=0
training_str = training_strs[tr]
ckpt_num = ckpt_strs[tr]
dataset = dataset_str[tr]  
for ii in range(nInits):
  
  # path info  
  param_str = param_strs[ii]
  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset) 
 
  # get information about the images/network
  if ii==0:
     info = load_activations.get_info(model,dataset)
     nSF = np.size(np.unique(info['sflist']))
     nLayers = info['nLayers']      
     layer_labels = info['layer_labels']    
     nOri = info['nOri']
     ori_axis = np.arange(0, nOri,1)
         
     # initialize these arrays (will be across all init of the network)    
     coords_all = []    
     tfs_all = []  
     mean_tfs_all = []  

  coords = []
  tfs = []
  mean_tfs = []
  # loop over layers and load fit parameters
  for ll in range(nLayers):
    
    # load coordinates of each network unit (spatial position and channel number)
    # [nUnits x 3] where third dim is [H,W,C]
    file_name =os.path.join(save_path,'%s_coordsHWC_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    coords.append(np.load(file_name))
  
    # load fit r2 [nUnits x nSF x nImageSets] 
#    file_name =os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
#    print('loading from %s\n'%file_name)
#    tfs.append(np.load(file_name))
    
    # load fit r2 [nUnits x nSF x nImageSets] 
    file_name =os.path.join(save_path,'%s_mean_resp_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    mean_tfs.append(np.load(file_name))
    
  coords_all.append(coords)
  tfs_all.append(tfs)
  mean_tfs_all.append(mean_tfs)
  
#%% create color map
nsteps = 8
colors_all_1 = np.moveaxis(np.expand_dims(cm.Greys(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
colors_all_2 = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
colors_all_3 = np.moveaxis(np.expand_dims(cm.YlGn(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
colors_all_4 = np.moveaxis(np.expand_dims(cm.OrRd(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
colors_all = np.concatenate((colors_all_1[np.arange(2,nsteps,1),:,:],colors_all_2[np.arange(2,nsteps,1),:,:],colors_all_3[np.arange(2,nsteps,1),:,:],colors_all_4[np.arange(2,nsteps,1),:,:]),axis=1)

int_inds = [3,3,3,3]
colors_main = np.asarray([colors_all[int_inds[ii],ii,:] for ii in range(np.size(int_inds))])
colors_main = np.concatenate((colors_main, colors_all[5,1:2,:]),axis=0)
# plot the color map
#plt.figure();plt.imshow(np.expand_dims(colors_main,axis=0))
cols_sf = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = cols_sf[np.arange(2,8,1),:,:]
#%% plot average response to each orientation across entire layers
# big plot with all layers
plt.close('all')
plt.rcParams.update({'font.size': 10})
sf2plot=np.arange(0,6);
for sf in sf2plot:
  ii=0
  layers2plot=np.arange(0,nLayers)
  npx = np.ceil(np.sqrt(np.size(layers2plot)))
  npy = np.ceil(np.size(layers2plot)/npx)
  plt.figure()
  for ll in range(np.size(layers2plot)):
    plt.subplot(npx,npy, ll+1)
     
    data = mean_tfs_all[ii][ll][:,sf,:]
    data = np.mean(np.reshape(data, [np.shape(data)[0],180,2],order='F'),2)
    meanvals = np.mean(data,axis=0)
    sdvals=np.std(data,axis=0)
    
    for kk in range(np.shape(data)[0]):
      plt.plot(ori_axis, data[kk,:],color=cols_sf[sf,0,:])
#    plt.errorbar(ori_axis,meanvals,sdvals,color='k',ecolor=colors_main[color_ind,:])
   
    plt.title('%s'%(layer_labels[layers2plot[ll]])) 
    
    if ll==nLayers-1:
        plt.xlabel('Orientation (deg)')
        plt.ylabel('Avg response')
        plt.xticks(np.arange(0,nOri+1,45))
    else:
        plt.xticks([]) 
        
  
  plt.suptitle('%s\n%s - init %d\n%s-%s\nAverage response across all units'%(model,training_str,init_nums[ii],dataset,sf_labels[sf]));
  plt.rcParams['figure.figsize']=[18,10]
