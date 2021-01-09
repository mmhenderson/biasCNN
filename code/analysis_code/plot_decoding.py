#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Fisher information estimated at each network layer.
Plot Fisher information bias computed from FI.
Run get_fisher_info_full.py to compute FI. 

"""

import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
import load_activations
from copy import deepcopy
import statsmodels.stats.multitest
import scipy.stats
import matplotlib.lines as mlines
#%% paths
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures','FisherInfoPop')

#%% define parameters for what to load here

# loading all networks at once - 
# [random, trained upright images, trained 22 deg rot images, trained 45 deg rot images, pretrained]
#training_strs=['scratch_imagenet_rot_0_cos_stop_early','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos','pretrained']
training_strs=['pretrained']

ckpt_strs=['0']
nInits_list = [1]
color_inds=[1]

# define other basic parameters
nImageSets = 1
model='vgg16'
param_str='params1'
param_strs=[]
for ii in range(np.max(nInits_list)):    
  if ii>0:
    param_strs.append(param_str+'_init%d'%ii)
  else:
    param_strs.append(param_str)

dataset_str=['FiltIms14AllSFCos']
nTrainingSchemes = np.size(training_strs)

 # values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)
nDeltaVals = np.size(delta_vals)

sf_labels=['broadband SF']
nSF=1
sf=0

#%% load the data (Decoding accs)

# load activations for each training set of images (training schemes)
for tr in range(nTrainingSchemes):
  training_str = training_strs[tr]
  ckpt_num = ckpt_strs[tr]
  dataset_all = dataset_str[0]
  nInits = nInits_list[tr]
  
  # different initializations with same training set
  for ii in range(nInits):
 
    param_str=param_strs[ii]
  
    # different versions of the evaluation image set (samples)
    for kk in range(nImageSets):
           
      dataset = '%s_rand%d'%(dataset_all,kk+1)
       
      if ii==0 and kk==0:
        info = load_activations.get_info(model,dataset)
        layer_labels = info['layer_labels']
        nOri = info['nOri']
        ori_axis = np.arange(0, nOri,1)
        
      # find the exact number of the checkpoint 
      ckpt_dirs = os.listdir(os.path.join(root,'code','decoding',model,training_str,param_str,dataset))
      ckpt_dirs = [dd for dd in ckpt_dirs if 'eval_at_ckpt-%s'%ckpt_num[0:2] in dd and '_reduced' in dd]
      nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'_reduced')] for dir in ckpt_dirs]            
  
      save_path = os.path.join(root,'code','decoding',model,training_str,param_str,dataset,'eval_at_ckpt-%s_reduced'%nums[0],'Dec_acc_all.npy')
      print('loading from %s\n'%save_path)
      
      # Fisher info array is [nLayer x nSF x nOri x nDeltaValues] in size
      dec_acc = np.load(save_path)
      
      if kk==0 and tr==0 and ii==0:
        nLayers = info['nLayers']         
        nOri = np.shape(dec_acc)[2]      
        # initialize this ND array to store all Fisher info calculated values
        all_dec_acc = np.zeros([nTrainingSchemes, np.max(nInits_list), nImageSets, nLayers, nSF, nOri, nOri])
       
      all_dec_acc[tr,ii,kk,:,sf,:,:] = np.squeeze(dec_acc);
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


#%%  Plot full decoding matrices
layers2plot = np.arange(0,nLayers)

tr=0  # can change this value to plot the netorks with different training sets

if tr==4 or tr==0:
  init2plot = [0]
else:
  init2plot = [0,1,2,3]
  
sf=0

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]

plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  acc_all_init = np.zeros([len(init2plot),nOri, nOri])
  
  for ii in init2plot:
   
    # average over image sets
    acc_mat = np.mean(all_dec_acc[tr,ii,:,layers2plot[ll],sf,:,:],axis=0)
   
    acc_all_init[ii,:,:] = acc_mat
    
  acc_mat = np.mean(acc_all_init,0)
 
  plt.pcolormesh(acc_mat)
  
  # finish up this subplot
  plt.axis('square')    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==nLayers-1:
    plt.xlabel('Orientation 1 (deg)')
    plt.xlabel('Orientation 2 (deg)')
    plt.xticks(np.arange(0,181,45))
    plt.yticks(np.arange(0,181,45))
    plt.colorbar()
  else:
    plt.xticks([])
    plt.yticks([])
  plt.xlim([np.min(ori_axis),np.max(ori_axis)])
  plt.ylim([np.min(ori_axis),np.max(ori_axis)])
 
 
# finish up the entire plot   
plt.suptitle('%s\nSVM decoding' % (training_strs[tr]))
figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)



#%%  Plot decoding of a single orient versus other orients
layers2plot = np.arange(0,nLayers)

tr=0  # can change this value to plot the netorks with different training sets

if tr==4 or tr==0:
  init2plot = [0]
else:
  init2plot = [0,1,2,3]
  
sf=0
ori1 = 45

#ylims=[[0, 0.1],[0,0.1],[0,0.1],[0,0.7]]

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]

plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

ylims = [0.2,1.2]

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  acc_all_init = np.zeros([len(init2plot),nOri])
  
  for ii in init2plot:
   
    # average over image sets
    acc_mat = np.mean(all_dec_acc[tr,ii,:,layers2plot[ll],sf,:,:],axis=0)
    
    # get all comparisons of this orientation versus others...
    acc_vec1 = acc_mat[0:ori1,ori1] # first orients smaller than main, versus main
    acc_vec2 = acc_mat[ori1,ori1+1:] # then main versus orients larger than
    
    acc_vec = np.append(acc_vec1, np.nan)
    acc_vec= np.append(acc_vec, acc_vec2)
    
    # center orientation of interest in the middle to make easier to see pattern
    my_center=90
    acc_vec_centered = np.roll(acc_vec,my_center-ori1)
    ori_axis_centered = ori_axis-my_center
#    acc_vec_centered=acc_vec
#    ori_axis_centered=ori_axis
    
    acc_all_init[ii,:] = acc_vec_centered
    
  # get mean and std, plot errorbars.
  # Errorbars are over network initializations, if there is more than one. 
  # otherwise, no errorbars are plotted.
  meanacc = np.mean(acc_all_init,0)    
  erracc = np.std(acc_all_init,0)
 
  if len(init2plot)>1:
    plt.errorbar(ori_axis_centered,meanacc,erracc,ecolor=colors_main[color_inds[tr],:],color=[0,0,0])
  else:
    plt.plot(ori_axis_centered,meanacc,color=colors_main[color_inds[tr],:])
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==nLayers-1:
    plt.xlabel('difference from ori1 (deg)')
    plt.xticks(np.arange(-90,91,45))
  else:
    plt.xticks([])
  plt.xlim([np.min(ori_axis_centered),np.max(ori_axis_centered)+1])
  plt.ylabel('dec acc')
  
  plt.ylim(ylims)
  
  for xx in np.arange(-90,91,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  plt.axhline(0.5, color=[0.8, 0.8, 0.8])
# finish up the entire plot   
plt.suptitle('%s\npopulation-level SVM decoding: %d deg versus other orients' % (training_strs[tr],ori1))
figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)


#%%  Plot decoding of a single orient versus other orients
# overlay multiple central orients
layers2plot = np.arange(0,nLayers)

tr=0  # can change this value to plot the netorks with different training sets

if tr==4 or tr==0:
  init2plot = [0]
else:
  init2plot = [0,1,2,3]
  
sf=0
ori_focus = np.arange(0,41,1)
cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(ori_focus)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(ori_focus),1),:,:]

#ylims=[[0, 0.1],[0,0.1],[0,0.1],[0,0.7]]

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]

#plt.close('all')

fig=plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

ylims = [0.2,1.2]

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  ax=fig.add_subplot(npx,npy, ll+1)
  allh=[]
  for oo in range(len(ori_focus)):
    ori1 = ori_focus[oo]
    acc_all_init = np.zeros([len(init2plot),nOri])
    
    for ii in init2plot:
     
      # average over image sets
      acc_mat = np.mean(all_dec_acc[tr,ii,:,layers2plot[ll],sf,:,:],axis=0)
      
      # get all comparisons of this orientation versus others...
      acc_vec1 = acc_mat[0:ori1,ori1] # first orients smaller than main, versus main
      acc_vec2 = acc_mat[ori1,ori1+1:] # then main versus orients larger than
      
      acc_vec = np.append(acc_vec1, np.nan)
      acc_vec= np.append(acc_vec, acc_vec2)
      
      # center orientation of interest in the middle to make easier to see pattern
      my_center=90
      acc_vec_centered = np.roll(acc_vec,my_center-ori1)
      ori_axis_centered = ori_axis-my_center
#      acc_vec_centered=acc_vec
#      ori_axis_centered=ori_axis
      
      acc_all_init[ii,:] = acc_vec_centered
      
    # get mean and std, plot errorbars.
    # Errorbars are over network initializations, if there is more than one. 
    # otherwise, no errorbars are plotted.
    meanacc = np.mean(acc_all_init,0)    
    erracc = np.std(acc_all_init,0)
   
    if len(init2plot)>1:
      plt.errorbar(ori_axis_centered,meanacc,erracc,ecolor=cols_grad[oo,0,:],color=[0,0,0])
    else:
      plt.plot(ori_axis_centered,meanacc,color=cols_grad[oo,0,:])
      
    myline = mlines.Line2D(ori_axis_centered,meanacc,color=cols_grad[oo,0,:])
    ax.add_line(myline)   
    allh.append(myline)
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==nLayers-1:
#    plt.xlabel('orient2')
    plt.xlabel('diff from orient1')
    plt.xticks(np.arange(-90,91,45))
    
  else:
    plt.xticks([])
  plt.xlim([np.min(ori_axis_centered),np.max(ori_axis_centered)+1])
  plt.ylabel('dec acc')
  
  if ll==nLayers-6:
    plt.legend(allh,['orient1=%d deg'%oo for oo in ori_focus])
  plt.ylim(ylims)
  
  for xx in np.arange(-90,91,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  plt.axhline(0.5, color=[0.8, 0.8, 0.8])
# finish up the entire plot   
plt.suptitle('%s\npopulation-level SVM decoding: each orient versus other orients' % (training_strs[tr]))
figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)



#%%  Plot decoding on off diagonals - information abt fine discriminations.
layers2plot = np.arange(0,nLayers)

tr=0  # can change this value to plot the netorks with different training sets

if tr==4 or tr==0:
  init2plot = [0]
else:
  init2plot = [0,1,2,3]
  
sf=0
dd=1
#ylims=[[0, 0.1],[0,0.1],[0,0.1],[0,0.7]]

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]

plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

ylims = [0.2,1.2]

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  acc_all_init = np.zeros([len(init2plot),nOri])
  
  for ii in init2plot:
   
    # average over image sets
    acc_mat = np.mean(all_dec_acc[tr,ii,:,layers2plot[ll],sf,:,:],axis=0)
    
    #get off-diagonal
    acc_diag = np.diagonal(acc_mat,offset=dd)
    for xx in range(dd):
      acc_diag = np.append(acc_diag, acc_mat[xx,180-dd+xx])
    # define where the "center" is for each of these small pairwise comparisons
    ori_axis_shifted = ori_axis + dd/2
    
    acc_all_init[ii,:] = acc_diag
    
  # get mean and std, plot errorbars.
  # Errorbars are over network initializations, if there is more than one. 
  # otherwise, no errorbars are plotted.
  meanacc = np.mean(acc_all_init,0)    
  erracc = np.std(acc_all_init,0)
 
  if len(init2plot)>1:
    plt.errorbar(ori_axis_shifted,meanacc,erracc,ecolor=colors_main[color_inds[tr],:],color=[0,0,0])
  else:
    plt.plot(ori_axis_shifted,meanacc,color=colors_main[color_inds[tr],:])
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==nLayers-1:
    plt.xlabel('Orientation (deg)')
    plt.xticks(np.arange(0,181,45))
  else:
    plt.xticks([])
  plt.xlim([np.min(ori_axis_shifted),np.max(ori_axis_shifted)])
  plt.ylabel('dec acc')
  
  plt.ylim(ylims)
  
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  plt.axhline(0.5, color=[0.8, 0.8, 0.8])
# finish up the entire plot   
plt.suptitle('%s\npopulation-level SVM decoding orients %d deg apart' % (training_strs[tr],dd))
figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)



#%%  Plot several different deltas (e.g. decode 0 vs 1 deg,, 0 vs 5 deg, etc...)
layers2plot = np.arange(0,nLayers)

tr=0  # can change this value to plot the netorks with different training sets

if tr==4 or tr==0:
  init2plot = [0]
else:
  init2plot = [0,1,2,3]
  
sf=0
#deltas = np.arange(1,6,1)
deltas = [1,5,10,15]
#ylims=[[0, 0.1],[0,0.1],[0,0.1],[0,0.7]]
cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(deltas)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(deltas),1),:,:]

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]

ylims = [0.2, 1]
plt.close('all')

fig=plt.figure()
#ax=fig.add_subplot(1,1,1)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  ax=fig.add_subplot(npx,npy, ll+1)
  allh=[];
  for dd in range(len(deltas)):
    acc_all_init = np.zeros([len(init2plot),nOri])
  
    for ii in init2plot:
   
      # average over image sets
      acc_mat = np.mean(all_dec_acc[tr,ii,:,layers2plot[ll],sf,:,:],axis=0)
      
   
      #get off-diagonal
      acc_diag = np.diagonal(acc_mat,offset=deltas[dd])
      for xx in range(deltas[dd]):
        acc_diag = np.append(acc_diag, acc_mat[xx,180-deltas[dd]+xx])
      # define where the "center" is for each of these small pairwise comparisons
      ori_axis_shifted = ori_axis + deltas[dd]/2
      
      acc_all_init[ii,:] = acc_diag
    
    # get mean and std, plot errorbars.
    # Errorbars are over network initializations, if there is more than one. 
    # otherwise, no errorbars are plotted.
    meanacc = np.mean(acc_all_init,0)    
    erracc = np.std(acc_all_init,0)
   
    if len(init2plot)>1:
      plt.errorbar(ori_axis_shifted,meanacc,erracc,ecolor=cols_grad[dd,0,:],color=[0,0,0])
    else:
      plt.plot(ori_axis_shifted,meanacc,color=cols_grad[dd,0,:])
      
    myline = mlines.Line2D(ori_axis_shifted,meanacc,color=cols_grad[dd,0,:])
    ax.add_line(myline)   
    allh.append(myline)
   
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==nLayers-1:
    plt.xlabel('Orientation (deg)')
    plt.xticks(np.arange(0,181,45))
    plt.legend(allh,['delta=%d deg'%dd for dd in deltas])
  else:
    plt.xticks([])
  plt.xlim([np.min(ori_axis_shifted),np.max(ori_axis_shifted)])
  plt.ylabel('dec acc')
  
  plt.ylim(ylims)

  plt.axhline(0.5, color=[0.8, 0.8, 0.8])

  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
             
# finish up the entire plot   
plt.suptitle('%s\npopulation-level SVM decoding orients delta deg apart' % (training_strs[tr]))
figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)



