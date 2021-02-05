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
#figfolder = os.path.join(root, 'figures','FisherInfoPop')

#%% define parameters for what to load here

# loading all networks at once - 
# [random, trained upright images, trained 22 deg rot images, trained 45 deg rot images, pretrained]
#training_strs=['scratch_imagenet_rot_0_cos_stop_early','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos','pretrained']
training_strs=['scratch_imagenet_rot_0_stop_early','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos','pretrained']

ckpt_strs=['0','400000','400000','400000','0']
nInits_list = [1,1,1,1,1]
color_inds=[0,1,2,3,1]

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

tr=3  # can change this value to plot the netorks with different training sets

#if tr==4 or tr==0:
init2plot = [0]
#else:
#  init2plot = [0,1,2,3]
  
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
#figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)



#%%  Plot decoding of a single orient versus other orients
layers2plot = np.arange(0,nLayers)

tr=1  # can change this value to plot the netorks with different training sets

#if tr==4 or tr==0:
init2plot = [0]
#else:
#  init2plot = [0,1,2,3]
  
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
#figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)


#%%  Plot decoding of a single orient versus other orients
# overlay multiple central orients
layers2plot = np.arange(0,nLayers)

tr=2  # can change this value to plot the netorks with different training sets

#if tr==4 or tr==0:
init2plot = [0]
#else:
#  init2plot = [0,1,2,3]
  
sf=0
ori_focus = [0,90,45,135]
intensity_inds =[3,3,1,1]
#ori_focus = np.arange(0,41,1)
if color_inds[tr]==1:
  cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(ori_focus)+2)),axis=2),[0,1,2],[0,2,1])
  cols_grad = cols_grad[np.arange(2,2+len(ori_focus),1),:,:]
elif color_inds[tr]==2:
  cols_grad = np.moveaxis(np.expand_dims(cm.Greens(np.linspace(0,1,len(ori_focus)+2)),axis=2),[0,1,2],[0,2,1])
  cols_grad = cols_grad[np.arange(2,2+len(ori_focus),1),:,:]
elif color_inds[tr]==3:
  cols_grad = np.moveaxis(np.expand_dims(cm.Reds(np.linspace(0,1,len(ori_focus)+2)),axis=2),[0,1,2],[0,2,1])
  cols_grad = cols_grad[np.arange(2,2+len(ori_focus),1),:,:]
else:
  cols_grad = np.moveaxis(np.expand_dims(cm.Greys(np.linspace(0,1,len(ori_focus)+2)),axis=2),[0,1,2],[0,2,1])
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
      plt.errorbar(ori_axis_centered,meanacc,erracc,ecolor=cols_grad[intensity_inds[oo],0,:],color=[0,0,0])
    else:
      plt.plot(ori_axis_centered,meanacc,color=cols_grad[intensity_inds[oo],0,:])
      
    myline = mlines.Line2D(ori_axis_centered,meanacc,color=cols_grad[intensity_inds[oo],0,:])
    ax.add_line(myline)   
    allh.append(myline)
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==nLayers-1:
#    plt.xlabel('orient2')
    plt.xlabel('diff from orient1')
    plt.ylabel('dec acc')
    plt.xticks(np.arange(-30,31,30))
    plt.legend(allh,['orient1=%d deg'%oo for oo in ori_focus],bbox_to_anchor=[1.01, 0.82])
  else:
    plt.xticks([])
  
  plt.xlim([-30,30])
  
#  plt.xlim([np.min(ori_axis_centered),np.max(ori_axis_centered)+1])
  
  plt.ylim(ylims)
  
  for xx in np.arange(-90,91,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  plt.axhline(0.5, color=[0.8, 0.8, 0.8])
# finish up the entire plot   
plt.suptitle('%s\npopulation-level SVM decoding: each orient versus other orients' % (training_strs[tr]))
#figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)



#%%  Plot decoding on off diagonals - information abt fine discriminations.
layers2plot = np.arange(0,nLayers)

tr=1  # can change this value to plot the netorks with different training sets

#if tr==4 or tr==0:
init2plot = [0]
#else:
#  init2plot = [0,1,2,3]
  
sf=0
dd=2
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
    plt.ylabel('dec acc')
  else:
    plt.xticks([])
 
  plt.xlim([np.min(ori_axis_shifted),np.max(ori_axis_shifted)])
  
  
  plt.ylim(ylims)
  
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  plt.axhline(0.5, color=[0.8, 0.8, 0.8])
# finish up the entire plot   
plt.suptitle('%s\npopulation-level SVM decoding orients %d deg apart' % (training_strs[tr],dd))
#figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)



#%%  Plot several different deltas (e.g. decode 0 vs 1 deg,, 0 vs 5 deg, etc...)
layers2plot = np.arange(0,nLayers)

tr=3  # can change this value to plot the netorks with different training sets

#if tr==4 or tr==0:
init2plot = [0]
#else:
#  init2plot = [0,1,2,3]
  
sf=0
deltas = np.arange(1,6,1)
#deltas = [1,5,10,15]
#ylims=[[0, 0.1],[0,0.1],[0,0.1],[0,0.7]]
cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(deltas)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(deltas),1),:,:]

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]

ylims = [0.2, 1.2]
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
    plt.legend(allh,['delta=%d deg'%dd for dd in deltas],bbox_to_anchor=[1.01, 1.01])
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
#figname = os.path.join(figfolder, '%s_decoding.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)


#%% Plot fine decoding bias - overlay types of bias, one network training scheme at a time

# which bias to plot?
pp2plot=[0,1,2]  # set to 0, 1 or 2 to plot [FIB-0, FIB-22, FIB-45]

tr=3
cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(tr2plot),1),:,:]

sf=0
delta=4


# based on the way the decoding dissimilarity matrix is set up, delta (orientation 
# diff between fine discrims) determines the "center" of each discrimination
ori_axis_shifted = ori_axis + delta/2

# parameters for calculating decoding bias
# define the bins of interest
b = np.arange(22.5,nOri,90)  # baseline
t = np.arange(67.5,nOri,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
c = np.arange(0,nOri,90) # cardinals
o = np.arange(45,nOri,90)  # obliques
bin_size=20

baseline_inds = []
for ii in range(np.size(b)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis_shifted-b[ii])<bin_size/2, np.abs(ori_axis_shifted-(nOri+b[ii]))<bin_size/2))[0])
  baseline_inds=np.append(baseline_inds,inds)
baseline_inds = np.uint64(baseline_inds)
            
card_inds = []
for ii in range(np.size(c)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis_shifted-c[ii])<bin_size/2, np.abs(ori_axis_shifted-(nOri+c[ii]))<bin_size/2))[0])
  card_inds=np.append(card_inds,inds)
card_inds = np.uint64(card_inds)
   
obl_inds = []
for ii in range(np.size(o)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis_shifted-o[ii])<bin_size/2, np.abs(ori_axis_shifted-(nOri+o[ii]))<bin_size/2))[0])
  obl_inds=np.append(obl_inds,inds)
obl_inds = np.uint64(obl_inds)
 
twent_inds = []
for ii in range(np.size(t)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis_shifted-t[ii])<bin_size/2, np.abs(ori_axis_shifted-(nOri+t[ii]))<bin_size/2))[0])
  twent_inds=np.append(twent_inds,inds)
twent_inds = np.uint64(twent_inds)

ii=0
peak_inds=[card_inds, twent_inds,obl_inds]
lstrings=['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline']

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

allh = []
layers2plot = np.arange(0,nLayers,1)

alpha=0.01;
# for each layer, compare bias for trained models versus random models
pvals_trained_vs_random=np.zeros([1, nLayers])
nTotalComp = np.size(pvals_trained_vs_random)
# matrix to store anisotropy index for each layer    
aniso_vals = np.zeros([len(pp2plot),1,nImageSets,np.size(layers2plot)])

# loop over network layers
for ww1 in range(np.size(layers2plot)):
  # loop over networks with each training set
  for pp in range(len(pp2plot)):
    
    # loop over random image sets
    for kk in range(nImageSets):

      # FI is nOri pts long
      # average over image sets
      acc_mat = np.mean(all_dec_acc[tr,ii,:,layers2plot[ww1],sf,:,:],axis=0)
      #get off-diagonal
      acc_diag = np.diagonal(acc_mat,offset=delta)
      for xx in range(delta):
        acc_diag = np.append(acc_diag, acc_mat[xx,180-delta+xx])
      
      # take the bins of interest to get bias
      base_discrim=  acc_diag[baseline_inds]
      peak_discrim = acc_diag[peak_inds[pp2plot[pp]]]
      
      # final value for this FIB: difference divided by sum 
      aniso_vals[pp,ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
  
# put the line for each FIB onto the plot 
# error bars are across 4 image sets
for pp in range(len(pp2plot)):    
  vals = np.squeeze(np.mean(aniso_vals[pp,:,:,:],1))
  errvals = np.squeeze(np.std(aniso_vals[pp,:,:,:],1)) 
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=colors_main[pp2plot[pp]+1,:],zorder=21)
  
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color=colors_main[pp2plot[pp]+1,:])
  ax.add_line(myline)   
  allh.append(myline)
  
# finish up the entire plot
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Information Bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.legend(allh, [lstrings[pp] for pp in range(len(pp2plot))])
plt.suptitle('Decoding bias: %s\ndelta=%d'%(training_strs[tr], delta))  
fig.set_size_inches(10,7)
#figname = os.path.join(figfolder, 'Pretrained_vs_random_%s.pdf'%lstrings[pp])
#plt.savefig(figname, format='pdf',transparent=True)



#%% Plot fine decoding bias - overlay training schemes, one type of bias at a time

# which bias to plot?
pp=2  # set to 0, 1 or 2 to plot [FIB-0, FIB-22, FIB-45]

tr2plot=[0,1,2,3] 
cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(tr2plot),1),:,:]


sf=0
delta=2

# based on the way the decoding dissimilarity matrix is set up, delta (orientation 
# diff between fine discrims) determines the "center" of each discrimination
ori_axis_shifted = ori_axis + delta/2

# parameters for calculating decoding bias
# define the bins of interest
b = np.arange(22.5,nOri,90)  # baseline
t = np.arange(67.5,nOri,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
c = np.arange(0,nOri,90) # cardinals
o = np.arange(45,nOri,90)  # obliques
bin_size=20

baseline_inds = []
for ii in range(np.size(b)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis_shifted-b[ii])<bin_size/2, np.abs(ori_axis_shifted-(nOri+b[ii]))<bin_size/2))[0])
  baseline_inds=np.append(baseline_inds,inds)
baseline_inds = np.uint64(baseline_inds)
            
card_inds = []
for ii in range(np.size(c)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis_shifted-c[ii])<bin_size/2, np.abs(ori_axis_shifted-(nOri+c[ii]))<bin_size/2))[0])
  card_inds=np.append(card_inds,inds)
card_inds = np.uint64(card_inds)
   
obl_inds = []
for ii in range(np.size(o)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis_shifted-o[ii])<bin_size/2, np.abs(ori_axis_shifted-(nOri+o[ii]))<bin_size/2))[0])
  obl_inds=np.append(obl_inds,inds)
obl_inds = np.uint64(obl_inds)
 
twent_inds = []
for ii in range(np.size(t)):        
  inds = list(np.where(np.logical_or(np.abs(ori_axis_shifted-t[ii])<bin_size/2, np.abs(ori_axis_shifted-(nOri+t[ii]))<bin_size/2))[0])
  twent_inds=np.append(twent_inds,inds)
twent_inds = np.uint64(twent_inds)

ii=0
peak_inds=[card_inds, twent_inds,obl_inds]
lstrings=['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline']

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

allh = []
layers2plot = np.arange(0,nLayers,1)

alpha=0.01;
# for each layer, compare bias for trained models versus random models
pvals_trained_vs_random=np.zeros([1, nLayers])
nTotalComp = np.size(pvals_trained_vs_random)
# matrix to store anisotropy index for each layer    
aniso_vals = np.zeros([len(tr2plot),1,nImageSets,np.size(layers2plot)])

# loop over network layers
for ww1 in range(np.size(layers2plot)):
  # loop over networks with each training set
  for tr in range(len(tr2plot)):
    
    # loop over random image sets
    for kk in range(nImageSets):

      # FI is nOri pts long
      # average over image sets
      acc_mat = np.mean(all_dec_acc[tr,ii,:,layers2plot[ww1],sf,:,:],axis=0)
      #get off-diagonal
      acc_diag = np.diagonal(acc_mat,offset=delta)
      for xx in range(delta):
        acc_diag = np.append(acc_diag, acc_mat[xx,180-delta+xx])
      
      # take the bins of interest to get bias
      base_discrim=  acc_diag[baseline_inds]
      peak_discrim = acc_diag[peak_inds[pp]]
      
      # final value for this FIB: difference divided by sum 
      aniso_vals[tr,ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
  
# put the line for each FIB onto the plot 
# error bars are across 4 image sets
for tr in range(len(tr2plot)):    
  vals = np.squeeze(np.mean(aniso_vals[tr,:,:,:],1))
  errvals = np.squeeze(np.std(aniso_vals[tr,:,:,:],1)) 
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=colors_main[color_inds[tr],:],zorder=21)
  
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color=colors_main[color_inds[tr],:])
  ax.add_line(myline)   
  allh.append(myline)
  
# finish up the entire plot
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Information Bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.legend(allh, training_strs)
plt.suptitle('Decoding bias: %s\ndelta=%d'%(lstrings[pp], delta))  
fig.set_size_inches(10,7)
#figname = os.path.join(figfolder, 'Pretrained_vs_random_%s.pdf'%lstrings[pp])
#plt.savefig(figname, format='pdf',transparent=True)
