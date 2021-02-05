#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:14:14 2021

@author: mmhender
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
import scipy
from scipy import ndimage

#import load_activations
#from copy import deepcopy
#import analyze_orient_tuning_jitter
#import matplotlib.lines as mlines
import pandas as pd

#%%
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))

training_strs = ['scratch_imagenet_rot_45_cos']
color_inds=[3]
#model='vgg16'
model='vgg16avgpool'
param_str='params1'
dataset_str=['FiltIms14AllSFCos']
#init_nums=[0,1,2,3] 
init_nums=[0]   
nInits = np.size(init_nums)
param_strs=[]
for ii in range(len(init_nums)):    
  if init_nums[ii]>0:
    param_strs.append(param_str+'_init%d'%init_nums[ii])
  else:
    param_strs.append(param_str)
    
nTrainingSchemes = len(training_strs)

#%%

all_acc = []
all_recall5 = []
all_loss = []
all_regloss = []

for tr in range(len(training_strs)):
  
  training_str = training_strs[tr]
  dataset = dataset_str[0]  
 
  all_acc_this_tr = []
  all_recall5_this_tr = []
  all_loss_this_tr = []
  all_regloss_this_tr = []
  for ii in range(nInits):
  
    # path info  
    param_str = param_strs[ii]
    save_path = os.path.join(root,'logs',model,'ImageNet',training_str,param_str)
    
    vals = pd.read_csv(os.path.join(save_path,'run-%s.-tag-eval_metrics_eval_acc.csv'%param_str)).values    
    all_acc_this_tr.append(vals)
    
    vals = pd.read_csv(os.path.join(save_path,'run-%s.-tag-eval_metrics_eval_recall_5.csv'%param_str)).values    
    all_recall5_this_tr.append(vals)
    
    vals = pd.read_csv(os.path.join(save_path,'run-%s.-tag-Losses_clone_loss.csv'%param_str)).values    
    all_loss_this_tr.append(vals)
    
    vals = pd.read_csv(os.path.join(save_path,'run-%s.-tag-Losses_regularization_loss.csv'%param_str)).values    
    all_regloss_this_tr.append(vals)
    
  all_acc.append(all_acc_this_tr) 
  all_recall5.append(all_recall5_this_tr) 
  all_loss.append(all_loss_this_tr) 
  all_regloss.append(all_regloss_this_tr) 
  
  
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

cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,nInits+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+nInits,1),:,:]

#%% plot acc over time
plt.rcParams['figure.figsize']=[10,6] 
tr=0;
eval_at_ckpt = [50000,100000,200000,400000]
smoothing=1
max_ckpt = 500000
plt.figure();
for cc in eval_at_ckpt:
  plt.axvline(cc,color=[0.8, 0.8, 0.8])

for ii in range(nInits):
  
  x_vals = all_acc[tr][ii][:,1]
  y_vals = all_acc[tr][ii][:,2]
  # smoothing values w gaussian kernel so trend can be better seen
  y_vals_smoothed = scipy.ndimage.gaussian_filter1d(y_vals,sigma=smoothing)
  inds2plot = np.where(x_vals<max_ckpt)[0]
#  plt.plot(x_vals[inds2plot],y_vals_smoothed[inds2plot],color=colors_main[color_inds[tr],:])
  plt.plot(x_vals[inds2plot],y_vals_smoothed[inds2plot],color=cols_grad[ii,0,:])
   
  
#  plt.errorbar(xvals,meanvals,sdvals,color = cols_grad[tr,0,:])

  plt.xlim([-10000,max_ckpt+10000])
  plt.xlabel('Training step number')
  plt.ylabel('Validation set accuracy')
  plt.title('%s\n(lines are different initializations of same model)'%training_strs[tr])
  
  
#%% plot top-5 recall over time
plt.rcParams['figure.figsize']=[10,6] 
tr=0;
eval_at_ckpt = [50000,100000,200000,400000]
smoothing = 50
max_ckpt = 500000
plt.figure();
for cc in eval_at_ckpt:
  plt.axvline(cc,color=[0.8, 0.8, 0.8])
#all_xvals = np.zeros()
for ii in range(nInits):
  
  x_vals = all_recall5[tr][ii][:,1]
  y_vals = all_recall5[tr][ii][:,2]
  # smoothing values w gaussian kernel so trend can be better seen
  y_vals_smoothed = scipy.ndimage.gaussian_filter1d(y_vals,sigma=smoothing)
  inds2plot = np.where(x_vals<max_ckpt)[0]
#  plt.plot(x_vals[inds2plot],y_vals_smoothed[inds2plot],color=colors_main[color_inds[tr],:])
  plt.plot(x_vals[inds2plot],y_vals_smoothed[inds2plot],color=cols_grad[ii,0,:])
  
  
#  plt.errorbar(xvals,meanvals,sdvals,color = cols_grad[tr,0,:])

  plt.xlim([-10000,max_ckpt+10000])
  plt.xlabel('Training step number')
  plt.ylabel('Validation set top-5 recall')
  plt.title('%s\n(lines are different initializations of same model)'%training_strs[tr])
  
  
#%% plot cross-entropy loss over time
plt.rcParams['figure.figsize']=[10,6] 
tr=0;
eval_at_ckpt = [50000,100000,200000,400000]
smoothing = 1
max_ckpt = 500000
plt.figure();
for cc in eval_at_ckpt:
  plt.axvline(cc,color=[0.8, 0.8, 0.8])
#all_xvals = np.zeros()
for ii in range(nInits):
  
  x_vals = all_loss[tr][ii][:,1]
  y_vals = all_loss[tr][ii][:,2]
  # smoothing values w gaussian kernel so trend can be better seen
  y_vals_smoothed = scipy.ndimage.gaussian_filter1d(y_vals,sigma=smoothing)
  inds2plot = np.where(x_vals<max_ckpt)[0]
#  plt.plot(x_vals[inds2plot],y_vals_smoothed[inds2plot],color=colors_main[color_inds[tr],:])
  plt.plot(x_vals[inds2plot],y_vals_smoothed[inds2plot],color=cols_grad[ii,0,:])
  
  
#  plt.errorbar(xvals,meanvals,sdvals,color = cols_grad[tr,0,:])

  plt.xlim([-10000,max_ckpt+10000])
  plt.xlabel('Training step number')
  plt.ylabel('Cross-entropy loss')
  plt.title('%s\n(lines are different initializations of same model)'%training_strs[tr])
  
  
#%% plot regularization loss over time
plt.rcParams['figure.figsize']=[10,6] 
tr=0;
eval_at_ckpt = [50000,100000,200000,400000]
smoothing = 1
max_ckpt = 500000
plt.figure();
for cc in eval_at_ckpt:
  plt.axvline(cc,color=[0.8, 0.8, 0.8])
#all_xvals = np.zeros()
for ii in range(nInits):
  
  x_vals = all_regloss[tr][ii][:,1]
  y_vals = all_regloss[tr][ii][:,2]
  # smoothing values w gaussian kernel so trend can be better seen
  y_vals_smoothed = scipy.ndimage.gaussian_filter1d(y_vals,sigma=smoothing)
  inds2plot = np.where(x_vals<max_ckpt)[0]
#  plt.plot(x_vals[inds2plot],y_vals_smoothed[inds2plot],color=colors_main[color_inds[tr],:])
  plt.plot(x_vals[inds2plot],y_vals_smoothed[inds2plot],color=cols_grad[ii,0,:])
  
  
#  plt.errorbar(xvals,meanvals,sdvals,color = cols_grad[tr,0,:])

  plt.xlim([-10000,max_ckpt+10000])
  plt.xlabel('Training step number')
  plt.ylabel('L2 regularization loss')
  plt.title('%s\n(lines are different initializations of same model)'%training_strs[tr])