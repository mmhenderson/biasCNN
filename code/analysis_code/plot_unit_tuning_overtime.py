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
import matplotlib.lines as mlines

von_mises_deg = analyze_orient_tuning_jitter.von_mises_deg
get_fwhm = analyze_orient_tuning_jitter.get_fwhm
get_r2 = analyze_orient_tuning_jitter.get_r2

#%% paths
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root,'figures','UnitTuning')

#%% define which network to load 
# plotting network trained on upright images, at several time steps.
#training_strs = ['scratch_imagenet_rot_0_cos','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_0_cos']  # model trained on upright images
training_strs = ['scratch_imagenet_rot_45_cos','scratch_imagenet_rot_45_cos','scratch_imagenet_rot_45_cos','scratch_imagenet_rot_45_cos']  # model trained on upright images

ckpt_strs=['50000','100000','200000','400000']
#%% define other basic parameters
nImageSets = 4
model='vgg16'
param_str='params1'
dataset_str=['FiltIms14AllSFCos']
color_ind=3
init_nums=[0]    
nInits = np.size(init_nums)
param_strs=[]
for ii in range(len(init_nums)):    
  if init_nums[ii]>0:
    param_strs.append(param_str+'_init%d'%init_nums[ii])
  else:
    param_strs.append(param_str)

nTrainingSchemes = len(training_strs)

# when identifying well-fit units, what criteria to use?
r2_cutoff = 0.4;

#%% loop to load all the data (orientation tuning fit parameters for all units)
for tr in range(len(training_strs)):
  
  training_str = training_strs[tr]
  ckpt_num = ckpt_strs[tr]
  dataset = dataset_str[0]  
 
  ii=0
  
  # path info  
  param_str = param_strs[ii]
  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset) 
 
  # get information about the images/network
  if tr==0:
     info = load_activations.get_info(model,dataset)
     nSF = np.size(np.unique(info['sflist']))
     nLayers = info['nLayers']      
     layer_labels = info['layer_labels']    
     nOri = info['nOri']
     ori_axis = np.arange(0, nOri,1)
         
     # initialize these arrays (will be across all init of the network)    
     coords_all = []    
     fit_pars_all = [] 
     r2_all = []     
     
  # find the random seed(s) for the jitter that was used
  files=os.listdir(os.path.join(save_path))
  [jitt_file] = [ff for ff in files if '%s_fit_jitter'%layer_labels[0] in ff and 'pars' in ff and ckpt_strs[tr] in ff];  
  rand_seed_str = jitt_file[jitt_file.find('jitter')+7:jitt_file.find('jitter')+13]  
    
  coords = []
  fit_pars = [] 
  r2 = []   
  # loop over layers and load fit parameters
  for ll in range(nLayers):
    
    # load coordinates of each network unit (spatial position and channel number)
    # [nUnits x 3] where third dim is [H,W,C]
    file_name =os.path.join(save_path,'%s_coordsHWC_all_responsive_units_eval_at_ckpt_%s.npy'%(layer_labels[ll],ckpt_num))
    print('loading from %s\n'%file_name)
    coords.append(np.load(file_name))
  
    # load fit r2 [nUnits x nSF x nImageSets] 
    file_name= os.path.join(save_path,'%s_fit_jitter_%s_r2_each_sample_eval_at_ckpt_%s.npy'%(layer_labels[ll],rand_seed_str,ckpt_num))
    print('loading from %s\n'%file_name)
    r2.append(np.load(file_name))
    
    # load the fit parameters [nUnits x nSF x nPars]
    # in par dimension, [0,1,2,3,4] are [center, k, amplitude, baseline, FWHM size]
    file_name= os.path.join(save_path,'%s_fit_jitter_%s_pars_eval_at_ckpt_%s.npy'%(layer_labels[ll],rand_seed_str,ckpt_num))
    print('loading from %s\n'%file_name)
    fit_pars.append(np.load(file_name))
 
  coords_all.append(coords)
  r2_all.append(r2)
  fit_pars_all.append(fit_pars)

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

#%% Plot histogram of tuning centers, for four example layers. Save a separate figure for each timept.

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]

# bins for the histograms
ori_bin_size=1
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2

layers2plot=np.asarray([0,6,12,18])
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
 
sf=0
ylims = [0, 0.065]
tr2plot=[0,1,2,3] 

cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(tr2plot),1),:,:]

for tr in range(len(tr2plot)):
  
  plt.figure()
  
  for ll in range(np.size(layers2plot)):
    plt.subplot(npx,npy, ll+1)
     
   
    vals_all = []
    
    ii=0
    vals = deepcopy(np.squeeze(fit_pars_all[tr2plot[tr]][layers2plot[ll]][:,sf,0]))
    rvals = deepcopy(np.squeeze(np.mean(r2_all[tr2plot[tr]][layers2plot[ll]][:,sf,:],axis=1)))   
    rvals[np.isnan(rvals)] = -1000    
    vals=vals[np.where(rvals>r2_cutoff)[0]]
    
    vals_all = np.concatenate((vals_all,vals),axis=0)
  
    vals_all = np.ravel(vals_all)
    h = np.histogram(vals_all, ori_bins) 
    # divide by total to get a proportion.
    real_y = h[0]/np.sum(h[0])
    
    plt.bar(bin_centers, real_y,width=ori_bin_size,color=cols_grad[tr2plot[tr],0,:],zorder=100)
    
    for xx in np.arange(45,180,45):
      plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
    plt.xlim([np.min(ori_bins), np.max(ori_bins)])
    plt.title('%s'%(layer_labels[layers2plot[ll]])) 
    plt.ylim(ylims)
    if ll==nLayers-1:
        plt.xlabel('Orientation (deg)')
        plt.ylabel('Prop. Units')
        plt.xticks(np.arange(0,nOri+1,45))
    else:
        plt.xticks([]) 
        
  plt.suptitle('%s\nFit Centers - All units r2>%.2f\nckpt=%s'%(training_str,r2_cutoff,ckpt_strs[tr2plot[tr]]));
  
  figname = os.path.join(figfolder, 'Centerhist_ckpt%s.pdf'%ckpt_strs[tr2plot[tr]])
  plt.savefig(figname, format='pdf',transparent=True)

#%% Plot K versus the center (scatter plot), for four example layers. Save a separate figure for each ckpt.
layers2plot = np.asarray([0,6,12,18])
pp2plot=1 # index of k in the parameters array
ppname='k'
sf=0
maxpts=20000
ylims = [-5,500]
alpha=1
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')


tr2plot=[0,1,2,3] 
cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(tr2plot),1),:,:]

for tr in range(len(tr2plot)):
 
  plt.figure()
  ll=2
#  for ll in range(len(layers2plot)):
  
  plt.subplot(npx,npy, ll+1)
 
  allc = []  
  allk = []
  ii=0
  
  rvals = deepcopy(np.squeeze(np.mean(r2_all[tr2plot[tr]][layers2plot[ll]][:,sf,:],axis=1)))
  rvals[np.isnan(rvals)] = -1000
  inds2use = np.where(rvals>r2_cutoff)[0]
  # get values of center and other parameter of interest (k)
  cvals = deepcopy(np.squeeze(fit_pars_all[tr2plot[tr]][layers2plot[ll]][inds2use,sf,0]))
  parvals = deepcopy(np.squeeze(fit_pars_all[tr2plot[tr]][layers2plot[ll]][inds2use,sf,pp2plot]))    
 
  allc = np.concatenate((allc,cvals),axis=0)
  allk = np.concatenate((allk,parvals),axis=0)
    
  cvals=allc
  parvals=allk
    
  if np.size(cvals)>maxpts:
    inds2plot = np.random.randint(0,np.size(cvals),maxpts)
  else:
    inds2plot = np.arange(0,np.size(cvals))
  plt.plot(cvals[inds2plot],parvals[inds2plot],'o',markersize=1,color=cols_grad[tr2plot[tr],0,:],alpha=alpha,zorder=100)

  plt.title(layer_labels[layers2plot[ll]])
  plt.ylim(ylims)
  plt.xlabel('Center (deg)')
  plt.ylabel(ppname)
  plt.xticks(np.arange(0,181,45))
  
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
      
  plt.suptitle('%s\n%s versus center - All units r2>%.2f\nckpt=%s'%(training_str,ppname,r2_cutoff,ckpt_strs[tr2plot[tr]]));

  figname = os.path.join(figfolder, 'KvsCenter_ds_ckpt%s.pdf'%ckpt_strs[tr2plot[tr]])
  plt.savefig(figname, format='pdf',transparent=True)

#%% plot the proportion of units above r2 threshold, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=plt.subplot(1,1,1)
allh=[]

layers2plot = np.arange(0,nLayers,1)
sf=0

tr2plot=[0,1,2,3] 
cols_grad = np.moveaxis(np.expand_dims(cm.Reds(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
#cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(tr2plot),1),:,:]

for tr in range(len(tr2plot)):
  # matrix to store anisotropy index for each layer    
  prop_vals = np.zeros([nInits, np.size(layers2plot)])
  
  ii=0
#  for ii in range(nInits):
    # loop over network layers
  for ll in range(np.size(layers2plot)):
  
    # values to plot    
    rvals = deepcopy(np.squeeze(np.mean(r2_all[tr2plot[tr]][layers2plot[ll]][:,sf,:],axis=1)))   
    # there are a few nans in here, putting a tiny value so it won't throw an error
    rvals[np.isnan(rvals)] = -1000
  
    prop_vals[ii,ll] = np.sum(rvals>r2_cutoff)/np.size(rvals)
      
  # put the line for this spatial frequency onto the plot      
  meanvals = np.mean(prop_vals,axis=0)
  sdvals = np.std(prop_vals,axis=0)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),meanvals,sdvals,color = cols_grad[tr,0,:])

  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),meanvals,color=cols_grad[tr,0,:])
  ax.add_line(myline)   
  allh.append(myline)
    
# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Proportion of units')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.legend(allh,[ckpt_strs[tr] for tr in tr2plot])

# finish up the entire plot
plt.suptitle('Prop of units with r2>%.2f\n%s'%(r2_cutoff,training_str))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, '%s_pct_units_vs_layer.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)


#%% Plot histogram of tuning centers, for all layers in the network.
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]

# bins for the histograms
ori_bin_size=1
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
 
sf=0
ylims = [0, 0.065]
tr2plot=[0,1,2,3] 
cols_grad = np.moveaxis(np.expand_dims(cm.Reds(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
#cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(tr2plot),1),:,:]

for tr in range(len(tr2plot)):
  
  plt.figure()
  
  for ll in range(np.size(layers2plot)):
    plt.subplot(npx,npy, ll+1)
     
    # combine centers from all network initializations into one big distribution
    vals_all = []
    
    ii=0
  #  for ii in range(nInits):
    vals = deepcopy(np.squeeze(fit_pars_all[tr2plot[tr]][layers2plot[ll]][:,sf,0]))
    rvals = deepcopy(np.squeeze(np.mean(r2_all[tr2plot[tr]][layers2plot[ll]][:,sf,:],axis=1)))   
    rvals[np.isnan(rvals)] = -1000    
    vals=vals[np.where(rvals>r2_cutoff)[0]]
    
    vals_all = np.concatenate((vals_all,vals),axis=0)
  
    vals_all = np.ravel(vals_all)
    h = np.histogram(vals_all, ori_bins) 
    # divide by total to get a proportion.
    real_y = h[0]/np.sum(h[0])
    
    plt.bar(bin_centers, real_y,width=ori_bin_size,color=cols_grad[tr,0,:],zorder=100)
    
    for xx in np.arange(45,180,45):
      plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
    plt.xlim([np.min(ori_bins), np.max(ori_bins)])
    plt.title('%s'%(layer_labels[layers2plot[ll]])) 
    plt.ylim(ylims)
    if ll==nLayers-1:
        plt.xlabel('Orientation (deg)')
        plt.ylabel('Prop. Units')
        plt.xticks(np.arange(0,nOri+1,45))
    else:
        plt.xticks([]) 
        
  plt.suptitle('%s\nFit Centers - All units r2>%.2f\nckpt=%s'%(training_str,r2_cutoff,ckpt_strs[tr2plot[tr]]));


#%% Plot K versus the center (scatter plot), all layers in the network
layers2plot=np.arange(0,nLayers,1)

pp2plot=1 # index of k in the parameters array
ppname='k'
#ylims = [[-5,50],[-5,50],[-5,500],[-5,500]]
sf=0
maxpts=100000000
ylims = [-5,500]
#ylims = [-5,150]
alpha=1
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')

tr2plot=[0,1,2,3] 
cols_grad = np.moveaxis(np.expand_dims(cm.Reds(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
#cols_grad = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,len(tr2plot)+2)),axis=2),[0,1,2],[0,2,1])
cols_grad = cols_grad[np.arange(2,2+len(tr2plot),1),:,:]
#tr2plot=[0]

for tr in range(len(tr2plot)):
 
  plt.figure()
  for ll in range(np.size(layers2plot)):
    plt.subplot(npx,npy, ll+1)
   
    allc = []  # going to combine data across all initializations
    allk = []
    ii=0
    
#    for ii in range(nInits):
    rvals = deepcopy(np.squeeze(np.mean(r2_all[tr2plot[tr]][layers2plot[ll]][:,sf,:],axis=1)))
    rvals[np.isnan(rvals)] = -1000
    inds2use = np.where(rvals>r2_cutoff)[0]
    # get values of center and other parameter of interest (k)
    cvals = deepcopy(np.squeeze(fit_pars_all[tr2plot[tr]][layers2plot[ll]][inds2use,sf,0]))
    parvals = deepcopy(np.squeeze(fit_pars_all[tr2plot[tr]][layers2plot[ll]][inds2use,sf,pp2plot]))    
   
    allc = np.concatenate((allc,cvals),axis=0)
    allk = np.concatenate((allk,parvals),axis=0)
      
    cvals=allc
    parvals=allk
      
    if np.size(cvals)>maxpts:
      inds2plot = np.random.randint(0,np.size(cvals),maxpts)
    else:
      inds2plot = np.arange(0,np.size(cvals))
    plt.plot(cvals[inds2plot],parvals[inds2plot],'.',markersize=1,color=cols_grad[tr,0,:],alpha=alpha,zorder=100)
  
    plt.title(layer_labels[layers2plot[ll]])
    plt.ylim(ylims)
    plt.xlabel('Center (deg)')
    plt.ylabel(ppname)
    plt.xticks(np.arange(0,181,45))
    
    for xx in np.arange(45,180,45):
      plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
    
  plt.suptitle('%s\n%s versus center - All units r2>%.2f\nckpt=%s'%(training_str,ppname,r2_cutoff,ckpt_strs[tr2plot[tr]]));


