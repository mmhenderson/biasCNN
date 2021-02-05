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

#training_strs = ['scratch_imagenet_rot_0_cos_stop_early']   # a randomly initialized, un-trained model
#training_strs = ['scratch_imagenet_rot_0_cos']  # model trained on upright images
#training_strs = ['scratch_imagenet_rot_22_cos']   # model trained on 22 deg rot iamges
#training_strs = ['scratch_imagenet_rot_45_cos']   # model trained on 45 deg rot images
training_strs = ['pretrained']   # a pre-trained model 

#%% define other basic parameters
nImageSets = 4
model='vgg16'
param_str='params1'
dataset_str=['FiltIms14AllSFCos']

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
  init_nums=[0,1,2,3]
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
     fit_pars_all = [] 
     r2_all = []    
     
     fastpars_all=[]
     
  # find the random seed(s) for the jitter that was used
  files=os.listdir(os.path.join(save_path))
  [jitt_file] = [ff for ff in files if '%s_fit_jitter'%layer_labels[0] in ff and 'pars' in ff];  
  rand_seed_str = jitt_file[jitt_file.find('jitter')+7:jitt_file.find('jitter')+13]  
    
  coords = []
  fit_pars = [] 
  r2 = []   
  fastpars = []
  # loop over layers and load fit parameters
  for ll in range(nLayers):
    
    # load coordinates of each network unit (spatial position and channel number)
    # [nUnits x 3] where third dim is [H,W,C]
    file_name =os.path.join(save_path,'%s_coordsHWC_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    coords.append(np.load(file_name))
  
    # load fit r2 [nUnits x nSF x nImageSets] 
    file_name= os.path.join(save_path,'%s_fit_jitter_%s_r2_each_sample_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    r2.append(np.load(file_name))
    
    # load the fit parameters [nUnits x nSF x nPars]
    # in par dimension, [0,1,2,3,4] are [center, k, amplitude, baseline, FWHM size]
    file_name= os.path.join(save_path,'%s_fit_jitter_%s_pars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    fit_pars.append(np.load(file_name))
    
    # load the fit parameters [nUnits x nSF x nPars]
    # in par dimension, [0,1,2,3,4] are [center, k, amplitude, baseline, FWHM size]
    file_name= os.path.join(save_path,'%s_fastpars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    fastpars.append(np.load(file_name).item())
 
  coords_all.append(coords)
  r2_all.append(r2)
  fit_pars_all.append(fit_pars)

  fastpars_all.append(fastpars)

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

#%% plot the proportion of units above r2 threshold, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
sf=0

# matrix to store anisotropy index for each layer    
prop_vals = np.zeros([nInits, np.size(layers2plot)])

for ii in range(nInits):
  # loop over network layers
  for ll in range(np.size(layers2plot)):
  
    # values to plot    
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))   
    # there are a few nans in here, putting a tiny value so it won't throw an error
    rvals[np.isnan(rvals)] = -1000
  
    prop_vals[ii,ll] = np.sum(rvals>r2_cutoff)/np.size(rvals)
    
# put the line for this spatial frequency onto the plot      
meanvals = np.mean(prop_vals,axis=0)
sdvals = np.std(prop_vals,axis=0)
plt.errorbar(np.arange(0,np.size(layers2plot),1),meanvals,sdvals,color = colors_main[color_ind,:])
plt.plot(np.arange(0,np.size(layers2plot),1),meanvals,marker='o',color = colors_main[color_ind,:])

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Proportion of units')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Prop of units with r2>%.2f\n%s'%(r2_cutoff,training_str))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, '%s_pct_units_vs_layer.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% plot the distribution of FWHM, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
sf=0

for ii in range(nInits):
  # loop over network layers
  for ll in range(np.size(layers2plot)):
  
    # values to plot    
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))   
    # there are a few nans in here, putting a tiny value so it won't throw an error
    rvals[np.isnan(rvals)] = -1000
    parvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,4]))
    vals_good = parvals[rvals>r2_cutoff]

    parts = plt.violinplot(vals_good,[ll])
    for pc in parts['bodies']:
        pc.set_color(colors_main[color_ind,:])
    parts['cbars'].set_color(colors_main[color_ind,:])
    parts['cmins'].set_color(colors_main[color_ind,:])
    parts['cmaxes'].set_color(colors_main[color_ind,:])
      

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('FWHM (deg)')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('FWHM (deg) averaged over units with r2>%.2f\n%s'%(r2_cutoff,training_str))  
fig.set_size_inches(10,7)
#%% plot the distribution of k, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
sf=0

for ii in range(nInits):
  # loop over network layers
  for ll in range(np.size(layers2plot)):
  
    # values to plot    
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))   
    # there are a few nans in here, putting a tiny value so it won't throw an error
    rvals[np.isnan(rvals)] = -1000
    parvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,1]))
    vals_good = parvals[rvals>r2_cutoff]

    parts = plt.violinplot(vals_good,[ll])
    for pc in parts['bodies']:
        pc.set_color(colors_main[color_ind,:])
    parts['cbars'].set_color(colors_main[color_ind,:])
    parts['cmins'].set_color(colors_main[color_ind,:])
    parts['cmaxes'].set_color(colors_main[color_ind,:])
      

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim([-10,500])
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('k (a.u.)')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Concentration parameter (k), all units with r2>%.2f\n%s'%(r2_cutoff,training_str))  
fig.set_size_inches(10,7)
#figname = os.path.join(figfolder, '%s_pct_units_vs_layer.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% plot the distribution of amplitudes, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
sf=0

for ii in range(nInits):
  # loop over network layers
  for ll in range(np.size(layers2plot)):
  
    # values to plot    
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))   
    # there are a few nans in here, putting a tiny value so it won't throw an error
    rvals[np.isnan(rvals)] = -1000
    parvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,2]))
    vals_good = parvals[rvals>r2_cutoff]

    parts = plt.violinplot(vals_good,[ll])
    for pc in parts['bodies']:
        pc.set_color(colors_main[color_ind,:])
    parts['cbars'].set_color(colors_main[color_ind,:])
    parts['cmins'].set_color(colors_main[color_ind,:])
    parts['cmaxes'].set_color(colors_main[color_ind,:])
      

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('amp (a.u.)')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Amplitude of tuning function, all units with r2>%.2f\n%s'%(r2_cutoff,training_str))  
fig.set_size_inches(10,7)
#%% Plot a histogram of tuning center distributions
# plot for a selected 4 layers only
layers2plot=[0,6,12,18]

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]

sf=0
ylims=[0,0.05]

# define bins to use the for the distribution
ori_bin_size=1
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2

npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  all_cvals=[]  # going to combine data across all initializations
  for ii in range(nInits):
    cvals = deepcopy(np.squeeze(fit_pars_all[ii][layers2plot[ll]][:,sf,0]))
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))
    rvals[np.isnan(rvals)] = -1000
    cvals=cvals[np.where(rvals>r2_cutoff)[0]]
    all_cvals = np.concatenate((all_cvals,cvals),axis=0)
    
  cvals=all_cvals
  # get the actual curve that describes the distribution of centers
  h = np.histogram(cvals, ori_bins) 
  # divide by total to get a proportion.
  real_y = h[0]/np.sum(h[0])
  
  plt.bar(bin_centers, real_y,width=ori_bin_size,color=colors_main[color_ind,:],zorder=100)
 
  plt.title('%s'%(layer_labels[layers2plot[ll]])) 
  plt.xlabel('Orientation (deg)')
  plt.ylabel('Proportion of units')
  plt.xticks(np.arange(0,nOri+1,45))
  plt.ylim(ylims)
  plt.xlim([np.min(ori_bins), np.max(ori_bins)])
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
  
    
plt.suptitle('%s\nFit Centers - All units r2>%.2f'%(training_str,r2_cutoff));

figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% Plot K versus the center (scatter plot)
# plot for a selected 4 layers only
layers2plot=[0,6,12,18]

plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]
maxpts = 10000 # downsampling a bit here to make the figure tractable to save
pp2plot=1 # index of k in the parameters array
ppname='k'
ylims = [[-5,50],[-5,50],[-5,500],[-5,500]]
sf=0

alpha_vals=[1,1,1,1]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')

plt.figure()
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
 
  allc = []  # going to combine data across all initializations
  allk = []
  for ii in range(nInits):
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))
    rvals[np.isnan(rvals)] = -1000
    inds2use = np.where(rvals>r2_cutoff)[0]
    # get values of center and other parameter of interest (k)
    cvals = deepcopy(np.squeeze(fit_pars_all[ii][layers2plot[ll]][inds2use,sf,0]))
    parvals = deepcopy(np.squeeze(fit_pars_all[ii][layers2plot[ll]][inds2use,sf,pp2plot]))    
   
    allc = np.concatenate((allc,cvals),axis=0)
    allk = np.concatenate((allk,parvals),axis=0)
    
  cvals=allc
  parvals=allk
  my_alpha=alpha_vals[ll]
  
  if np.size(cvals)>maxpts:
    inds2plot = np.random.randint(0,np.size(cvals),maxpts)
  else:
    inds2plot = np.arange(0,np.size(cvals))
  plt.plot(cvals[inds2plot],parvals[inds2plot],'.',markersize=1,color=colors_main[color_ind,:],alpha=my_alpha,zorder=100)

  plt.title(layer_labels[layers2plot[ll]])
  plt.ylim(ylims[ll])
  plt.xlabel('Center (deg)')
  plt.ylabel(ppname)
  plt.xticks(np.arange(0,181,45))
  
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
  
plt.suptitle('%s\n%s versus center'%(training_str,ppname))

figname = os.path.join(figfolder, '%s_%s.pdf' % (training_str,ppname))
#plt.savefig(figname, format='pdf',transparent=True)

#%% Plot histogram of tuning centers, for all layers in the network.
plt.rcParams.update({'font.size': 10})

# bins for the histograms
ori_bin_size=1
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2

layers2plot=np.arange(0,nLayers)
sf=0

npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # combine centers from all network initializations into one big distribution
  vals_all = []
  for ii in range(nInits):
    vals = deepcopy(np.squeeze(fit_pars_all[ii][layers2plot[ll]][:,sf,0]))
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))   
    rvals[np.isnan(rvals)] = -1000    
    vals=vals[np.where(rvals>r2_cutoff)[0]]
    
    vals_all = np.concatenate((vals_all,vals),axis=0)

  vals_all = np.ravel(vals_all)
  h = np.histogram(vals_all, ori_bins) 
  # divide by total to get a proportion.
  real_y = h[0]/np.sum(h[0])
  
  plt.bar(bin_centers, real_y,width=ori_bin_size,color=colors_main[color_ind,:],zorder=100)
  
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
  plt.xlim([np.min(ori_bins), np.max(ori_bins)])
  plt.title('%s'%(layer_labels[layers2plot[ll]])) 
 
  if ll==nLayers-1:
      plt.xlabel('Orientation (deg)')
      plt.ylabel('Prop. Units')
      plt.xticks(np.arange(0,nOri+1,45))
  else:
      plt.xticks([]) 
      
plt.suptitle('%s\nFit Centers - All units r2>%.2f'%(training_str,r2_cutoff));

#%% Plot K versus the center (scatter plot), all layers in the network
layers2plot=np.arange(0,nLayers,1)

pp2plot=1 # index of k in the parameters array
ppname='k'
#ylims = [[-5,50],[-5,50],[-5,500],[-5,500]]
sf=0
maxpts=10000
ylims = [-5,500]
alpha=1
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')

plt.figure()
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
 
  allc = []  # going to combine data across all initializations
  allk = []
  for ii in range(nInits):
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))
    rvals[np.isnan(rvals)] = -1000
    inds2use = np.where(rvals>r2_cutoff)[0]
    # get values of center and other parameter of interest (k)
    cvals = deepcopy(np.squeeze(fit_pars_all[ii][layers2plot[ll]][inds2use,sf,0]))
    parvals = deepcopy(np.squeeze(fit_pars_all[ii][layers2plot[ll]][inds2use,sf,pp2plot]))    
   
    allc = np.concatenate((allc,cvals),axis=0)
    allk = np.concatenate((allk,parvals),axis=0)
    
  cvals=allc
  parvals=allk
    
  if np.size(cvals)>maxpts:
    inds2plot = np.random.randint(0,np.size(cvals),maxpts)
  else:
    inds2plot = np.arange(0,np.size(cvals))
  plt.plot(cvals[inds2plot],parvals[inds2plot],'.',markersize=1,color=colors_main[color_ind,:],alpha=alpha,zorder=100)

  plt.title(layer_labels[layers2plot[ll]])
  plt.ylim(ylims)
  plt.xlabel('Center (deg)')
  plt.ylabel(ppname)
  plt.xticks(np.arange(0,181,45))
  
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
  
plt.suptitle('%s\n%s versus center - All units r2>%.2f'%(training_str,ppname,r2_cutoff));

#%% Plot histogram of r2, for all layers in the network.
plt.rcParams.update({'font.size': 10})

# bins for the histograms
layers2plot=np.arange(0,nLayers)
sf=0

npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # combine centers from all network initializations into one big distribution
  vals_all = []
  for ii in range(nInits):  
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))   
  
    vals_all = np.concatenate((vals_all,rvals),axis=0)

  vals_all = np.ravel(vals_all)
  plt.hist(vals_all, bins=np.arange(0,1,0.01),color=colors_main[color_ind,:]) 
  # divide by total to get a proportion.
#  real_y = h[0]/np.sum(h[0])
#  
#  plt.bar(bin_centers, real_y,width=ori_bin_size,color=colors_main[color_ind,:],zorder=100)
  
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
  plt.xlim([-0.05, 1.01])
  plt.title('%s'%(layer_labels[layers2plot[ll]])) 
 
  plt.axvline(r2_cutoff,color='k')
  
  if ll==nLayers-1:
      plt.xlabel('r2')
      plt.ylabel('Prop. Units')
      
  else:
      plt.xticks([]) 
      
plt.suptitle('%s\ndistribution r2 for units each layer'%(training_str));

#%% plot spatial position of all well-fit units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
pp=0;
alpha=0.1
r2_here=0.4
ii=0
nspat = info['activ_dims'][ll]
rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][ll][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_here)[0]

xvals = coords_all[ii][ll][:,1]
yvals = nspat - coords_all[ii][ll][:,0] # flip the y axis so orientations go clockwise from vertical

plt.subplot(1,2,1)
plt.plot(xvals,yvals,'.',color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('All responsive units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.subplot(1,2,2)
plt.plot(xvals[units_good],yvals[units_good],'.',color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('All well-fit units (r2>%.2f)'%r2_here)
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.suptitle('%s\n%s: sf=%.2f\n%s'%(training_str,dataset,info['sf_vals'][sf],layer_labels[ll]))

#%% plot spatial position of all cardinal-tuned units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
pp=0;
alpha=0.1
r2_here=0.4
nspat = info['activ_dims'][ll]
#r2_here=r2_cutoff
ii=0
# values to plot  
rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][ll][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_here)[0]
centers = deepcopy(np.squeeze(fit_pars_all[ii][ll][units_good,sf,0]))
amps = deepcopy(np.squeeze(fit_pars_all[ii][ll][units_good,sf,2]))

horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ii][ll][units_good,1]
yvals = nspat - coords_all[ii][ll][units_good,0] # flip the y axis so orientations go clockwise from vertical

plt.subplot(1,2,1)
plt.plot(xvals[horiz_units],yvals[horiz_units],'.',color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('Horizontal-tuned units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.subplot(1,2,2)
plt.plot(xvals[vert_units],yvals[vert_units],'.',color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('Vertical-tuned units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.suptitle('%s\n%s\n%s'%(training_str,dataset,layer_labels[ll]))


#%% plot spatial positions of units with high slope regions near 90
plt.figure();
unit_colors = cm.Blues(np.linspace(0,1,5))
alpha=1
ll=12

nspat = info['activ_dims'][ll]
maxvals = fastpars_all[ii][ll]['maxsqslopeori'][ii,:,sf]
highest_card_slope = np.where(np.logical_and(maxvals>85, maxvals<95))[0]
inds2plot = highest_card_slope
#chan_coords = coords_all[ll][:,2]
x_coords = coords_all[ii][ll][:,1]
y_coords = nspat - coords_all[ii][ll][:,0] # flip the y axis so orientations go clockwise from vertical

plt.plot(x_coords[inds2plot],y_coords[inds2plot],'.', color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('%s, %s\nunits with highest slope near 90'%(training_str, layer_labels[ll]))
plt.xlim([0,nspat])
plt.ylim([0,nspat])

#%% plot distribution of the maximum values of tuning functions, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')
plt.figure();
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
sf=0
layers2do = np.arange(0,nLayers,1)
for ll in range(nLayers):
  plt.subplot(npx,npy,ll+1)
  
  maxvals = fastpars_all[ii][ll]['maxori'][ii,:,sf]
  maxvals = maxvals[maxvals!=0]
  plt.hist(maxvals,bins=np.arange(0,nOri,1),color=colors_main[color_ind,:])
  plt.yticks([])
  if ll==len(layers2do)-1:
    plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
  else:
    plt.xticks([])
  
  plt.title(layer_labels[ll])
  
plt.suptitle('max of orient tuning functions\n%s, %s'%(training_str,dataset))

#%% plot distribution of the maximum slope regions of each unit's tuning function, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')
plt.figure();
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
sf=0
layers2do = np.arange(0,nLayers,1)
for ll in range(nLayers):
  plt.subplot(npx,npy,ll+1)
  
  maxvals = fastpars_all[ii][ll]['maxsqslopeori'][ii,:,sf]
  maxvals = maxvals[maxvals!=0]
  plt.hist(maxvals,bins=np.arange(0,nOri,1),color=colors_main[color_ind,:])
  plt.yticks([])
  if ll==len(layers2do)-1:
    plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
  else:
    plt.xticks([])
  
  plt.title(layer_labels[ll])
  
plt.suptitle('max slope regions of orient tuning functions\n%s, %s'%(training_str,dataset))

#%% plot mean resp vs the maximum slope regions of each unit's tuning function, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')
plt.figure();
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
sf=0
layers2do = np.arange(0,nLayers,1)
nUnitsPlot=10000
alpha=0.1
for ll in range(nLayers):
  plt.subplot(npx,npy,ll+1)
  
  maxvals = fastpars_all[ii][ll]['maxsqslopeori'][ii,:,sf]
  meanresp = fastpars_all[ii][ll]['meanresp'][ii,:,sf]
  inds2use = np.where(maxvals!=0)[0]
  inds2use= np.random.choice(inds2use,nUnitsPlot)
  plt.plot(maxvals[inds2use], meanresp[inds2use],'.',color=colors_main[color_ind,:],alpha=alpha)
#  plt.yticks([])
  if ll==len(layers2do)-1:
    plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
  else:
    plt.xticks([])
  
  plt.title(layer_labels[ll])
  
plt.suptitle('mean resp vs. orient where max slope occurs\n%s, %s'%(training_str,dataset))

#%% plot mean resp vs the maximum slope regions of each unit's tuning function, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')
plt.figure();
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
sf=0
layers2do = np.arange(0,nLayers,1)
nUnitsPlot=10000
alpha=0.1
for ll in range(nLayers):
  plt.subplot(npx,npy,ll+1)
  
  maxvals = fastpars_all[ii][ll]['maxsqslopeori'][ii,:,sf]
  maxslope = fastpars_all[ii][ll]['maxsqslopeval'][ii,:,sf]
  inds2use = np.where(maxvals!=0)[0]
  inds2use= np.random.choice(inds2use,nUnitsPlot)
  plt.plot(maxvals[inds2use], maxslope[inds2use],'.',color=colors_main[color_ind,:],alpha=alpha)
#  plt.yticks([])
  if ll==len(layers2do)-1:
    plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
  else:
    plt.xticks([])
  
  plt.title(layer_labels[ll])
  
plt.suptitle('Maximum slope (squared) vs. orient where max slope occurs\n%s, %s'%(training_str,dataset))



#%% load actual tfs from a single layer (slow)
ll=12
tr=0
ii=0
fn = os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num))
print('loading from %s\n'%fn)
tfs=np.load(fn)

#%% make plots of randomly selected well-fit units that are close to cardinals.
plt.rcParams['figure.figsize']=[14,10]
plt.close('all')
sf = 0
nUnitsPlot = 12
r2_here=r2_cutoff
nOri=180
rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][ll][:,sf,:],axis=1)))
cvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,0]))
avals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,2]))
svals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,4]))

rvals[np.isnan(rvals)] = -1000

# Now choose the units to plot, sorting by size
#units_good = np.where(np.logical_and(np.logical_and(np.logical_and(rvals>r2_here, ~np.isnan(cvals)), cvals>80), cvals<100))[0]
units_good = np.where(rvals>r2_here)[0]
#units_good = np.where(np.logical_and(rvals>r2_here, svals<80))[0]

units2plot = np.random.choice(units_good, nUnitsPlot, replace='False')

npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)
  tc = np.squeeze(tfs[:,units2plot[uu],sf,0:nOri])
  real_y = np.mean(tc,axis=0)
  pars = fit_pars_all[ii][ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])
  
  for ss in range(np.shape(tc)[0]):
    yvals = tc[ss,:]
#    plt.plot(ori_axis,yvals,color=colors_all[color_ind,2,:])
  plt.plot(ori_axis, real_y, color=colors_main[color_ind,:]) 
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.title('center=%d, size=%d, r2=%.2f'%(pars[0],pars[4], rvals[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('%s, %s\nExamples of good fits (r2>%.2f), %s'%(training_str,dataset,r2_here,layer_labels[ll]))
  


#%% plot a couple example units which have high slope near the cardinal axes

plt.figure();
sf=0
nOri=180
orients = np.arange(0,nOri,1)
dat = np.transpose(tfs[ii,:,sf,0:nOri])
sqslope_cards = np.mean(np.diff(dat[88:92,:],axis=0)**2,axis=0)
highest_card_slope = np.flipud(np.argsort(sqslope_cards))

unit_colors = cm.Blues(np.linspace(0,1,5))
nUnitsPlot=10

inds2plot = highest_card_slope[0:nUnitsPlot]
#inds2plot = np.random.choice(np.arange(0,np.shape(activ_by_ori_all[ll])[1]), nUnitsPlot, replace='False')
npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)

for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)
  ydat=dat[:,inds2plot[uu]]
#  ydat = ydat - np.mean(ydat)
  plt.plot(orients,ydat,color=colors_main[color_ind,:])
  coord = coords_all[ii][ll][inds2plot[uu],:]
  plt.title('coords=%d,%d,%d'%(coord[0],coord[1],coord[2]))
meany = np.mean(dat,axis=1)
meany = meany - np.mean(meany)
#plt.plot(orients,meany,color='k')
plt.xticks(np.arange(0,181,45))
plt.suptitle('units with high slope near cardinals: %s, %s, eval on %s sf=%.2f'%(training_str, layer_labels[ll], dataset, info['sf_vals'][sf]))
 