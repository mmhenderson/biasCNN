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
#import scipy
from copy import deepcopy
import analyze_orient_tuning_jitter
import circ_reg_tools
import matplotlib.lines as mlines
import analyze_fit_params

#import astropy
von_mises_deg = analyze_orient_tuning_jitter.von_mises_deg
get_fwhm = analyze_orient_tuning_jitter.get_fwhm
get_r2 = analyze_orient_tuning_jitter.get_r2

#% make a big color map - nSF x nNetworks x RBGA
nsteps = 8
colors_all_1 = np.moveaxis(np.expand_dims(cm.Greys(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
colors_all_2 = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
#colors_all_2 = np.moveaxis(np.expand_dims(cm.Blues(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
colors_all_3 = np.moveaxis(np.expand_dims(cm.YlGn(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
#colors_all_3 = np.moveaxis(np.expand_dims(cm.Greens(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
colors_all_4 = np.moveaxis(np.expand_dims(cm.OrRd(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
#colors_all_4 = np.moveaxis(np.expand_dims(cm.Reds(np.linspace(0,1,nsteps)),axis=2),[0,1,2],[0,2,1])
colors_all = np.concatenate((colors_all_1[np.arange(2,nsteps,1),:,:],colors_all_2[np.arange(2,nsteps,1),:,:],colors_all_3[np.arange(2,nsteps,1),:,:],colors_all_4[np.arange(2,nsteps,1),:,:]),axis=1)

# plot the color map
#plt.close('all')
#plt.figure();plt.imshow(colors_all)

int_inds = [3,3,3,3]

colors_main = np.asarray([colors_all[int_inds[ii],ii,:] for ii in range(np.size(int_inds))])
# plot the color map
#plt.figure();plt.imshow(np.expand_dims(colors_main,axis=0))

#%% get the data ready to go...then can run any below cells independently.
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root,'figures','UnitTuning')

load_stats=0
load_weights=0
load_mean_tfs = 1
load_all_tfs = 0

nSamples = 4

model='vgg16'
#model='vgg16avgpool'

param_str='params1'
init_nums=[0]
nInits = np.size(init_nums)
param_strs=[]
for ii in range(len(init_nums)):    
  if init_nums[ii]>0:
    param_strs.append(param_str+'_init%d'%init_nums[ii])
  else:
    param_strs.append(param_str)
if np.size(init_nums)>1:
  init_str='inits '
  for ii in range(np.size(init_nums)):
    init_str+='%d '%init_nums[ii]
else:
  init_str='init %d'%init_nums[0]
  
#training_strs=['pretrained']
#training_strs = ['scratch_imagenet_rot_0_cos_stop_early']
training_strs = ['scratch_imagenet_rot_0_stop_early_init_ones']
#training_strs = ['scratch_imagenet_rot_0_stop_early_weight_init_var_scaling']
#training_strs = ['scratch_imagenet_rot_0_cos']

color_ind=0
ckpt_strs = ['0']
#ckpt_strs=['400000']
#dataset_str = ['FiltIms11Cos_SF_0.01']
dataset_str=['FiltIms14AllSFCos']
#dataset_str=['SpatFreqGratings']


nTrainingSchemes = np.size(training_strs)
if np.size(dataset_str)==1:
  dataset_str = np.tile(dataset_str,nTrainingSchemes)

sf_vals = np.logspace(np.log10(0.02),np.log10(.4),6)*140/224

# when identifying well-fit units, what criteria to use?
r2_cutoff = 0.4;

tr=0
for ii in range(nInits):
  
  # path info
  training_str = training_strs[tr]
  ckpt_num = ckpt_strs[tr]
  dataset = dataset_str[tr]  
  param_str = param_strs[ii]
  save_path = os.path.join(root,'code','unit_tuning',model,training_str,param_str,dataset) 
  weight_save_path = os.path.join(root,'weights',model,training_str,param_str)
   
  # get informatio about the images/network
  if ii==0:
     info = load_activations.get_info(model,dataset)
     nSF = np.size(np.unique(info['sflist']))
     nLayers = info['nLayers']      
     layer_labels = info['layer_labels']
     if info['nPhase']==2:
       nOri=360
     else:
       nOri = info['nOri']
     ori_axis = np.arange(0, nOri,1)
    
     if nSF==1:
       sf_labels=['broadband SF']
     else:
       sf_labels=['%.2f cpp'%ff for ff in sf_vals]
       
     # initialize these arrays (will be across all init of the network)
     resp_units_all = []  
     resp_mean_all = []
     coords_all = []    
     fit_pars_all = [] 
     r2_all = []     
     fit_pars_CI_all = []
     center_dist_pars_all = []
  
  # load the actual network weights at the given time step
  if not 'pixel' in model and load_weights:
    
    file_name= os.path.join(weight_save_path,'AllNetworkWeights_eval_at_ckpt_%s0000.npy'%(ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    w_all = np.load(file_name)
    w_layer_inds = np.asarray([0,1,3,4,6,7,8,10,11,12,14,15,16,18,19,20])
    w_layer_labels = [layer_labels[ii] for ii in w_layer_inds if ii<nLayers]
        
  # find the random seed(s) for the jitter that was used
  files=os.listdir(os.path.join(save_path))
  [jitt_file] = [ff for ff in files if '%s_fit_jitter'%layer_labels[0] in ff and 'pars' in ff];  
  rand_seed_str = jitt_file[jitt_file.find('jitter')+7:jitt_file.find('jitter')+13]
  
   
  # initialize these sub-arrays (will be across all layers)
  resp_units = []  
  coords = []
  fit_pars = [] 
  r2 = []
  
  # load average resp of each layer over orientation (mean tuning function)
  if load_mean_tfs==1:
    file_name =os.path.join(save_path,'All_layers_mean_TF_eval_at_ckpt-%s0000.npy'%(ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    resp_mean_all.append(np.load(file_name))
  else:
    resp_mean_all.append([])
 
  # loop over layers and load fit parameters
  for ll in range(nLayers):
                 
    if load_all_tfs==1:
      file_name =os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
      print('loading from %s\n'%file_name)
      r = np.load(file_name)
    else:
      r=[]
    resp_units.append(r)

    try:
      file_name =os.path.join(save_path,'%s_coordsHWC_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
      print('loading from %s\n'%file_name)
      coords.append(np.load(file_name))
    
      file_name= os.path.join(save_path,'%s_fit_jitter_%s_r2_each_sample_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_num[0:2]))
      print('loading from %s\n'%file_name)
      r2.append(np.load(file_name))
      
      file_name= os.path.join(save_path,'%s_fit_jitter_%s_pars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],rand_seed_str,ckpt_num[0:2]))
      print('loading from %s\n'%file_name)
      fit_pars.append(np.load(file_name))
    except:
      print('layer %s not done'%layer_labels[ll])
      continue
   
  # append onto bigger arrays
  nLayers = np.shape(r2)[0]
  resp_units_all.append(resp_units)
  coords_all.append(coords)
  r2_all.append(r2)
  fit_pars_all.append(fit_pars)

  # load additional stats about these fits parameters
  if load_stats:  
    # confidence intervals calculated for each fit parameter - all layers in one file.  
    file_name= os.path.join(save_path,'All_layers_fit_jitter_%s_CI_all_params_eval_at_ckpt_%s0000.npy'%(rand_seed_str,ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    fit_pars_CI_all.append(np.load(file_name))
    
    # confidence intervals calculated for each fit parameter - all layers in one file.  
    file_name= os.path.join(save_path,'All_layers_fit_jitter_%s_center_dist_pars_eval_at_ckpt_%s0000.npy'%(rand_seed_str,ckpt_num[0:2]))
    print('loading from %s\n'%file_name)
    center_dist_pars_all.append(np.load(file_name))
      
 
#%% make plots of r2 distributions

ii=1
plot_jitt=1
sf=0

npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
plt.close('all')
for ll in range(nLayers):
  plt.subplot(npx,npy, ll+1)

  r2_mean_each = np.mean(r2_all[ii][ll][:,sf,:],axis=1)
 
  rvals=r2_mean_each

  r_bins = np.arange(-1,1,0.05)
  plt.hist(rvals,bins=r_bins,color=colors_main[color_ind,:])
  plt.axvline(r2_cutoff,color='k')
  plt.title(layer_labels[ll])
  
  if ll==nLayers-1:
      plt.xlabel('r2')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([])  
#      plt.yticks([])

plt.suptitle('%s - init %d\n%s-%s\nFit r2'%(training_str,init_nums[ii],dataset,sf_labels[sf]))

#%% plot the proportion of units above r2 threshold, as a function of layer
ii=0
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
sf=0

# matrix to store anisotropy index for each layer    
prop_vals = np.zeros([1, np.size(layers2plot)])

# loop over network layers
for ll in range(np.size(layers2plot)):

  # values to plot    
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))   
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000

  prop_vals[0,ll] = np.sum(rvals>r2_cutoff)/np.size(rvals)
  
# put the line for this spatial frequency onto the plot      
vals = np.squeeze(prop_vals)
plt.plot(np.arange(0,np.size(layers2plot),1),vals,'-',color = colors_main[color_ind,:])

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
plt.suptitle('Prop of units with r2>%.2f\n%s-%s-init %d\n%s'%(r2_cutoff,training_strs[tr],sf_labels[sf],init_nums[ii],dataset))  
fig.set_size_inches(10,7)

#%% Plot K versus the center (scatter w transparency)
# subset of layers to save
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]
maxpts = 10000 # downsampling a bit here to make the figure tractable to save
pp2plot=1
ppname='k'
ylims = [-5,500]
sf=0
layers2plot=[18]
layers2plot=[0,6,12,18]
#alpha_vals = [0.8,0.8, 0.8, 0.8]
alpha_vals=[1,1,1,1]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')

plt.figure()
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
 
  rvals = deepcopy(np.squeeze(np.mean(r2_all[layers2plot[ll]][:,sf,:],axis=1)))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  inds2use = np.where(rvals>r2_cutoff)[0]
  # get values of center and other parameter of interest
  cvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][inds2use,sf,0]))
  parvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][inds2use,sf,pp2plot]))    
 
#  my_alpha = 1/np.size(cvals)*50
  my_alpha=alpha_vals[ll]
  print(np.size(cvals))
  if np.size(cvals)>maxpts:
    inds2plot = np.random.randint(0,np.size(cvals),maxpts)
  else:
    inds2plot = np.arange(0,np.size(cvals))
  plt.plot(cvals[inds2plot],parvals[inds2plot],'.',markersize=1,color=colors_main[color_ind,:],alpha=my_alpha,zorder=100)

  plt.title(layer_labels[layers2plot[ll]])
  plt.ylim(ylims)
  plt.xlabel('Center (deg)')
  plt.ylabel(ppname)
  plt.xticks(np.arange(0,181,45))
  
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
  
plt.suptitle('%s\n%s\n%s versus center'%(training_str,dataset,ppname))

figname = os.path.join(figfolder, '%s_%s.pdf' % (training_str,ppname))
#plt.savefig(figname, format='pdf',transparent=True)
#figname = os.path.join(figfolder, '%s_%s.epsc' % (training_str,ppname))
#plt.savefig(figname, format='eps',transparent=True)


#%% make plots of center distributions, across all units
# big plot with all layers
ii=0
plt.rcParams.update({'font.size': 10})
sf=0
# bins for the histograms
ori_bin_size=1
my_bins = np.arange(-ori_bin_size/2,nOri+0.5+ori_bin_size,ori_bin_size)

rad_lim = 500;
#cols = cm.Blues(np.linspace(0,1,nSamples))
#r2_here=0
r2_here=r2_cutoff

layers2plot=np.arange(0,nLayers)
#layers2plot=np.arange(16,21,1)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # values to plot  
  vals = deepcopy(np.squeeze(fit_pars_all[ii][layers2plot[ll]][:,sf,0]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))
 
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  
  vals=vals[np.where(rvals>r2_here)[0]]
  

#  print(np.size(vals))
  
#  if np.size(vals)>800:
#    hr_stats = []
#    print('\nlayer %d'%(ll))
#    for xx in range(5):
#      
#      vals_subsampled = vals[np.random.randint(0,np.size(vals),800)]
#      hr_stat = circ_reg_tools.hermans_rasson_stat(vals_subsampled/180*2*np.pi)
#      hr_stats.append(hr_stat)
#      print('    stat=%.2f'%hr_stat)
#    hr_stat = np.mean(hr_stat)
#  else:
#    hr_stat=circ_reg_tools.hermans_rasson_stat(vals/180*2*np.pi)
#    
  h=plt.hist(vals,bins=my_bins,color=colors_main[color_ind,:])
#  
#  num_per_bin=h[0]
#  bin_centers=h[1][0:np.size(num_per_bin)]+ori_bin_size/2
#  bin_centers = bin_centers/180*2*np.pi
#  dKL = circ_reg_tools.divergence_from_uniform(bin_centers, num_per_bin)
  
  
  plt.title('%s'%(layer_labels[layers2plot[ll]])) 
#  plt.title('%s - dKL=%.2f'%(layer_labels[layers2plot[ll]], dKL))  
#  plt.title('%s - HR=%.2f'%(layer_labels[layers2plot[ll]], hr_stat))  
  
  if ll==nLayers-1:
      plt.xlabel('Orientation (deg)')
      plt.ylabel('Num Units')
      plt.xticks(np.arange(0,nOri+1,45))
  else:
      plt.xticks([]) 
      

plt.suptitle('%s - init %d\n%s-%s\nFit Centers - All units r2>%.2f'%(training_str,init_nums[ii],dataset,sf_labels[sf],r2_here));
plt.rcParams['figure.figsize']=[18,10]

#figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% make plots of center distributions, across all units - combine network initializations
# big plot with all layers
ii=0
plt.rcParams.update({'font.size': 10})
sf=0
# bins for the histograms
ori_bin_size=1
# these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2

rad_lim = 500;
#cols = cm.Blues(np.linspace(0,1,nSamples))
#r2_here=0
r2_here=r2_cutoff
ylims = [0, 0.08]

layers2plot=np.arange(0,nLayers)
#layers2plot=np.arange(16,21,1)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # values to plot - combine centers from all network initializations into one big distribution
  vals_all = []
  for ii in range(nInits):
    vals = deepcopy(np.squeeze(fit_pars_all[ii][layers2plot[ll]][:,sf,0]))
    rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][layers2plot[ll]][:,sf,:],axis=1)))   
    # there are a few nans in here, putting a tiny value so it won't throw an error
    rvals[np.isnan(rvals)] = -1000    
    vals=vals[np.where(rvals>r2_here)[0]]
    
    vals_all = np.concatenate((vals_all,vals),axis=0)

  vals_all = np.ravel(vals_all)
  h = np.histogram(vals_all, ori_bins) # h[0] is the number per bin, h[1] is the bin edges
  # divide by total to get a proportion.
  real_y = h[0]/np.sum(h[0])
  
  plt.bar(bin_centers, real_y,width=ori_bin_size,color=colors_main[color_ind,:],zorder=100)
  
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
  plt.xlim([np.min(ori_bins), np.max(ori_bins)])
  plt.ylim(ylims)
  plt.title('%s'%(layer_labels[layers2plot[ll]])) 
 
  if ll==nLayers-1:
      plt.xlabel('Orientation (deg)')
      plt.ylabel('Prop. Units')
      plt.xticks(np.arange(0,nOri+1,45))
  else:
      plt.xticks([]) 
      

plt.suptitle('%s - %s\n%s-%s\nFit Centers - All units r2>%.2f'%(training_str,init_str,dataset,sf_labels[sf],r2_here));
plt.rcParams['figure.figsize']=[18,10]

#figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% plot average response to each orientation across entire layers
# big plot with all layers

plt.rcParams.update({'font.size': 10})
sf=0
ii=0
layers2plot=np.arange(0,np.shape(resp_mean_all)[1])
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  data = resp_mean_all[ii][ll][:,sf,:]
  
  meanvals = np.mean(data,axis=0)
  sdvals=np.std(data,axis=0)
  
  plt.errorbar(ori_axis,meanvals,sdvals,color='k',ecolor=colors_main[color_ind,:])
 
  plt.title('%s'%(layer_labels[layers2plot[ll]])) 
  
  if ll==nLayers-1:
      plt.xlabel('Orientation (deg)')
      plt.ylabel('Avg response')
      plt.xticks(np.arange(0,nOri+1,45))
  else:
      plt.xticks([]) 
      

plt.suptitle('%s - init %d\n%s-%s\nAverage response, all units'%(training_str,init_nums[ii],dataset,sf_labels[sf]));
plt.rcParams['figure.figsize']=[18,10]

#figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% plot Fisher information based on average response of the layer only
# big plot with all layers
import classifiers_custom as classifiers
plt.rcParams.update({'font.size': 10})
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  data = resp_mean[ll][:,sf,:]
  # treat the 4 samples as different repetitions to get a variance estimate
  datlong=np.reshape(data, [nSamples*nOri, 1])
  labslong = np.tile(np.expand_dims(ori_axis,axis=1), [nSamples,1])
  # get fisher info based on this mean signal as one "unit"
  ori_axis_out, fi, d, v = classifiers.get_fisher_info(datlong,labslong,delta=4)
  
  plt.plot(ori_axis_out,fi,color=colors_main[color_ind,:])
 
  plt.title('%s'%(layer_labels[layers2plot[ll]])) 

  if ll==nLayers-1:
      plt.xlabel('Orientation (deg)')
      plt.ylabel('Avg response')
      plt.xticks(np.arange(0,nOri+1,45))
  else:
      plt.xticks([]) 
      

plt.suptitle('%s\n%s-%s\nFisher information from average response'%(training_str,dataset,sf_labels[sf]));
plt.rcParams['figure.figsize']=[18,10]

#figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% make plots of center distributions, subset of layers to save
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]
exp_peak_ind=2
mu_pairs = [[0, 90], [22.5, 112.5], [45, 135], [67.5, 157.5]]
sf=0
ylims=[0,0.05]

# define bins to use the for the distribution
ori_bin_size=1
# these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2

layers2plot=[0,6,12,18]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # values to plot  
  cvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,0]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[layers2plot[ll]][:,sf,:],axis=1)))
 
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000

  cvals=cvals[np.where(rvals>r2_cutoff)[0]]
  # get the actual curve that describes the distribution of centers
  h = np.histogram(cvals, ori_bins) # h[0] is the number per bin, h[1] is the bin edges
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
  
    
plt.suptitle('%s\n%s-%s\nFit Centers - All units r2>%.2f'%(training_str,dataset,sf_labels[sf],r2_here));
plt.rcParams['figure.figsize']=[18,10]

figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
plt.savefig(figname, format='pdf',transparent=True)

#%% make plots of center distributions, with bimodal Von-Mises curve fits overlaid
# big plot with all layers
plt.rcParams.update({'font.size': 10})
plot_jitt=1
exp_peak_ind=0

pp=0  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
# define bins to use the for the distribution
ori_bin_size=1
# these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2

r2_here=r2_cutoff

layers2plot=np.arange(0,nLayers)
#layers2plot=np.arange(16,21,1)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # values to plot  
  vals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,pp]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[layers2plot[ll]][:,sf,:],axis=1)))
 
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  
  xvals = coords_all[layers2plot[ll]][:,1]
  yvals = coords_all[layers2plot[ll]][:,0]
  center = info['activ_dims'][layers2plot[ll]]/2
  rad_vals = np.sqrt(np.power(xvals-center,2)+np.power(yvals-center,2))

  cvals=vals[np.where(np.logical_and(rvals>r2_here, rad_vals<rad_lim))[0]]
  # get the actual curve that describes the distribution of centers
  h = np.histogram(cvals, ori_bins) # h[0] is the number per bin, h[1] is the bin edges
  # divide by total to get a proportion.
  real_y = h[0]/np.sum(h[0])

  # the best-fit curve to this distribution
  params = np.squeeze(center_dist_pars_all[ll,exp_peak_ind,:])
  if params[7]>0:
    pred_y = analyze_fit_params.double_von_mises_deg(bin_centers, params[0],params[1],params[2],params[3],params[4],params[5],params[6])
    plt.plot(bin_centers, pred_y,color=[0,0,0])
  
  plt.bar(bin_centers, real_y,width=ori_bin_size,color=colors_all[3,color_ind,:])
 
         
  plt.title('%s - r2=%.2f'%(layer_labels[layers2plot[ll]], params[7])) 

#  plt.ylim([0,0.10])
  if ll==nLayers-1:
      plt.xlabel('Orientation (deg)')
      plt.ylabel('Proportion Units')
      plt.xticks(np.arange(0,nOri+1,45))
  else:
      plt.xticks([]) 
      

plt.suptitle('%s\n%s-%s\nFit Centers + best fit curves- All units r2>%.2f'%(training_str,dataset,sf_labels[sf],r2_here));
plt.rcParams['figure.figsize']=[18,10]


#%% plot center distribution K parameter for each layer
mu_pairs = [[0, 90], [22.5, 112.5], [45, 135], [67.5, 157.5]]
exp_peak_ind=0
r2_cutoff=0.4
plt.rcParams.update({'font.size': 10})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles=[]
layers2plot =np.arange(0,nLayers,1)

# get the parameter values to plot
rvals = np.squeeze(center_dist_pars_all[:,exp_peak_ind,7])
parvals = np.squeeze(deepcopy(center_dist_pars_all[:,exp_peak_ind,2:4]))
parvals[rvals<r2_cutoff] = np.nan

# put two lines onto the plot
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),parvals[:,0],color = colors_all[4,color_ind,:])
ax.add_line(myline)   
handles.append(myline)
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),parvals[:,1],color = colors_all[2,color_ind,:])
ax.add_line(myline)   
handles.append(myline)
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),np.mean(parvals,axis=1),color = [0,0,0])
ax.add_line(myline)   
handles.append(myline)

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['peak at %d deg'%mu_pairs[exp_peak_ind][0], 'peak at %d deg'%mu_pairs[exp_peak_ind][1], 'average'])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Concentration param (k)')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Concentration parameter from best fit to center distribution\n%s-%s\n%s'%(training_strs[tr],sf_labels[sf],dataset))  
fig.set_size_inches(10,7)

#%% plot center distribution amplitude parameter for each layer
mu_pairs = [[0, 90], [22.5, 112.5], [45, 135], [67.5, 157.5]]
exp_peak_ind=0
r2_cutoff=0.4
plt.rcParams.update({'font.size': 10})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles=[]
layers2plot =np.arange(0,nLayers,1)

# get the parameter values to plot
rvals = np.squeeze(center_dist_pars_all[:,exp_peak_ind,7])
parvals = np.squeeze(deepcopy(center_dist_pars_all[:,exp_peak_ind,4:6]))
parvals[rvals<r2_cutoff] = np.nan

# put two lines onto the plot
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),parvals[:,0],color = colors_all[4,color_ind,:])
ax.add_line(myline)   
handles.append(myline)
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),parvals[:,1],color = colors_all[2,color_ind,:])
ax.add_line(myline)   
handles.append(myline)
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),np.mean(parvals,axis=1),color = [0,0,0])
ax.add_line(myline)   
handles.append(myline)

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['peak at %d deg'%mu_pairs[exp_peak_ind][0], 'peak at %d deg'%mu_pairs[exp_peak_ind][1], 'average'])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Amplitude (prop of units)')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Amplitude from best fit to center distribution\n%s-%s\n%s'%(training_strs[tr],sf_labels[sf],dataset))  
fig.set_size_inches(10,7)

#%% plot center distribution baseline parameter for each layer
mu_pairs = [[0, 90], [22.5, 112.5], [45, 135], [67.5, 157.5]]
exp_peak_ind=0
r2_cutoff=0.4
plt.rcParams.update({'font.size': 10})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles=[]
layers2plot =np.arange(0,nLayers,1)

# get the parameter values to plot
rvals = np.squeeze(center_dist_pars_all[:,exp_peak_ind,7])
parvals = np.squeeze(deepcopy(center_dist_pars_all[:,exp_peak_ind,6]))
parvals[rvals<r2_cutoff] = np.nan

# put two lines onto the plot
#myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),parvals[:,0],color = colors_all[4,color_ind,:])
#ax.add_line(myline)   
#handles.append(myline)
#myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),parvals[:,1],color = colors_all[2,color_ind,:])
#ax.add_line(myline)   
#handles.append(myline)
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),parvals,color = [0,0,0])
ax.add_line(myline)   
handles.append(myline)

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

#plt.legend(['peak at %d deg'%mu_pairs[exp_peak_ind][0], 'peak at %d deg'%mu_pairs[exp_peak_ind][1], 'average'])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Baseline (prop of units)')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Baseline from best fit to center distribution\n%s-%s\n%s'%(training_strs[tr],sf_labels[sf],dataset))  
fig.set_size_inches(10,7)

#%% plot center distribution R2 parameter for each layer
mu_pairs = [[0, 90], [22.5, 112.5], [45, 135], [67.5, 157.5]]
exp_peak_inds=[0,1,2,3]
color_order = [1,0,3,2]
r2_cutoff=0.4
plt.rcParams.update({'font.size': 10})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles=[]
layers2plot =np.arange(0,nLayers,1)

for pp in range(np.size(exp_peak_inds)):
  # get the parameter values to plot
  rvals = np.squeeze(center_dist_pars_all[:,exp_peak_inds[pp],7])
  
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),rvals,color = colors_all[3,color_order[pp]])
  ax.add_line(myline)   
  handles.append(myline)

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['peaks at %d and %d deg'%(mu_pairs[pp][0],mu_pairs[pp][1]) for pp in range(np.size(exp_peak_inds))])

plt.plot(xlims, [0,0], 'k')
plt.plot(xlims, [r2_cutoff, r2_cutoff],color=[0.8, 0.8, 0.8])
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('r2')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('r2 from best fit to center distribution\n%s-%s\n%s'%(training_strs[tr],sf_labels[sf],dataset))  
fig.set_size_inches(10,7)

#%% Plot each fit parameter versus the center (scatter w transparency)
# plot all layers
plt.rcParams.update({'font.size': 10})
plot_jitt=1
pp2plot = [0]
ppinds = [1,2,3,4]
#pp2plot = [1]
ppnames = ['k','amp','baseline','fwhm']
#ylims = [[0,10],[-1,1],[-1,1],[0,91]]
ylims = [[0,150],[-1,20],[-1,10],[0,91]]
tr=0
sf=0
#nbins_pars = 30;
layers2plot=np.arange(0,nLayers)
# going to bin the units by their centers, then get CI of each other parameter within each center bin.
ori_bin_size=10
# these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2
r2_here=r2_cutoff

npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')

cols = cm.Blues(np.linspace(0,1,nSamples))

for pp in range(np.size(pp2plot)):
  plt.figure()
  for ll in range(np.size(layers2plot)):
    plt.subplot(npx,npy, ll+1)
   
    rvals = deepcopy(np.squeeze(np.mean(r2_all[layers2plot[ll]][:,sf,:],axis=1)))
    # there are a few nans in here, putting a tiny value so it won't throw an error
    rvals[np.isnan(rvals)] = -1000
    inds2use = np.where(rvals>r2_cutoff)[0]
    # get values of center and other parameter of interest
    cvals = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,0]))
    parvals = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,ppinds[pp2plot[pp]]]))    
 
    plt.plot(cvals,parvals,'.',markersize=1,color=colors_all[3,color_ind,:],alpha=0.05)

    plt.title(layer_labels[layers2plot[ll]])
    plt.ylim(ylims[pp2plot[pp]])
    if ll==nLayers-1:
      plt.xlabel('Center (deg)')
      plt.ylabel(ppnames[pp2plot[pp]])
      plt.xticks(np.arange(0,181,45))
    else:
      plt.xticks([]) 
#      plt.yticks([])
  
  plt.suptitle('%s\n%s\n%s versus center'%(training_str,dataset,ppnames[pp2plot[pp]]))
 
#%% Plot each fit parameter versus the center (resampled Mean +/- CI)
# plot all layers
plt.rcParams.update({'font.size': 10})
plot_jitt=1
pp2plot = [0,1,2,3]
ppinds = [1,2,3,4]
#pp2plot = [1]
ppnames = ['k','amp','baseline','fwhm']
#ylims = [[0,10],[-1,1],[-1,1],[0,91]]
ylims = [[0,150],[-1,20],[-1,10],[0,91]]
tr=0
sf=0
#nbins_pars = 30;
layers2plot=np.arange(0,nLayers)
# going to bin the units by their centers, then get CI of each other parameter within each center bin.
ori_bin_size=11.25
# these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2
r2_here=r2_cutoff

npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')

cols = cm.Blues(np.linspace(0,1,nSamples))

for pp in range(np.size(pp2plot)):
  plt.figure()
  for ll in range(np.size(layers2plot)):
    plt.subplot(npx,npy, ll+1)
#    for xx in np.arange(0,181,45):
#       plt.axvline(xx,color=[0.8, 0.8, 0.8])
    civals = np.squeeze(fit_pars_CI_all[ll,pp2plot[pp],:,:])
    meanvals=np.expand_dims(civals[:,0],axis=0)
    errvals=np.abs(np.tile(meanvals,[2,1])-np.transpose(civals[:,1:3]))
#    plt.plot(bin_centers, civals[:,0],color='k')
    plt.errorbar(bin_centers, np.squeeze(meanvals), errvals, ecolor=colors_all[2,color_ind,:],color=[0,0,0],zorder=100)

    minval = np.min(civals[~np.isnan(civals[:,1]),1])
    maxval = np.max(civals[~np.isnan(civals[:,2]),2])
    plt.ylim([minval - (maxval-minval)/10,maxval + (maxval-minval)/10])
    plt.title(layer_labels[layers2plot[ll]])
#    plt.ylim(ylims[pp2plot[pp]])
    if ll==nLayers-1:
      plt.xlabel('Center (deg)')
      plt.ylabel(ppnames[pp2plot[pp]])
      plt.xticks(np.arange(0,181,45))
    else:
      plt.xticks([]) 
#      plt.yticks([])
  
  plt.suptitle('%s\n%s\n%s versus center\n(resampled mean +/- 95pct CI)'%(training_str,dataset,ppnames[pp2plot[pp]]))


#%% plot anisotropy based on each fit parameter, as a function of layer

tr=0
plt.rcParams.update({'font.size': 10})
plot_jitt=1
pp2plot = [0,1,2,3]
ppinds = [1,2,3,4]
#pp2plot = [1]
ppnames = ['k','amp','baseline','fwhm']
ori_bin_size=11.25
# these describe all the edges of the bins - so the leftmost value is left edge of first bin, rightmost value is the right edge of the final bin.
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2
#% define the orientation bins of interest
b = np.arange(22.5,nOri,90) # baseline
binds = [ii for ii in range(np.size(bin_centers)) if bin_centers[ii] in b]
#binds = [ii for ii in range(np.size(bin_centers)) if bin_centers[ii] not in t and bin_centers[ii] not in o and bin_centers[ii] not in c]
t = np.arange(67.5,nOri,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
tinds = [ii for ii in range(np.size(bin_centers)) if bin_centers[ii] in t]
c = np.arange(0,nOri+1,90) # cardinals
cinds = [ii for ii in range(np.size(bin_centers)) if bin_centers[ii] in c]
o = np.arange(45,nOri,90)  # obliques
oinds = [ii for ii in range(np.size(bin_centers)) if bin_centers[ii] in o]
#bin_size = 10
#bin_size=20

peak_inds=[cinds,tinds,oinds]

plt.close('all')

layers2plot = np.arange(0,nLayers,1)
ylims = [-1,1]
xlims = [-1, np.size(layers2plot)]
  
for pp in range(np.size(pp2plot)):
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  handles = []
  # loop over network training schemes (upright versus rot images etc)
  for cc in range(np.shape(peak_inds)[0]):
  
    # matrix to store value for each layer    
    par_ratio_vals = np.zeros([1, np.size(layers2plot)])
    
    # loop over network layers
    for ll in range(np.size(layers2plot)):
  
      civals = np.squeeze(fit_pars_CI_all[ll,pp2plot[pp],:,:])
      meanvals=civals[:,0]
#      errvals=np.abs(np.tile(meanvals,[2,1])-np.transpose(civals[:,1:3]))
      peak_vals=meanvals[peak_inds[cc]]
      base_vals=meanvals[binds]
      
      par_ratio_vals[0,ll] = (np.mean(peak_vals) - np.mean(base_vals))/(np.mean(peak_vals) + np.mean(base_vals))
        
    # put the line for this spatial frequency onto the plot      
    vals = par_ratio_vals
  #  errvals = np.std(aniso_vals,0)
    myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = colors_all[3,cc+1,:])
    ax.add_line(myline)   
    handles.append(myline)
#  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=colors_all[3,pp+1,:])

  
  # finish up this subplot 
  
  plt.legend(['0 + 90', '67.5 + 157.5', '45 + 135'])
  
  plt.plot(xlims, [0,0], 'k')
  plt.xlim(xlims)
#  plt.ylim(ylims)
  plt.yticks([-1,-0.5,0, 0.5,1])
  plt.ylabel('Parameter-based anisotropy')
  plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
  
  # finish up the entire plot
  plt.suptitle('Relative value of %s\n%s-%s\n%s'%(ppnames[pp2plot[pp]],training_strs[tr],sf_labels[sf],dataset))  
  fig.set_size_inches(10,7)
  #figname = os.path.join(figfolder, 'AnisoEachType_%s.pdf'%training_strs[tr])
#plt.savefig(figname, format='pdf',transparent=True)
 
#%% plot spatial position of all responsive units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
pp=0;
alpha=0.1
r2_here=r2_cutoff
#r2_here=0
# use these metrics to get rid of poorly-fit units

rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))

# there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000

# choose just the units of interest here
units_good = np.where(rvals>r2_here)[0]

#units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]

# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll][:,1]
yvals = coords_all[ll][:,0]

plt.subplot(1,2,1)
plt.plot(xvals,yvals,'.',color=colors_all[3,color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('all responsive units')

plt.subplot(1,2,2)
plt.plot(xvals[units_good],yvals[units_good],'.',color=colors_all[3,color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('well-fit units (r2>%.2f)'%r2_here)

plt.suptitle('%s\n%s-%s\n%s'%(training_str,dataset,sf_labels[sf],layer_labels[ll]))


#%% plot spatial position of all cardinal-tuned units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
pp=0;
alpha=0.1
r2_here=0
#r2_here=r2_cutoff

# values to plot  
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_here)[0]
centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
amps = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))

horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll][units_good,1]
yvals = coords_all[ll][units_good,0]

plt.subplot(1,2,1)
plt.plot(xvals[horiz_units],yvals[horiz_units],'.',color=colors_all[3,color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('Horizontal-tuned units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.subplot(1,2,2)
plt.plot(xvals[vert_units],yvals[vert_units],'.',color=colors_all[3,color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('Vertical-tuned units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.suptitle('%s\n%s-%s\n%s'%(training_str,dataset,sf_labels[sf],layer_labels[ll]))


#%% plot feature map position of all cardinal-tuned units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
pp=0;
alpha=0.1

# make sure this isn't a pooling layer
assert(ll in w_layer_inds)
r2_here=0
#r2_here=r2_cutoff

# values to plot
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_here)[0]
centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
amps = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))

  
horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll][units_good,1]
yvals = coords_all[ll][units_good,0]
cvals = coords_all[ll][units_good,2]

plt.subplot(1,2,1)
# bins for the histograms
my_bins = np.arange(0,info['output_chans'][ll]+1,1)
# make the histogram  
plt.hist(cvals[horiz_units],bins=my_bins,color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Num units')
plt.title('Horizontal-tuned units')

plt.subplot(1,2,2)
# bins for the histograms
my_bins = np.arange(0,info['output_chans'][ll]+1,1)
# make the histogram  
plt.hist(cvals[vert_units],bins=my_bins,color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Num units')
plt.title('Vertical-tuned units')


plt.suptitle('%s\n%s-%s\n%s'%(training_str,dataset,sf_labels[sf],layer_labels[ll]))


#%% plot feature map position of all 45-tuned units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=19;
sf=0;
pp=0;
alpha=0.1

# make sure this isn't a pooling layer
assert(ll in w_layer_inds)
#r2_here=0
r2_here=r2_cutoff

# values to plot
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_here)[0]
centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
amps = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))

  
obl45_units = np.where(np.logical_and(np.logical_and(centers>40, centers<50), amps>0))[0]
obl135_units = np.where(np.logical_and(np.logical_and(centers>130, centers<140), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll][units_good,1]
yvals = coords_all[ll][units_good,0]
cvals = coords_all[ll][units_good,2]

plt.subplot(1,2,1)
# bins for the histograms
my_bins = np.arange(0,info['output_chans'][ll]+1,1)
# make the histogram  
plt.hist(cvals[obl45_units],bins=my_bins,color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Num units')
plt.title('45-tuned units')

plt.subplot(1,2,2)
# bins for the histograms
my_bins = np.arange(0,info['output_chans'][ll]+1,1)
# make the histogram  
plt.hist(cvals[obl135_units],bins=my_bins,color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Num units')
plt.title('135-tuned units')


plt.suptitle('%s\n%s-%s\n%s'%(training_str,dataset,sf_labels[sf],layer_labels[ll]))


#%% plot weights that map between layers
plt.close('all')
plt.figure();
tr=0;
ll1=0;
ll2=1;

sf=0;
pp=0;
alpha=0.1
# make sure this isn't a pooling layer
assert(ll1 in w_layer_inds and ll2 in w_layer_inds)

# first plot the channel distribution of cardinal-tuned units in the shallower layer
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll1][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_cutoff)[0]

centers = deepcopy(np.squeeze(fit_pars_all[ll1][units_good,sf,0]))
amps =  deepcopy(np.squeeze(fit_pars_all[ll1][units_good,sf,2]))
horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll1][units_good,1]
yvals = coords_all[ll1][units_good,0]
cvals = coords_all[ll1][units_good,2]

plt.subplot(3,2,1)
# bins for the histograms
my_bins = np.arange(0,info['output_chans'][ll1]+1,1)
# make the histogram  
plt.hist(cvals[horiz_units],bins=my_bins,color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Num units')
plt.title('Horizontal-tuned units, layer %s'%layer_labels[ll1])

plt.subplot(3,2,2)
# bins for the histograms
my_bins = np.arange(0,info['output_chans'][ll1]+1,1)
# make the histogram  
plt.hist(cvals[vert_units],bins=my_bins,color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Num units')
plt.title('Vertical-tuned units, layer %s'%layer_labels[ll1])


# now find units of interest in the deeper layer
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll2][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_cutoff)[0]

centers = deepcopy(np.squeeze(fit_pars_all[ll2][units_good,sf,0]))
amps =  deepcopy(np.squeeze(fit_pars_all[ll2][units_good,sf,2]))
horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll2][units_good,1]
yvals = coords_all[ll2][units_good,0]
cvals = coords_all[ll2][units_good,2]

# find channel in deep layer that has the most horizontally-tuned units
my_bins = np.arange(0,info['output_chans'][ll2]+1,1)
h=np.histogram(cvals[horiz_units],bins=my_bins)
sortinds=np.flip(np.argsort(h[0]))
c_max = sortinds[0]

# now get the weights for that layer's readout from previous layer feature maps
weights_l2l=w_all[ll2][:,:,:,c_max]
#weights_l2l = np.squeeze(np.mean(np.mean(weights_l2l,1),0))
weights_l2l = np.reshape(weights_l2l,[9,np.shape(weights_l2l)[2]])
plt.subplot(3,2,3)
plt.plot(np.arange(0,np.shape(weights_l2l)[1],1), np.transpose(weights_l2l))
plt.plot(np.arange(0,np.shape(weights_l2l)[1],1), np.mean(weights_l2l,0),color='k')
#, color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Weight')
plt.title('Weights from all %s feature maps to channel %d in %s'%(layer_labels[ll1],c_max,layer_labels[ll2]))

#  now find channel in deep layer that has the most vertically-tuned units
my_bins = np.arange(0,info['output_chans'][ll2]+1,1)
h=np.histogram(cvals[vert_units],bins=my_bins)
sortinds=np.flip(np.argsort(h[0]))
c_max = sortinds[0]

# now get the weights for that layer's readout from previous layer feature maps
weights_l2l=w_all[ll2][:,:,:,c_max]
#weights_l2l = np.squeeze(np.mean(np.mean(weights_l2l,1),0))
weights_l2l = np.reshape(weights_l2l,[9,np.shape(weights_l2l)[2]])
plt.subplot(3,2,4)
plt.plot(np.arange(0,np.shape(weights_l2l)[1],1), np.transpose(weights_l2l))
plt.plot(np.arange(0,np.shape(weights_l2l)[1],1), np.mean(weights_l2l,0),color='k')
#, color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Weight')
plt.title('Weights from all %s feature maps to channel %d in %s'%(layer_labels[ll1],c_max,layer_labels[ll2]))


# now plot dist of channels over deeper layer
plt.subplot(3,2,5)
# bins for the histograms
my_bins = np.arange(0,info['output_chans'][ll2]+1,1)
# make the histogram  
plt.hist(cvals[horiz_units],bins=my_bins,color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Num units')
plt.title('Horizontal-tuned units, layer %s'%layer_labels[ll2])



plt.subplot(3,2,6)
# bins for the histograms
my_bins = np.arange(0,info['output_chans'][ll2]+1,1)
# make the histogram  
plt.hist(cvals[vert_units],bins=my_bins,color=colors_all[3,color_ind,:])
plt.xlabel('Channel number')
plt.ylabel('Num units')
plt.title('Vertical-tuned units, layer %s'%layer_labels[ll2])


plt.suptitle('%s\n%s'%(training_str,dataset))


#%% plot spatial filters for some horizontal-tuned units

plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
pp=0;
alpha=0.1
input_chan=0
#my_clims=[-0.15, 0.15]
my_clims=[-3,3]
ninputchan2plot=3
noutputchan2plot=2

# make sure this isn't a pooling layer
assert(tr in w_layer_inds)

# use these metrics to get rid of poorly-fit units

rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
# there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_cutoff)[0]

centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]

# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll][units_good,1]
yvals = coords_all[ll][units_good,0]
cvals = coords_all[ll][units_good,2]

# find channels that have the most horizontally-tuned units
my_bins = np.arange(0,info['output_chans'][ll]+1,1)
h=np.histogram(cvals[horiz_units],bins=my_bins)
sortinds=np.flip(np.argsort(h[0]))
c_max = sortinds[0:noutputchan2plot]

# plot filters from these channels  
for ii in range(ninputchan2plot):
  
  for ff in range(noutputchan2plot):
    plt.subplot(ninputchan2plot,noutputchan2plot,ff+(ii*noutputchan2plot)+1)
    w_ind= np.where(w_layer_inds==ll)[0][0]
    my_filter = np.squeeze(w_all[w_ind][:,:,ii,c_max[ff]])
   
    plt.pcolormesh(my_filter)
    plt.axis('square')
    plt.clim(my_clims)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('Input channel %d, output channel %d'%(ii, c_max[ff]))
    plt.xlabel('Horizontal dimension')
    plt.ylabel('Vertical dimension')
   

plt.suptitle('Example 3x3 spatial filters from horizontally-tuned units\n%s\n%s\n%s'%(training_str,dataset,layer_labels[ll]))

#%% plot spatial filters for some vertically-tuned units

plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
pp=0;
alpha=0.1
input_chan=0
#my_clims=[-0.15, 0.15]
my_clims=[-3,3]
ninputchan2plot=3
noutputchan2plot=2
r2_here=0
# make sure this isn't a pooling layer
assert(tr in w_layer_inds)

# use these metrics to get rid of poorly-fit units

rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
# there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_here)[0]

centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]

# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll][units_good,1]
yvals = coords_all[ll][units_good,0]
cvals = coords_all[ll][units_good,2]

# find channels that have the most horizontally-tuned units
my_bins = np.arange(0,info['output_chans'][ll]+1,1)
h=np.histogram(cvals[vert_units],bins=my_bins)
sortinds=np.flip(np.argsort(h[0]))
c_max = sortinds[0:noutputchan2plot]

# plot filters from these channels  
for ii in range(ninputchan2plot):
  
  for ff in range(noutputchan2plot):
    plt.subplot(ninputchan2plot,noutputchan2plot,ff+(ii*noutputchan2plot)+1)
    w_ind= np.where(w_layer_inds==ll)[0][0]
    my_filter = np.squeeze(w_all[w_ind][:,:,ii,c_max[ff]])
   
    plt.pcolormesh(my_filter)
    plt.axis('square')
    plt.clim(my_clims)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('Input channel %d, output channel %d'%(ii, c_max[ff]))
    plt.xlabel('Horizontal dimension')
    plt.ylabel('Vertical dimension')
   

plt.suptitle('Example 3x3 spatial filters from vertically-tuned units\n%s\n%s\n%s'%(training_str,dataset,layer_labels[ll]))

cvals = coords_all[ll][units_good,2]



#%% plot spatial position of cardinal-tuned units within particular channels at a time

plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
pp=0;
alpha=0.1
noutputchan2plot=2
# use these metrics to get rid of poorly-fit units
fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
#  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
# there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
fvals[np.isnan(rvals)] = -1000
# choose just the units of interest here
units_good = np.where(rvals>r2_cutoff)[0]

centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll][units_good,1]
yvals = coords_all[ll][units_good,0]
c_vals = coords_all[ll][units_good,2]

# find channels that have the most horizontally-tuned units
my_bins = np.arange(0,info['output_chans'][ll]+1,1)
h=np.histogram(c_vals[horiz_units],bins=my_bins)
sortinds=np.flip(np.argsort(h[0]))
c_max = sortinds[0:noutputchan2plot]

for nn in range(noutputchan2plot):
  plt.subplot(noutputchan2plot,2,(2*nn)+1)
  inds2plot = np.where(np.logical_and(np.logical_and(np.logical_and(centers>85, centers<95), amps>0), c_vals==c_max[nn]))[0]
  plt.plot(xvals[inds2plot],yvals[inds2plot],'.',color=colors_all[3,color_ind,:],markersize=2,alpha=alpha)
  plt.xlabel('horizontal unit position')
  plt.ylabel('vertical unit position')
  plt.axis('square')
  plt.title('Horizontal-tuned units - channel %d'%c_max[nn])
  plt.xlim([0,info['activ_dims'][ll]])
  plt.ylim([0,info['activ_dims'][ll]])

# find channels that have the most vertically-tuned units
my_bins = np.arange(0,info['output_chans'][ll]+1,1)
h=np.histogram(c_vals[vert_units],bins=my_bins)
sortinds=np.flip(np.argsort(h[0]))
c_max = sortinds[0:noutputchan2plot]

for nn in range(noutputchan2plot):
  plt.subplot(noutputchan2plot,2,(2*nn)+2)
  inds2plot = np.where(np.logical_and(np.logical_and(np.logical_or(centers>175, centers<5), amps>0), c_vals==c_max[nn]))[0]
  plt.plot(xvals[inds2plot],yvals[inds2plot],'.',color=colors_all[3,color_ind,:],markersize=2,alpha=alpha)
  plt.xlabel('horizontal unit position')
  plt.ylabel('vertical unit position')
  plt.axis('square')
  plt.title('Vertical-tuned units - channel %d'%c_max[nn])
  plt.xlim([0,info['activ_dims'][ll]])
  plt.ylim([0,info['activ_dims'][ll]])

plt.suptitle('%s\n%s\n%s'%(training_str,dataset,layer_labels[ll]))

#%% plot spatial position of all obliquely-tuned units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=14;
sf=0;
pp=0;
alpha=0.01
#r2_here=0
r2_here=r2_cutoff

# values to plot
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000
units_good = np.where(rvals>r2_here)[0]
centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
amps = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))

obl1_units = np.where(np.logical_and(np.logical_and(centers>40, centers<50), amps>0))[0]
obl2_units = np.where(np.logical_and(np.logical_and(centers>130, centers<140), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ll][units_good,1]
yvals = coords_all[ll][units_good,0]

plt.subplot(1,2,1)
plt.plot(xvals[obl1_units],yvals[obl1_units],'.',color=colors_all[3,color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('45-tuned units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.subplot(1,2,2)
plt.plot(xvals[obl2_units],yvals[obl2_units],'.',color=colors_all[3,color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('135-tuned units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.suptitle('%s\n%s\n%s'%(training_str,dataset,layer_labels[ll]))

#%% make plots of vertical position of all well-fit units, histogram for each layer
#pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for li in range(np.size(layers2plot)):
  plt.subplot(npx,npy, li+1)
  ll=layers2plot[li]
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
  #  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
  units_good = np.where(rvals>r2_cutoff)[0]

#  units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
  
  centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
  amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
#  horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
#  vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
  # coords goes [vertical pos, horizontal pos, output channel number]
  xvals = coords_all[ll][units_good,1]
  yvals = coords_all[ll][units_good,0]

  # bins for the histograms
  my_bins = np.arange(0,info['activ_dims'][ll]+1,1)

  # make the histogram  
  plt.hist(yvals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[ll])

  if ll==nLayers-1:
      plt.xlabel('Unit vertical position (pix)')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nVertical positions of all well-fit units (r2>%.2f)'%(training_str,dataset,r2_cutoff))


#%% make plots of horizontal position of all well-fit units, histogram for each layer
#pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for li in range(np.size(layers2plot)):
  plt.subplot(npx,npy, li+1)
  ll=layers2plot[li]
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
  #  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
  units_good = np.where(rvals>r2_cutoff)[0]

#  units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
  
  centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
  amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
#  horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
#  vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
  # coords goes [vertical pos, horizontal pos, output channel number]
  xvals = coords_all[ll][units_good,1]
  yvals = coords_all[ll][units_good,0]

  # bins for the histograms
  my_bins = np.arange(0,info['activ_dims'][ll]+1,1)

  # make the histogram  
  plt.hist(xvals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[ll])

  if ll==nLayers-1:
      plt.xlabel('Unit horizontal position (pix)')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\Horizontal positions of all well-fit units (r2>%.2f)'%(training_str,dataset,r2_cutoff))

#%% make plots of vertical position of 45-tuned units, histogram for each layer
#pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for li in range(np.size(layers2plot)):
  plt.subplot(npx,npy, li+1)
  ll=layers2plot[li]
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
  #  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
  units_good = np.where(rvals>r2_cutoff)[0]

#  units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
  
  centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
  amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
  obl45_units = np.where(np.logical_and(np.logical_and(centers>40, centers<50), amps>0))[0]
#  vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
  # coords goes [vertical pos, horizontal pos, output channel number]
  xvals = coords_all[ll][units_good,1]
  yvals = coords_all[ll][units_good,0]

  # bins for the histograms
  my_bins = np.arange(0,info['activ_dims'][ll]+1,1)

  # make the histogram  
  plt.hist(yvals[obl45_units],bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[ll])

  if ll==nLayers-1:
      plt.xlabel('Unit vertical position (pix)')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nVertical positions of 45-tuned units'%(training_str,dataset))


#%% make plots of horizontal position of 45-tuned units, histogram for each layer
#pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for li in range(np.size(layers2plot)):
  plt.subplot(npx,npy, li+1)
  ll=layers2plot[li]
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
  #  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
  units_good = np.where(rvals>r2_cutoff)[0]

#  units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
  
  centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
  amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
  obl45_units = np.where(np.logical_and(np.logical_and(centers>40, centers<50), amps>0))[0]
#  vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
  # coords goes [vertical pos, horizontal pos, output channel number]
  xvals = coords_all[ll][units_good,1]
  yvals = coords_all[ll][units_good,0]

  # bins for the histograms
  my_bins = np.arange(0,info['activ_dims'][ll]+1,1)

  # make the histogram  
  plt.hist(xvals[obl45_units],bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[ll])

  if ll==nLayers-1:
      plt.xlabel('Unit horizontal position (pix)')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nHorizontal positions of 45-tuned units'%(training_str,dataset))


#%% make plots of vertical position of horizontal-tuned units, histogram for each layer
#pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for li in range(np.size(layers2plot)):
  plt.subplot(npx,npy, li+1)
  ll=layers2plot[li]
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
  #  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
  units_good = np.where(rvals>r2_cutoff)[0]

#  units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
  
  centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
  amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
  horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
  vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
  # coords goes [vertical pos, horizontal pos, output channel number]
  xvals = coords_all[ll][units_good,1]
  yvals = coords_all[ll][units_good,0]

  # bins for the histograms
  my_bins = np.arange(0,info['activ_dims'][ll]+1,1)

  # make the histogram  
  plt.hist(yvals[horiz_units],bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[ll])

  if ll==nLayers-1:
      plt.xlabel('Unit vertical position (pix)')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nVertical positions of horizontally-tuned units'%(training_str,dataset))

#%% make plots of horizontal position of horizontal-tuned units, histogram for each layer
#pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for li in range(np.size(layers2plot)):
  plt.subplot(npx,npy, li+1)
  ll=layers2plot[li]
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
  #  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
  units_good = np.where(rvals>r2_cutoff)[0]

#  units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
  
  centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
  amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
  horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
  vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
  # coords goes [vertical pos, horizontal pos, output channel number]
  xvals = coords_all[ll][units_good,1]
  yvals = coords_all[ll][units_good,0]

  # bins for the histograms
  my_bins = np.arange(0,info['activ_dims'][ll]+1,1)

  # make the histogram  
  plt.hist(xvals[horiz_units],bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[ll])

  if ll==nLayers-1:
      plt.xlabel('Unit horizontal position (pix)')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nHorizontal positions of horizontally-tuned units'%(training_str,dataset))


#%% make plots of horizontal position of vertical-tuned units, histogram for each layer
pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for li in range(np.size(layers2plot)):
  plt.subplot(npx,npy, li+1)
  ll=layers2plot[li]
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
  #  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
  units_good = np.where(rvals>r2_cutoff)[0]

#  units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
  
  centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
  amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
  horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
  vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
  # coords goes [vertical pos, horizontal pos, output channel number]
  xvals = coords_all[ll][units_good,1]
  yvals = coords_all[ll][units_good,0]

  # bins for the histograms
  my_bins = np.arange(0,info['activ_dims'][ll]+1,1)

  # make the histogram  
  plt.hist(xvals[vert_units],bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[ll])
#  plt.ylim([0,10])
  if ll==nLayers-1:
      plt.xlabel('Unit horizontal position (pix)')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nHorizontal positions of vertically-tuned units'%(training_str,dataset))

#%% make plots of vertical position of vertical-tuned units, histogram for each layer
pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for li in range(np.size(layers2plot)):
  plt.subplot(npx,npy, li+1)
  ll=layers2plot[li]
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
  #  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
  units_good = np.where(rvals>r2_cutoff)[0]

#  units_good=np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
  
  centers = deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,0]))
  amps =  deepcopy(np.squeeze(fit_pars_all[ll][units_good,sf,2]))
  horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
  vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
  # coords goes [vertical pos, horizontal pos, output channel number]
  xvals = coords_all[ll][units_good,1]
  yvals = coords_all[ll][units_good,0]

  # bins for the histograms
  my_bins = np.arange(0,info['activ_dims'][ll]+1,1)

  # make the histogram  
  plt.hist(yvals[vert_units],bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[ll])
#  plt.ylim([0,10])
  if ll==nLayers-1:
      plt.xlabel('Unit horizontal position (pix)')
      plt.ylabel('Num Units')
#      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nVertical positions of vertically-tuned units'%(training_str,dataset))

#%% make plots of center distributions, across edge units only
# big plot with all layers
pp=0  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
# bins for the histograms
ori_bin_size=1
my_bins = np.arange(-ori_bin_size/2,180.5+ori_bin_size,ori_bin_size)

units_from_edge=1;
#cols = cm.Blues(np.linspace(0,1,nSamples))

layers2plot=np.arange(0,nLayers)
#layers2plot=[0,1,2,3,4]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # values to plot
  vals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,pp]))
  
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[layers2plot[ll]][:,sf,:],axis=1)))
#  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
#  vals = vals[np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]]
  
  
  xvals = coords_all[layers2plot[ll]][:,1]
  yvals = coords_all[layers2plot[ll]][:,0]
#  center = info['activ_dims'][layers2plot[ll]]/2
  dim=info['activ_dims'][layers2plot[ll]]
  xdist = np.minimum(np.abs(xvals), np.abs(xvals-dim))
  ydist = np.minimum(np.abs(yvals), np.abs(yvals-dim))
#  rad_vals = np.sqrt(np.power(xvals-center,2)+np.power(yvals-center,2))
  dist_vals = np.minimum(xdist,ydist)
  
  vals=vals[np.where(np.logical_and(rvals>r2_cutoff, dist_vals<(units_from_edge)))[0]]
#  vals=vals[np.where(np.logical_and(rvals>r2_cutoff, rad_vals<rad_lim))[0]]
#  vals=vals[np.where(rvals>r2_cutoff)[0]]
  # make the histogram  
  plt.hist(vals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[layers2plot[ll]])  
  
  if ll==nLayers-1:
      plt.xlabel('Orientation (deg)')
      plt.ylabel('Num Units')
      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
      

plt.suptitle('%s\n%s\nFit Centers -  Edge units only'%(training_str,dataset))
plt.rcParams['figure.figsize']=[18,10]

#figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% make plots of center distributions, across center units only
# big plot with all layers
pp=0  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
# bins for the histograms
ori_bin_size=1
my_bins = np.arange(-ori_bin_size/2,180.5+ori_bin_size,ori_bin_size)

rad_lim=2

layers2plot=np.arange(0,nLayers)
#layers2plot=[0,1,2,3,4]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # values to plot
  vals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,pp]))
  
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_all[layers2plot[ll]][:,sf,:],axis=1)))
#  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  fvals[np.isnan(rvals)] = -1000
  # choose just the units of interest here
#  vals = vals[np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]]
  
  xvals = coords_all[layers2plot[ll]][:,1]
  yvals = coords_all[layers2plot[ll]][:,0]
  center = info['activ_dims'][layers2plot[ll]]/2
  rad_vals = np.sqrt(np.power(xvals-center,2)+np.power(yvals-center,2))
  inds=np.where(np.logical_and(rvals>r2_cutoff, rad_vals<rad_lim))[0]

  vals=vals[inds]
#  vals=vals[np.where(np.logical_and(rvals>r2_cutoff, rad_vals<rad_lim))[0]]
#  vals=vals[np.where(rvals>r2_cutoff)[0]]
  # make the histogram  
  plt.hist(vals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[layers2plot[ll]])  
  
  if ll==nLayers-1:
      plt.xlabel('Orientation (deg)')
      plt.ylabel('Num Units')
      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
      

plt.suptitle('%s\n%s\nFit Centers -  Center units only'%(training_str,dataset))
plt.rcParams['figure.figsize']=[18,10]

#figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% make plots of center distributions, across all units
# save plots of individual layers
# params for figure saving
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]


pp=0  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
# bins for the histograms
ori_bin_size=1
my_bins = np.arange(-ori_bin_size/2,180.5+ori_bin_size,ori_bin_size)

#my_bins = np.arange(-0.5,181,0.5)
#cols = cm.Blues(np.linspace(0,1,nSamples))

layers2plot = [0,6,12,18]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
   
  # values to plot
  vals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,pp]))
  
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[layers2plot[ll]][:,sf,:],axis=1)))
#  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  
  #  vals = vals[np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]]
  vals=vals[np.where(rvals>r2_cutoff)[0]]
  
  # make the histogram  
  plt.hist(vals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[layers2plot[ll]])  
  
#  if ll==nLayers-1:
  plt.xlabel('Orientation (deg)')
  plt.ylabel('Num Units')
  plt.xticks(np.arange(0,181,45))
#  else:
#      plt.xticks([]) 
  plt.yticks([])

plt.suptitle('%s\nFit Centers'%training_str)
figname = os.path.join(figfolder, '%s_centers.pdf' % (training_str))
plt.savefig(figname, format='pdf',transparent=True)


#%% Plot each fit parameter versus the center (error bars)
# plot some example layers, save figure
pp2plot = [1,2,3,4]
#pp2plot = [1]
ppnames = ['k','amp','baseline','fwhm']
ylims = [[-50,150],[-5,30],[-5,20],[-10,120]]
tr=0
sf=0
binsize = 11.25
layers2plot=np.arange(0,nLayers)
#layers2plot =[0,6,12,18]
ori_bins = np.arange(-binsize/2,182,binsize)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')

# params for figure saving
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]

cols = cm.Blues(np.linspace(0,1,nSamples))

for pp in range(np.size(pp2plot)):
  plt.figure()
  for ll in range(np.size(layers2plot)):
    plt.subplot(npx,npy, ll+1)
          
    # use these metrics to get rid of poorly-fit units
    fvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,4]))
    rvals = deepcopy(np.squeeze(np.mean(r2_all[layers2plot[ll]][:,sf,:],axis=1)))
#    rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
     # there are a few nans in here, putting a tiny value so it won't throw an error
    rvals[np.isnan(rvals)] = -1000
#    inds2use = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
    inds2use = np.where(rvals>r2_cutoff)[0]

    # get values of center and other parameter of interest
    cvals = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,0]))
    parvals = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,pp2plot[pp]]))    
   
    # calculate mean k within each center orientation
    meanvals = np.zeros((np.size(ori_bins)-1,1))
    sdvals = np.zeros((np.size(ori_bins)-1,1))
    bincenters = np.zeros((np.size(ori_bins)-1,1))
    for oo in range(np.size(ori_bins)-1):
      inds = np.logical_and(cvals>ori_bins[oo], cvals<ori_bins[oo+1])
      bincenters[oo] = np.mean(ori_bins[oo:oo+2])
      if np.sum(inds)>0:
        vals = parvals[inds]
        meanvals[oo] = np.mean(vals)
        sdvals[oo] = np.std(vals)
      else:
        meanvals[oo] = np.nan
        sdvals[oo] = np.nan 
  
    plt.errorbar(bincenters,meanvals,sdvals,ecolor=colors_all[3,color_ind,:],linestyle='')
#    plt.errorbar(bincenters,meanvals,sdvals,ecolor=colors_all[3,color_ind,:],linestyle='')
    plt.plot(bincenters,meanvals,color=[0,0,0])
    plt.title(layer_labels[layers2plot[ll]])
#    plt.ylim(ylims[pp])
#    if ll==nLayers-1:
    plt.xlabel('Center (deg)')
    plt.ylabel(ppnames[pp])
    plt.xticks(np.arange(0,181,45))
#    else:
#        plt.xticks([]) 
#    plt.yticks([])
  
  plt.suptitle('%s\n%s versus center'%(training_str,ppnames[pp]))
#  figname = os.path.join(figfolder, '%s_%s_vs_center.pdf' % (training_str,ppnames[pp]))
#  plt.savefig(figname, format='pdf',transparent=True)
 
  
#%% make plots of FWHM distributions, across all units
pp=4  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
# bins for the histograms
my_bins = np.arange(0,180,1)

layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
  
  # values to plot
  vals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,pp]))
  
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[layers2plot[ll]][:,sf,:],axis=1)))
#  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  
  # choose just the units of interest here
#  vals = vals[np.where(np.logic®l_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]]
  vals=vals[np.where(rvals>r2_cutoff)[0]]
  # make the histogram  
  plt.hist(vals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[layers2plot[ll]])

  if ll==nLayers-1:
      plt.xlabel('Size (deg)')
      plt.ylabel('Num Units')
      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nSize (FWHM) in deg'%(training_str,dataset))

#%% make plots of K (concentration param) across all units
pp=1  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
# bins for the histograms
nbins=250;
my_bins = np.linspace(0,200,nbins)
ticks = np.arange(0,201,50)
layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
  
  # values to plot
  vals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,pp]))
  
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[layers2plot[ll]][:,sf,:],axis=1)))
#  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  
#  vals = vals[np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]]
  vals=vals[np.where(rvals>r2_cutoff)[0]]
  
  # make the histogram  
  plt.hist(vals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[layers2plot[ll]])

  if ll==nLayers-1:
      plt.xlabel('K')
      plt.ylabel('Num Units')
      plt.xticks(ticks)
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nConcentration param (k)'%(training_str,dataset))

#%% make plots of amplitude across all units
pp=2  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
r2_here=0
# bins for the histograms
nbins=250;
my_bins = np.linspace(-5,100,nbins)
ticks = np.arange(-5,101,25)
layers2plot=np.arange(0,nLayers)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
  
  # values to plot
  vals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,pp]))
  
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[layers2plot[ll]][:,sf,:],axis=1)))
#  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  
  # choose just the units of interest here
  #  vals = vals[np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]]
  vals=vals[np.where(rvals>r2_here)[0]]
  
  # make the histogram  
  plt.hist(vals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[layers2plot[ll]])

  if ll==nLayers-1:
      plt.xlabel('Resp magnitude (a.u.)')
      plt.ylabel('Num Units')
      plt.xticks(ticks)
  else:
      plt.xticks([]) 
#      plt.yticks([])
plt.axvline(0,color='k')
plt.suptitle('%s\n%s\nAmplitude'%(training_str,dataset))

#%% make plots of baseline across all units
pp=3  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
# bins for the histograms
nbins=250;
my_bins = np.linspace(-5,100,nbins)
ticks = np.arange(-5,100,25)
layers2plot=np.arange(0,nLayers)
#layers2plot=[20]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)
  
  # values to plot
  vals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,pp]))
  
  # use these metrics to get rid of poorly-fit units
  fvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,4]))
  rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[layers2plot[ll]][:,sf,:],axis=1)))
#  rvals = deepcopy(np.squeeze(r2_all[layers2plot[ll]][:,sf]))
  # there are a few nans in here, putting a tiny value so it won't throw an error
  rvals[np.isnan(rvals)] = -1000
  
  # choose just the units of interest here
  vals=vals[np.where(rvals>r2_cutoff)[0]]
#  vals = vals[np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]]

  # make the histogram  
  h=plt.hist(vals,bins=my_bins,color=colors_all[3,color_ind,:])
  
  plt.title(layer_labels[layers2plot[ll]])

  if ll==np.size(layers2plot)-1:
      plt.xlabel('Resp magnitude (a.u.)')
      plt.ylabel('Num Units')
      plt.xticks(ticks)
  else:
      plt.xticks([]) 
#      plt.yticks([])

plt.suptitle('%s\n%s\nBaseline'%(training_str,dataset))

#%% make plot of FWHM size versus k
tr=0
sf=0
pp1=1  # 0,1,2,3,4 are [center, k, amp, baseline, FWHM]
pp2=4
ll=15

plt.close('all')
plt.figure()
     
# use these metrics to get rid of poorly-fit units
fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
#rvals = deepcopy(np.squeeze(r2_all[ll][:,sf]))
 # there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
#inds2use = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
inds2use = np.where(rvals>r2_cutoff)[0]

# choose just the units of interest here
vals1 = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,pp1]))
vals2 = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,pp2]))

plt.plot(vals1,vals2,'.',color=colors_all[3,color_ind,:])
plt.title('FWHM versus k\n%s'%layer_labels[ll])
plt.xlabel('k')
plt.ylabel('FWHM')
plt.xlim([0,1])
#%% make plot of amplitude versus k
tr=0
sf=0
pp1=1  # 0,1,2,3,4 are [center, k, amp, baseline, FWHM]
pp2=2
ll=10

plt.close('all')
plt.figure()
     
# use these metrics to get rid of poorly-fit units
fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)))
#rvals = deepcopy(np.squeeze(r2_all[ll][:,sf]))
 # there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
#inds2use = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
inds2use = np.where(rvals>r2_cutoff)[0]

# choose just the units of interest here
vals1 = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,pp1]))
vals2 = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,pp2]))

plt.plot(vals1,vals2,'.',color=colors_all[3,color_ind,:])
plt.title('amplitude versus k\n%s'%layer_labels[ll])
plt.xlabel('k')
plt.ylabel('amplitude')

#%% make plot of center versus FWHM
tr=0
sf=0
pp1=0  # 0,1,2,3,4 are [center, k, amp, baseline, FWHM]
pp2=4
ll=0

plt.close('all')
plt.figure()
     
# use these metrics to get rid of poorly-fit units
fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)))
#rvals = deepcopy(np.squeeze(r2_all[ll][:,sf]))
 # there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
#inds2use = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
inds2use = np.where(rvals>r2_cutoff)[0]

# choose just the units of interest here
vals1 = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,pp1]))
vals2 = deepcopy(np.squeeze(fit_pars_all[ll][inds2use,sf,pp2]))

plt.plot(vals1,vals2,'.',markersize=1)
plt.title('size versus center\n%s'%layer_labels[ll])
plt.xlabel('center')
plt.ylabel('fwhm size')

  
#%% make plots of randomly selected well-fit units
plot_jitt=1
plt.close('all')
tr = 0
ll = 20
sf = 0
nUnitsPlot = 12
#r2_here=0;
r2_here=r2_cutoff

#if plot_jitt:
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
cvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,0]))
#else:
#   rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)))
#   cvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,0]))

rvals[np.isnan(rvals)] = -1000

# Now choose the units to plot, sorting by size
units_good = np.where(np.logical_and(rvals>r2_here, ~np.isnan(cvals)))[0]

#units_good = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
units2plot = np.random.choice(units_good, nUnitsPlot, replace='False')

npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)
cols = cm.Blues(np.linspace(0,1,nSamples))
cols2 = cm.Reds(np.linspace(0,1,nSamples))

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)
  tc = deepcopy(np.squeeze(resp_units[ll][:,units2plot[uu],sf,:]))
  real_y = np.mean(tc,axis=0)
  
 
  pars = fit_pars_all[ll][units2plot[uu],sf,:]
#  else:
#    pars = fit_pars_all[ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])
  
  for ss in range(np.shape(tc)[0]):
    yvals = tc[ss,:]
    plt.plot(ori_axis,yvals,color=colors_all[ss,color_ind,:])
   

#  plt.plot(ori_axis,real_y,color=[0,0,0])
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.title('center=%d, size=%d, r2=%.2f'%(pars[0],pars[4], rvals[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('%s\n%s-%s\nExamples of good fits (r2>%.2f)\n%s'%(training_str,dataset,sf_labels[sf],r2_here,layer_labels[ll]))

#%% make plots of randomly selected well-fit units that are close to cardinals.
#plot_jitt=1
plt.close('all')
tr = 0
ll = 0
sf = 0
nUnitsPlot = 12
#r2_here=0;
r2_here=r2_cutoff

#if plot_jitt:
rvals = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
cvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,0]))
avals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,2]))
#else:
#   rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)))
#   cvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,0]))

rvals[np.isnan(rvals)] = -1000

# Now choose the units to plot, sorting by size
#units_good = np.where(np.logical_and(np.logical_and(np.logical_and(rvals>r2_here, ~np.isnan(cvals)), cvals>175), cvals<185))[0]
#units_good = np.where(np.logical_and(np.logical_and(np.logical_and(rvals>r2_here, ~np.isnan(cvals)), cvals>85), cvals<95))[0]
#units_good = np.where(np.logical_and(np.logical_and(np.logical_and(rvals>r2_here, ~np.isnan(cvals)), cvals>40), cvals<50))[0]
units_good = np.where(np.logical_and(np.logical_and(np.logical_and(rvals>r2_here, ~np.isnan(cvals)), cvals>130), cvals<140))[0]
#units_good = np.where(np.logical_and(np.logical_and(np.logic al_and(rvals>r2_here, ~np.isnan(cvals)), cvals>15), cvals<25))[0]
#units_good = np.where(np.logical_and(np.logical_and(np.logical_and(rvals>r2_here, ~np.isnan(cvals)), np.abs(cvals-90)<90), aval><0))[0]

#units_good = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
units2plot = np.random.choice(units_good, nUnitsPlot, replace='False')

npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)
cols = cm.Blues(np.linspace(0,1,nSamples))
cols2 = cm.Reds(np.linspace(0,1,nSamples))

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)
  tc = deepcopy(np.squeeze(resp_units[ll][:,units2plot[uu],sf,:]))
  real_y = np.mean(tc,axis=0)
  
#  if plot_jitt:
  pars = fit_pars_all[ll][units2plot[uu],sf,:]
#  else:
#    pars = fit_pars_all[ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])
  
  for ss in range(np.shape(tc)[0]):
    yvals = tc[ss,:]
    plt.plot(ori_axis,yvals,color=colors_all[ss,color_ind,:])
   

#  plt.plot(ori_axis,real_y,color=[0,0,0])
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.title('center=%d, size=%d, r2=%.2f'%(pars[0],pars[4], rvals[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('%s\n%s-%s\nExamples of good fits (r2>%.2f)\n%s'%(training_str,dataset,sf_labels[sf],r2_here,layer_labels[ll]))
  
#%% make plots of randomly selected units - compare between two fit methods

plt.close('all')
tr = 0
ll = 0
sf = 0
nUnitsPlot = 12
#r2_here=0;
r2_here=r2_cutoff
# use these metrics to get rid of poorly-fit units
fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)))
rvals_jitt = deepcopy(np.squeeze(np.mean(r2_all[ll][:,sf,:],axis=1)))
#rvals = deepcopy(np.squeeze(r2_all[ll][:,sf]))
# there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
fvals[np.isnan(fvals)]= -1000

# Now choose the units to plot, sorting by size
units_good = np.where(np.logical_and(rvals>r2_here, rvals_jitt>r2_here))[0]

#units_good = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
units2plot = np.random.choice(units_good, nUnitsPlot, replace='False')

npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)
cols = cm.Blues(np.linspace(0,1,nSamples))
cols2 = cm.Reds(np.linspace(0,1,nSamples))

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)
  tc = deepcopy(np.squeeze(resp_units[ll][:,units2plot[uu],sf,:]))
  real_y = np.mean(tc,axis=0)
  
  pars = fit_pars_all[ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])
  
  pars_jitt = fit_pars_all[ll][units2plot[uu],sf,:]
  ypred_jitt = von_mises_deg(ori_axis, pars_jitt[0],pars_jitt[1],pars_jitt[2],pars_jitt[3])
  
  for ss in range(np.shape(tc)[0]):
    yvals = tc[ss,:]
    plt.plot(ori_axis,yvals,color=colors_all[ss,color_ind,:])
   

#  plt.plot(ori_axis,real_y,color=[0,0,0])
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.plot(ori_axis, ypred_jitt,color=colors_all[3,3,:])
  plt.title('center=%d/%d, size=%d/%d, r2=%.2f/%.2f'%(pars[0],pars_jitt[0],pars[4],pars_jitt[4], rvals[units2plot[uu]],rvals_jitt[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('%s\n%s\nExamples of good fits (r2>%.2f)\n%s'%(training_str,dataset,r2_here,layer_labels[ll]))

#%% make plots of well-fit units with narrowest tuning

plt.close('all')
tr = 0
ll = 0
sf = 0
nUnitsPlot = 12
#r2_here=r2_cutoff
r2_here=0

# use these metrics to get rid of poorly-fit units
fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)))
#rvals = deepcopy(np.squeeze(r2_all[ll][:,sf]))
# there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
fvals[np.isnan(fvals)]= -1000

# Now choose the units to plot, sorting by size
units_good = np.where(rvals>r2_here)[0]

#units_good = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
sizevals=deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
sizevals = sizevals[units_good]
sortinds = np.argsort(sizevals)

nUnitsTotal = np.size(units_good)
units2plot = units_good[sortinds[0:nUnitsPlot]]
npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)
cols = cm.Blues(np.linspace(0,1,nSamples))
cols2 = cm.Reds(np.linspace(0,1,nSamples))

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)
  tc = deepcopy(np.squeeze(resp_units[ll][:,units2plot[uu],sf,:]))
  real_y = np.mean(tc,axis=0)
  
  pars = fit_pars_all[ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])
  
  for ss in range(np.shape(tc)[0]):
    yvals = tc[ss,:]
    plt.plot(ori_axis,yvals,color=colors_all[ss,color_ind,:])
   

#  plt.plot(ori_axis,real_y,color=[0,0,0])
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.title('center=%d, size=%d, r2=%.2f'%(pars[0],pars[4], rvals[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('%s\n%s\nExamples of narrow fits (r2>%.2f)\n%s'%(training_str,dataset,r2_cutoff,layer_labels[ll]))

#%% make plots of well-fit units with widest tuning

plt.close('all')
tr = 0
ll = 0
sf = 0
nUnitsPlot = 12
r2_here=0

# use these metrics to get rid of poorly-fit units
fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)))
#rvals = deepcopy(np.squeeze(r2_all[ll][:,sf]))
# there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000
fvals[np.isnan(fvals)]= -1000

# Now choose the units to plot, sorting by size
units_good = np.where(rvals>r2_here)[0]

#units_good = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
sizevals=deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
sizevals = sizevals[units_good]
sortinds = np.flip(np.argsort(sizevals),axis=0)

nUnitsTotal = np.size(units_good)
units2plot = units_good[sortinds[0:nUnitsPlot]]
npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)
cols = cm.Blues(np.linspace(0,1,nSamples))
cols2 = cm.Reds(np.linspace(0,1,nSamples))

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)
  tc = deepcopy(np.squeeze(resp_units[ll][:,units2plot[uu],sf,:]))
  real_y = np.mean(tc,axis=0)
  
  pars = fit_pars_all[ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])
  
  for ss in range(np.shape(tc)[0]):
    yvals = tc[ss,:]
    plt.plot(ori_axis,yvals,color=colors_all[ss,color_ind,:])
   

#  plt.plot(ori_axis,real_y,color=[0,0,0])
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.title('center=%d, size=%d, r2=%.2f'%(pars[0],pars[4], rvals[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('Examples of wide fits (r2>%.2f)\n%s'%(r2_cutoff,layer_labels[ll]))



#%% Use the slope of tuning curves to calculate FI (making sure this roughly matches what you get for raw data before fitting...)
# this isn't taking into account variance so it is not quite correct, but provides some intuition...
tr=0
sf=0
ll=20

# use these metrics to get rid of poorly-fit units
fvals = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,4]))
rvals = deepcopy(np.squeeze(np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)))
#rvals = deepcopy(np.squeeze(r2_all[ll][:,sf]))
# there are a few nans in here, putting a tiny value so it won't throw an error
rvals[np.isnan(rvals)] = -1000

# choose just the units of interest here
units_good = np.where(rvals>r2_cutoff)[0]

#inds2use = np.where(np.logical_and(np.logical_and(rvals>r2_cutoff, fvals>fwhm_range[0]), fvals<fwhm_range[1]))[0]
nUnits = np.size(inds2use)
parslist = fit_pars_all[ll][inds2use,sf,:]

fi = np.zeros([nUnits, nOri])

for uu in range(nUnits):
  
  pars = parslist[uu,:]
  
  tf = von_mises_deg(ori_axis,pars[0],pars[1],pars[2],pars[3])
  
  slope = np.diff(np.concatenate(([tf[-1]], tf),axis=0))

  fi[uu,:] = np.power(slope,2)
  
fi_all = np.sum(fi,axis=0)

plt.close('all')
plt.figure()
 
plt.plot(ori_axis,fi_all)
plt.title('FI versus orientation\n%s'%layer_labels[ll])
plt.xlabel('orientation')
plt.ylabel('Fisher Info')
plt.xticks(np.arange(0,181,45))

#%% make plots of the sparsity (number of zero units per layer)
plt.close('all')
tr=0
sf=0
plt.figure();
cols = cm.Blues(np.linspace(0,1,nSamples))

vals = np.squeeze(propZero[tr,:])
plt.plot(np.arange(0,nLayers), vals,color=cols[3,:])
plt.xticks(np.arange(0,nLayers,1),layer_labels,rotation=90);
plt.ylabel('Proportion of units')
plt.title('All zero responses at each layer')

tr=0
sf=0
plt.figure();
cols = cm.Blues(np.linspace(0,1,nSamples))

vals = np.squeeze(propConst[tr,:])
plt.plot(np.arange(0,nLayers), vals,color=cols[3,:])
plt.xticks(np.arange(0,nLayers,1),layer_labels,rotation=90);
plt.ylabel('Proportion of units')
plt.title('Constant but non-zero responses at each layer')

tr=0
sf=0
plt.figure();
cols = cm.Blues(np.linspace(0,1,nSamples))

vals = np.squeeze(propZero[tr,:]) + np.squeeze(propConst[tr,:])
plt.plot(np.arange(0,nLayers), vals,color=cols[3,:])
plt.xticks(np.arange(0,nLayers,1),layer_labels,rotation=90);
plt.ylabel('Proportion of units')
plt.title('%s\nTotal non-responsive units at each layer'%training_str)


#%% plot r2 of average tuning curve versus mean r2 of individual sample tuning curves
plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=0;
r2_mean =r2_all[ll][:,sf]
r2_mean_each = np.mean(r2_each_sample_all[ll][:,sf,:],axis=1)
plt.plot(r2_mean,r2_mean_each,'o',color=colors_all[3,color_ind,:])
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.axhline(y=r2_cutoff, color='k')
plt.xlabel('r2 of fit to mean tuning curve')
plt.ylabel('mean r2 of fits to individual tuning curves')
plt.title(layer_labels[ll])


#%% plot center versus mean r2 of individual sample tuning curves
# plot all layers
plot_jitt=1

ylims = [-0.5, 1]
tr=0
sf=5
layers2plot=np.arange(0,nLayers)
alpha=0.1
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
plt.close('all')

plt.figure()
for ll in range(np.size(layers2plot)):
  plt.subplot(npx,npy, ll+1)

  # get values of center and other parameter of interest
  if plot_jitt:
    cvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,0]))
    r2_mean_each = np.mean(r2_all[layers2plot[ll]][:,sf,:],axis=1)  
  else:
    cvals = deepcopy(np.squeeze(fit_pars_all[layers2plot[ll]][:,sf,0]))
    r2_mean_each = np.mean(r2_each_sample_all[layers2plot[ll]][:,sf,:],axis=1)  
 
  plt.plot(cvals,r2_mean_each,'.',markersize=1,color=colors_all[3,color_ind,:],alpha=alpha)
  plt.axhline(y=r2_cutoff, color='k')
  plt.title(layer_labels[layers2plot[ll]])
  plt.ylim(ylims)
  if ll==nLayers-1:
    plt.xlabel('Center (deg)')
    plt.ylabel('Mean r2')
    plt.xticks(np.arange(0,181,45))
  else:
    plt.xticks([]) 
    plt.yticks([])

plt.suptitle('%s\n%s-%s\nr2 versus center'%(training_str,dataset,sf_labels[sf]))

   
#%% plot r2 with shifted fitting, versus r2 with regular fitting
plt.close('all')
pp=0  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
ll=1

#r2_here=-2
r2_here=r2_cutoff

r2_mean_each_jitt = np.mean(r2_all[ll][:,sf,:],axis=1) 
rvals_jitt=r2_mean_each_jitt

r2_mean_each = np.mean(r2_each_sample_all[ll][:,sf,:],axis=1) 
rvals=r2_mean_each

#inds2use=np.where(np.logical_and(np.logical_or(vals1<2, vals1>178), rvals>r2_here))[0]

plt.figure();
plt.plot(rvals,rvals_jitt,'.',markersize=10,alpha=0.01)
plt.axis('square')
plt.plot([-2,2],[-2,2],'-',color='k')
plt.axhline(r2_cutoff,color='k')
plt.axvline(r2_cutoff,color='k')
plt.xlabel('no-jitter method')
plt.ylabel('jitter method')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.title('%s - %s\n%s\nr2, two methods'%(training_strs[tr],layer_labels[ll],dataset))
#%% plot center with shifted fitting, versus center with regular fitting
plt.close('all')
plt.figure();
pp=0  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
ll=1

#r2_here=-2
r2_here=r2_cutoff

# values to plot
vals1 = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,pp]))
vals2 = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,pp]))

r2_mean_each_jitt = np.mean(r2_all[ll][:,sf,:],axis=1) 
rvals_jitt=r2_mean_each_jitt

r2_mean_each = np.mean(r2_each_sample_all[ll][:,sf,:],axis=1) 
rvals=r2_mean_each

plt.subplot(1,2,1)
inds2use = np.where(np.logical_and(rvals>r2_here, rvals_jitt>r2_here))[0]
#inds2use=np.where(np.logical_and(np.logical_and(~np.logical_or(vals1<2, vals1>178), rvals>r2_here),rvals_jitt>r2_here))[0]
plt.plot(vals1[inds2use],vals2[inds2use],'.',alpha=0.01,markersize=10)
plt.axis('square')
plt.plot([0,180],[0,180],'-',color='k')
plt.xlabel('no-jitter method')
plt.ylabel('jitter method')
plt.title('r2>%.2f'%r2_here)

plt.subplot(1,2,2)
r2_here=0
inds2use = np.where(np.logical_and(rvals>r2_here, rvals_jitt>r2_here))[0]
#inds2use=np.where(np.logical_and(np.logical_and(~np.logical_or(vals1<2, vals1>178), rvals>r2_here),rvals_jitt>r2_here))[0]
plt.plot(vals1[inds2use],vals2[inds2use],'.',alpha=0.01,markersize=10)
plt.axis('square')
plt.plot([0,180],[0,180],'-',color='k')
plt.xlabel('no-jitter method')
plt.ylabel('jitter method')
plt.title('r2>%.2f'%r2_here)
#plt.xlim([85,95])
#plt.ylim([85,95])
plt.suptitle('%s - %s\n%s\ncenters, two methods'%(training_strs[tr],dataset,layer_labels[ll]))

#%% plot center with shifted fitting, versus center with regular fitting
plt.close('all')
plt.figure();
pp=0  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
ll=1

#r2_here=-2
r2_here=r2_cutoff

# values to plot
vals1 = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,pp]))
vals2 = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,pp]))

r2_mean_each_jitt = np.mean(r2_all[ll][:,sf,:],axis=1) 
rvals_jitt=r2_mean_each_jitt

r2_mean_each = np.mean(r2_each_sample_all[ll][:,sf,:],axis=1) 
rvals=r2_mean_each

plt.subplot(1,2,1)
plt.plot(vals1,rvals,'.',alpha=0.01,markersize=10)
plt.xlabel('Center')
plt.ylabel('r2')
plt.title('no-jitter method')
plt.axhline(r2_cutoff,color='k')
plt.xlim([-1,181])
plt.ylim([-1,1])

plt.subplot(1,2,2)
plt.plot(vals2,rvals_jitt,'.',alpha=0.01,markersize=10)
plt.title('jitter method')
plt.xlabel('Center')
plt.ylabel('r2')
plt.axhline(r2_cutoff,color='k')
plt.xlim([-1,181])
plt.ylim([-1,1])

plt.suptitle('%s - %s\n%s\nr2 versus centers, two methods'%(training_strs[tr],dataset,layer_labels[ll]))
#%% plot hist of diff in center
plt.close('all')
pp=0  # 0,1,2,3,4 are center, k, amp, baseline, FWHM
tr=0
sf=0
ll=0
plt.figure();
ori_bin_size=0.5
my_bins = np.arange(-25-ori_bin_size/2,25.5+ori_bin_size,ori_bin_size)

#r2_here=-2
r2_here=r2_cutoff

# values to plot
vals1 = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,pp]))
vals2 = deepcopy(np.squeeze(fit_pars_all[ll][:,sf,pp]))

r2_mean_each_jitt = np.mean(r2_all[ll][:,sf,:],axis=1) 
rvals_jitt=r2_mean_each_jitt

r2_mean_each = np.mean(r2_each_sample_all[ll][:,sf,:],axis=1) 
rvals=r2_mean_each


plt.subplot(1,2,1);
inds2use=np.where(np.logical_and(np.logical_and(~np.logical_or(vals1<2, vals1>178), rvals>r2_here),rvals_jitt>r2_here))[0]
plt.hist(vals1[inds2use]-vals2[inds2use],my_bins)
plt.axvline(0)
#plt.axis('square')
#plt.plot([0,180],[0,180],'-',color='k')
plt.xlabel('diff in center between methods')
plt.ylabel('number of units')
plt.title('Not tuned to edges in first method')
plt.ylim([0,500])

plt.subplot(1,2,2);
inds2use=np.where(np.logical_and(np.logical_and(np.logical_or(vals1<2, vals1>178), rvals>r2_here),rvals_jitt>r2_here))[0]
plt.hist(vals1[inds2use]-vals2[inds2use],my_bins)
plt.axvline(0)
#plt.axis('square')
#plt.plot([0,180],[0,180],'-',color='k')
plt.xlabel('diff in center between methods')
plt.ylabel('number of units')
plt.title('Tuned to edges in first method')
plt.ylim([0,500])

#plt.xlim([85,95])
#plt.ylim([85,95])
plt.suptitle('%s - %s\n%s\ncenters, two methods'%(training_strs[tr],dataset,layer_labels[ll]))