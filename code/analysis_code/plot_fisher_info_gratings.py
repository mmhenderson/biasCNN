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

#%% paths
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures','FisherInfoPop')

#%% define parameters for what to load here

# loading all networks at once - 
# [random, trained upright images, trained 22 deg rot images, trained 45 deg rot images, pretrained]
#training_strs=['scratch_imagenet_rot_0_cos_stop_early','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos','pretrained']
training_strs=['scratch_imagenet_rot_0_stop_early','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_45_cos','pretrained']
ckpt_strs=['0','400000','400000','0']
nInits_list = [1,1,1,1]
color_inds=[0,1,3,4]

# define other basic parameters
nImageSets = 2
model='vgg16'
param_str='params1'
param_strs=[]
for ii in range(np.max(nInits_list)):    
  if ii>0:
    param_strs.append(param_str+'_init%d'%ii)
  else:
    param_strs.append(param_str)

#dataset_str=['FiltIms14AllSFCos']
dataset_str=['CosGratings']
nTrainingSchemes = np.size(training_strs)

 # values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)
nDeltaVals = np.size(delta_vals)

sf_labels=['0.01 cpp', '0.02 cpp', '0.04 cpp','0.08 cpp','0.14 cpp','0.25 cpp']
nSF=6
sf=0

#%% load the data (Fisher information calculated from each layer)
all_fisher = []

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
      
      if kk==0:
        dataset=dataset_all
      else:          
        dataset = '%s%d'%(dataset_all,kk)
       
      if ii==0 and kk==0:
        info = load_activations.get_info(model,dataset)
        layer_labels = info['layer_labels']
        nOri = info['nOri']
        ori_axis = np.arange(0, nOri,1)
        
      # find the exact number of the checkpoint 
      ckpt_dirs = os.listdir(os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset))
      ckpt_dirs = [dd for dd in ckpt_dirs if 'eval_at_ckpt-%s'%ckpt_num[0:2] in dd and '_full' in dd]
      nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'_full')] for dir in ckpt_dirs]            
  
      save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'eval_at_ckpt-%s_full'%nums[0],'Fisher_info_all_units.npy')
      print('loading from %s\n'%save_path)
      
      # Fisher info array is [nLayer x nSF x nOri x nDeltaValues] in size
      FI = np.load(save_path)
      
      if kk==0 and tr==0 and ii==0:
        nLayers = info['nLayers']         
        nOri = np.shape(FI)[2]      
        # initialize this ND array to store all Fisher info calculated values
        all_fisher = np.zeros([nTrainingSchemes, np.max(nInits_list), nImageSets, nLayers, nSF, nOri, nDeltaVals])
       
      all_fisher[tr,ii,kk,:,:,:,:] = np.squeeze(FI);
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
#%% parameters for calculating Fisher information bias
# define the bins of interest
b = np.arange(22.5,nOri,90)  # baseline
t = np.arange(67.5,nOri,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
c = np.arange(0,nOri,90) # cardinals
o = np.arange(45,nOri,90)  # obliques
bin_size=20

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

#%%  Plot Fisher information - one network training set at a time.
# plot just a subset of layers here
#layers2plot = [0,6,12,18]
layers2plot=range(nLayers)
# which network to plot here? [0,1,2,3,4] are
# [random, trained upright images, trained 22 deg rot images, trained 45 deg rot images, pretrained]
tr=2  # can change this value to plot the netorks with different training sets

#if tr==4 or tr==0:
init2plot = [0]
#else:
#  init2plot = [0,1,2,3]
  
sf=4
dd=3
ylims=[0, 200]

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
  
  FI_all_init = np.zeros([len(init2plot),nOri])
  
  for ii in init2plot:
   
    fish= all_fisher[tr,ii,:,layers2plot[ll],sf,:,dd] 
    # correct for the different numbers of units in each layer
    nUnits = info['activ_dims'][layers2plot[ll]]**2*info['output_chans'][layers2plot[ll]]  
    fish = fish/nUnits  
    # average over image sets
    FI_all_init[ii,:] = np.mean(fish,axis=0)
    
  # get mean and std, plot errorbars.
  # Errorbars are over network initializations, if there is more than one. 
  # otherwise, no errorbars are plotted.
  meanfish = np.mean(FI_all_init,0)    
  errfish = np.std(FI_all_init,0)
 
  if len(init2plot)>1:
    plt.errorbar(ori_axis,meanfish,errfish,ecolor=colors_main[color_inds[tr],:],color=[0,0,0])
  else:
    plt.plot(ori_axis,meanfish,color=colors_main[color_inds[tr],:])
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  plt.xlabel('Orientation (deg)')
  plt.xticks(np.arange(0,181,45))
  plt.xlim([np.min(ori_axis),np.max(ori_axis)])
  plt.ylabel('FI (a.u.)')
  
#  plt.ylim(ylims)
  
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
             
# finish up the entire plot   
plt.suptitle('%s\n%s, %s' % (training_strs[tr], dataset_str[0],sf_labels[sf]))
figname = os.path.join(figfolder, '%s_FisherInfo.pdf' % (training_strs[tr]))
#plt.savefig(figname, format='pdf',transparent=True)

#%%  Plot Fisher information - one network training set at a time.
# plot all layers
layers2plot = np.arange(0,nLayers)
# which network to plot here? [0,1,2,3,4] are
# [random, trained upright images, trained 22 deg rot images, trained 45 deg rot images, pretrained]
tr=1 # can change this value to plot the models with different training sets 
if tr==4 or tr==0:
  init2plot = [0]
else:
  init2plot = [0,1,2,3]
  
sf=0
dd=3

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
  
  FI_all_init = np.zeros([len(init2plot),nOri])
  
  for ii in init2plot:
    
    fish= all_fisher[tr,ii,:,layers2plot[ll],sf,:,dd] 
    # correct for the different numbers of units in each layer
    nUnits = info['activ_dims'][layers2plot[ll]]**2*info['output_chans'][layers2plot[ll]]  
    fish = fish/nUnits  
    # average over image sets
    FI_all_init[ii,:] = np.mean(fish,axis=0)
    
  # get mean and std, plot errorbars.
  # Errorbars are over network initializations, if there is more than one. 
  # otherwise, no errorbars are plotted.
  meanfish = np.mean(FI_all_init,0)    
  errfish = np.std(FI_all_init,0)
 
  if len(init2plot)>1:
    plt.errorbar(ori_axis,meanfish,errfish,ecolor=colors_main[color_inds[tr],:],color=[0,0,0])
  else:
    plt.plot(ori_axis,meanfish,color=colors_main[color_inds[tr],:])
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  plt.xlabel('Orientation (deg)')
  plt.xticks(np.arange(0,181,45))
  plt.xlim([np.min(ori_axis),np.max(ori_axis)])
  plt.ylabel('FI (a.u.)')
  
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
      
# finish up the entire plot   
plt.suptitle('%s' % (training_strs[tr]))

#%% Plot FIB: separate by SF of gratings

# which type of FIB to plot?
pp=2  # set to 0, 1 or 2 to plot [FIB-0, FIB-22, FIB-45]

tr=2 # plot just one model here
ii=0
sf2plot=np.arange(0,nSF)
dd=3
peak_inds=[card_inds, twent_inds,obl_inds]
lstrings=['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline']

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

handles = []
layers2plot = np.arange(0,nLayers,1)

alpha=0.01;
# for each layer, compare bias for trained models versus random models
pvals_trained_vs_random=np.zeros([1, nLayers])
nTotalComp = np.size(pvals_trained_vs_random)
# matrix to store anisotropy index for each layer    
aniso_vals = np.zeros([len(sf2plot),1,nImageSets,np.size(layers2plot)])

# loop over network layers
for ww1 in range(np.size(layers2plot)):
  # loop over networks with each training set
  for sf in range(len(sf2plot)):
    
    # loop over random image sets
    for kk in range(nImageSets):

      # FI is nOri pts long
      all_fish= np.squeeze(deepcopy(all_fisher[tr,ii,kk,layers2plot[ww1],sf,:,dd]))
      
      # take the bins of interest to get bias
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[peak_inds[pp]]
      
      # final value for this FIB: difference divided by sum 
      aniso_vals[sf,ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
  # now do pairwise stats: compare trained model versus the random model
  # 4 values per model here
#  real_vals_rand = np.squeeze(aniso_vals[0,:,:,ww1])
#  real_vals_trained = np.squeeze(aniso_vals[1,:,:,ww1])
#
#  t, p = scipy.stats.ttest_ind(real_vals_rand,real_vals_trained,equal_var=False)
#  # making this one-tailed
#  if t<0:
#    p_one_tailed = p/2
#  else:
#    p_one_tailed = 1-p/2
#  pvals_trained_vs_random[0,ww1] = p_one_tailed
   
# do FDR correction on all these pairwise comparisons
#[is_sig_fdr, pvals_fdr] = statsmodels.stats.multitest.fdrcorrection(np.ravel(pvals_trained_vs_random),alpha)
#is_sig_fdr = np.reshape(is_sig_fdr, np.shape(pvals_trained_vs_random))
#pvals_fdr = np.reshape(pvals_fdr, np.shape(pvals_trained_vs_random))
    
# put the line for each FIB onto the plot 
# error bars are across 4 image sets
handles=[]
for sf in range(len(sf2plot)):    
  vals = np.squeeze(np.mean(aniso_vals[sf,:,:,:],1))
  errvals = np.squeeze(np.std(aniso_vals[sf,:,:,:],1)) 
  eb=plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[sf,0,:],zorder=21)
  handles.append(eb)
# add asterisks for comparisons that were significant
#all_xvals = np.arange(0,np.size(layers2plot),1)
#is_sig_greater = np.logical_and(is_sig_fdr[tr-1,:], vals>np.mean(np.mean(aniso_vals[0,:,:,:],1),0))
#plt.scatter(all_xvals[is_sig_greater], vals[is_sig_greater]+0.1,marker=(5,2,0),s=10,color=colors_main[color_inds[tr2plot[tr]],:],edgecolors=None,zorder=0)

# finish up the entire plot
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Information Bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.legend(handles,sf_labels)
plt.suptitle('FIB: %s\npre-trained model versus random model'%(lstrings[pp]))  
fig.set_size_inches(10,7)


#%% Plot FIB: comparing pretrained model versus random model. 
# do stats comparing the values between trained and random models.

# which type of FIB to plot?
pp=0  # set to 0, 1 or 2 to plot [FIB-0, FIB-22, FIB-45]

tr2plot=[0,1] # plot just pre-trained and random model here.
ii=0
sf=0
dd=3
peak_inds=[card_inds, twent_inds,obl_inds]
lstrings=['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline']

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

handles = []
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
      all_fish= np.squeeze(deepcopy(all_fisher[tr2plot[tr],ii,kk,layers2plot[ww1],sf,:,dd]))
      
      # take the bins of interest to get bias
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[peak_inds[pp]]
      
      # final value for this FIB: difference divided by sum 
      aniso_vals[tr,ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
  # now do pairwise stats: compare trained model versus the random model
  # 4 values per model here
#  real_vals_rand = np.squeeze(aniso_vals[0,:,:,ww1])
#  real_vals_trained = np.squeeze(aniso_vals[1,:,:,ww1])
#
#  t, p = scipy.stats.ttest_ind(real_vals_rand,real_vals_trained,equal_var=False)
#  # making this one-tailed
#  if t<0:
#    p_one_tailed = p/2
#  else:
#    p_one_tailed = 1-p/2
#  pvals_trained_vs_random[0,ww1] = p_one_tailed
   
# do FDR correction on all these pairwise comparisons
#[is_sig_fdr, pvals_fdr] = statsmodels.stats.multitest.fdrcorrection(np.ravel(pvals_trained_vs_random),alpha)
#is_sig_fdr = np.reshape(is_sig_fdr, np.shape(pvals_trained_vs_random))
#pvals_fdr = np.reshape(pvals_fdr, np.shape(pvals_trained_vs_random))
    
# put the line for each FIB onto the plot 
# error bars are across 4 image sets
for tr in range(len(tr2plot)):    
  vals = np.squeeze(np.mean(aniso_vals[tr,:,:,:],1))
  errvals = np.squeeze(np.std(aniso_vals[tr,:,:,:],1)) 
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=colors_main[color_inds[tr2plot[tr]],:],zorder=21)
 
# add asterisks for comparisons that were significant
#all_xvals = np.arange(0,np.size(layers2plot),1)
#is_sig_greater = np.logical_and(is_sig_fdr[tr-1,:], vals>np.mean(np.mean(aniso_vals[0,:,:,:],1),0))
#plt.scatter(all_xvals[is_sig_greater], vals[is_sig_greater]+0.1,marker=(5,2,0),s=10,color=colors_main[color_inds[tr2plot[tr]],:],edgecolors=None,zorder=0)

# finish up the entire plot
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Information Bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.suptitle('FIB: %s\npre-trained model versus random model'%(lstrings[pp]))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'Pretrained_vs_random_%s.pdf'%lstrings[pp])
#plt.savefig(figname, format='pdf',transparent=True)

#%% Plot FIB: comparing models trained with rotated images, versus random model. 
# do stats comparing the values between trained and random models.

# which type of FIB to plot?
pp=1  # set to 0, 1 or 2 to plot [FIB-0, FIB-22, FIB-45]

tr2plot=[0,1,2,3] # not including pre-trained model on this plot
nInits=4
sf=0
dd=3
peak_inds=[card_inds, twent_inds,obl_inds]
lstrings=['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline']
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)

alpha=0.01;
# for each layer, compare bias for trained models versus random models
pvals_trained_vs_random=np.zeros([1, nLayers])
nTotalComp = np.size(pvals_trained_vs_random)
# matrix to store anisotropy index for each layer    
aniso_vals = np.zeros([nTrainingSchemes,nInits,nImageSets,np.size(layers2plot)])

# loop over network layers
for ww1 in range(np.size(layers2plot)):
  # loop over networks with each training set
  for tr in range(len(tr2plot)):
    # loop over network initializations
    for ii in range(nInits):
      # loop over random image sets
      for kk in range(nImageSets):

        # FI is nOri pts long
        all_fish= np.squeeze(deepcopy(all_fisher[tr2plot[tr],ii,kk,layers2plot[ww1],sf,:,dd]))
        
        # take the bins of interest to get bias
        base_discrim=  all_fish[baseline_inds]
        peak_discrim = all_fish[peak_inds[pp]]
        
        # final value for this layer: difference divided by sum 
        aniso_vals[tr,ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
        
  # stats: compare trained models versus the random models
  # depending on which FIB type we're looking at, compare the models with that 
  # type of bias, versus the random models.
  # 16 total values per training set here
  real_vals_rand = np.reshape(aniso_vals[0,:,:,ww1], [nImageSets*nInits,1])
  real_vals_trained = np.reshape(aniso_vals[pp+1,:,:,ww1], [nImageSets*nInits,1])

  t, p = scipy.stats.ttest_ind(real_vals_rand,real_vals_trained,equal_var=False)
  # making this one-tailed
  if t<0:
    p_one_tailed = p/2
  else:
    p_one_tailed = 1-p/2
  pvals_trained_vs_random[0,ww1] = p_one_tailed
    
# do FDR correction on all these pairwise comparisons
[is_sig_fdr, pvals_fdr] = statsmodels.stats.multitest.fdrcorrection(np.ravel(pvals_trained_vs_random),alpha)
is_sig_fdr = np.reshape(is_sig_fdr, np.shape(pvals_trained_vs_random))
pvals_fdr = np.reshape(pvals_fdr, np.shape(pvals_trained_vs_random))
    
# put the line for each FIB onto the plot 
# error bars are across initializations and image sets
for tr in range(len(tr2plot)): 

  vals = np.reshape(aniso_vals[tr,:,:,:],[nInits*nImageSets,np.size(layers2plot)])
  meanvals = np.mean(vals,axis=0)
  errvals = np.std(vals,axis=0)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),meanvals,errvals,color=colors_main[color_inds[tr2plot[tr]],:],zorder=21-tr)

  # put markers on the plot to indicate difference between trained and random models
  if tr==pp+1:
    all_xvals = np.arange(0,np.size(layers2plot),1)
    is_sig_greater = np.logical_and(is_sig_fdr[0,:], meanvals>np.mean(np.mean(aniso_vals[0,:,:,:],1),0))
    plt.scatter(all_xvals[is_sig_greater], meanvals[is_sig_greater]+0.1,marker=(5,2,0),s=10,color=colors_main[color_inds[tr2plot[tr]],:],edgecolors=None,zorder=0)
    
# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(handles,training_strs)

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Information Bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('FIB: %s\nModels trained on rot images versus random model'%(lstrings[pp]))    
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'Trained_rot_versus_random_%s.pdf'%(lstrings[pp]))
#plt.savefig(figname, format='pdf',transparent=True)

