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
#from copy import deepcopy
import statsmodels.stats.multitest
#import scipy.stats
import matplotlib.lines as mlines

#%% paths
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures','FisherInfoPop')

#%% define parameters for what to load here

training_strs=['pretrained']

ckpt_strs=['0']

nInits_list = [1]
color_inds=[1]

# define other basic parameters
nImageSets_list = [4]
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

info = load_activations.get_info(model, dataset_str[0])
sf_labels = ['broadband SF']
nSF=1
sf=0
maxfeat = 8640

# how many PCs were combined to compute the measure? 
ncomp2do = np.arange(2,48,1)

#%% load the values for FI (calculated outside this code then saved)
all_fish_cov = []


# load activations for each training set of images (training schemes)
for tr in range(nTrainingSchemes):
  training_str = training_strs[tr]
  ckpt_num = ckpt_strs[tr]
  dataset_all = dataset_str[0]
  nInits = nInits_list[tr]
  nImageSets = nImageSets_list[tr]
  # different initializations with same training set
  for ii in range(nInits):
 
    param_str=param_strs[ii]
  
    # different versions of the evaluation image set (samples)
    for kk in range(nImageSets):
           
      if 'Filt' in dataset_all:
        dataset = '%s_rand%d'%(dataset_all,kk+1)
      elif kk==0:
        dataset = dataset_all
      else:
        dataset = '%s%d'%(dataset_all,kk)
        
      if ii==0 and kk==0:
        info = load_activations.get_info(model,dataset)
        layer_labels = info['layer_labels']
        nOri = info['nOri']
        ori_axis = np.arange(0, nOri,1)
        
      # find the exact number of the checkpoint 
#      ckpt_dirs = os.listdir(os.path.join(root,'code','fisher_info_cov',model,training_str,param_str,dataset))
      ckpt_dirs = os.listdir(os.path.join(root,'code','fisher_info_cov_new',model,training_str,param_str,dataset))
      ckpt_dirs = [dd for dd in ckpt_dirs if 'eval_at_ckpt-%s'%ckpt_num[0:2] in dd and '_reduced_varyncomps' in dd]
      nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'_reduced')] for dir in ckpt_dirs]            
  
#      save_path = os.path.join(root,'code','fisher_info_cov',model,training_str,param_str,dataset,'eval_at_ckpt-%s_reduced_varyncomps'%nums[0],'Fisher_info_cov_vary_ncomps.npy')
      save_path = os.path.join(root,'code','fisher_info_cov_new',model,training_str,param_str,dataset,'eval_at_ckpt-%s_reduced_varyncomps'%nums[0],'Fisher_info_cov_vary_ncomps.npy')
      print('loading from %s\n'%save_path)
      # Fisher info array is [nLayer x nSF x nOri x nDeltaValues] in size
      fi = np.load(save_path)

      if kk==0 and tr==0 and ii==0:
        nLayers = info['nLayers']         
        nOri = np.shape(fi)[2]      
        # initialize this ND array to store all Fisher info calculated values
        all_fish_cov = np.zeros([nTrainingSchemes, np.max(nInits_list), np.max(nImageSets_list), nLayers, nSF, nOri, len(ncomp2do), nDeltaVals])

      all_fish_cov[tr,ii,kk,:,sf,:,:,:] = np.squeeze(fi);

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

#%%  Plot FI for one "ncomp" value at a time - four example layers. Save.
# subset of layers
ncomp2do = np.arange(2,48,1)
nn=8
layers2plot = np.asarray([0,6,12,18])
tr2plot=[0] 
init2plot = [0]

cols_grad = cm.Blues(np.linspace(0,1,6))
cols_grad=cols_grad[2:,:]

ii=0
sf=0
dd=3

plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]
ylimits = [-1,5]
plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  for tr in tr2plot:
  
    vals_all= np.zeros([nImageSets,nOri])
    
    for kk in range(nImageSets):
      
      vals= np.squeeze(all_fish_cov[tr,ii,kk,layers2plot[ll],sf,:,nn,dd])
      vals_all[kk,:] = vals
      
    # average over image sets
    meanvals = np.mean(vals_all,0)    
    errvals = np.std(vals_all,0)

    plt.errorbar(ori_axis,meanvals,errvals,ecolor=colors_main[color_inds[tr],:],color=[0,0,0])
#    plt.plot(ori_axis,meanvals,color=colors_main[color_inds[tr],:])
    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  plt.xlabel('Orientation (deg)')
  plt.xticks(np.arange(0,181,45))
  plt.xlim([np.min(ori_axis),np.max(ori_axis)])
  plt.ylabel('fisher info')

  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  plt.ylim(ylimits)
  plt.yticks([0,4])
# finish up the entire plot   
plt.suptitle('%s - top %d PCs\nFisher info (w covariance) %d deg apart'% (training_strs[tr], ncomp2do[nn], delta_vals[dd]))
figname = os.path.join(figfolder, 'FIcov_pretrained_%dcomps.pdf'%ncomp2do[nn])
plt.savefig(figname, format='pdf',transparent=True)


#%% plot just cardinal bias across layers, for different ncomponents overlaid. Save.

ncomp2do = np.arange(2,48,1)
nn2plot = [3,8,18]

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr=0
cols_grad = cm.Blues(np.linspace(0,1,len(nn2plot)+2))
cols_grad=cols_grad[2:,:]
nInits=1
nImageSets=4
# loop over type of FIB - where are we looking for FI peaks?
for nn in range(len(nn2plot)):
  
  # matrix to store anisotropy index for each layer    
  all_vals = np.zeros([nInits,nImageSets,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):
    # loop over network initializations
    for ii in range(nInits):
      # loop over image sets
      for kk in range(nImageSets):
        
        vals= all_fish_cov[tr,ii,kk,layers2plot[ww1],sf,:,nn2plot[nn],dd]
       
        base_discrim=  vals[baseline_inds]
        peak_discrim = vals[card_inds]
        
        all_vals[ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
  # average over image sets
  vals = np.mean(np.mean(all_vals,0),0)
  errvals = np.std(np.mean(all_vals,0),0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_grad[nn,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_grad[nn,:])


# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['%d PC'%ncomp2do[nn] for nn in nn2plot],loc='upper left')

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Cardinal bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Cardinal bias\n%s'%(training_strs[tr]))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'FIBcov_pretrained_varyncomps.pdf')
plt.savefig(figname, format='pdf',transparent=True)


#%%  Plot for one "ncomp" value at a time - all layers. 
ncomp2do = np.arange(2,48,1)
nn=8

layers2plot = np.arange(0,nLayers)
tr=0 
init2plot = [0]
cols_grad = cm.Blues(np.linspace(0,1,6))
cols_grad=cols_grad[2:,:]

sf=0
dd=3

plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]
ylimits = [0,5]
plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  vals_all_init = np.zeros([len(init2plot),nImageSets, nOri])
    
  for ii in init2plot:
      
    for kk in range(nImageSets):
      
      vals= np.squeeze(all_fish_cov[tr,ii,kk,layers2plot[ll],sf,:,nn,dd])
      # average over image sets
      vals_all_init[ii,kk,:] = vals

  meanvals = np.mean(np.mean(vals_all_init,0),0)    
  errvals = np.std(np.mean(vals_all_init,0),0)

  plt.errorbar(ori_axis,meanvals,errvals,ecolor=colors_main[color_inds[tr],:],color=[0,0,0])
#  plt.plot(ori_axis,meanvals,color=colors_main[color_inds[tr],:])
  
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==len(layers2plot)-1:
    plt.xlabel('Orientation (deg)')
    plt.xticks(np.arange(0,181,45))
    plt.xlim([np.min(ori_axis),np.max(ori_axis)])
    plt.ylabel('fisher info')
  else:
    plt.xticks([])
  
#  plt.ylim(ylimits)
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  plt.ylim(ylimits)
# finish up the entire plot   
plt.suptitle('%s - top %d PCs\nFisher info (w covariance) %d deg apart'% (training_strs[tr], ncomp2do[nn], delta_vals[dd]))

#%%  Plot FI for all layers - overlay different numbers of components

ncomp2do = np.arange(2,48,1)
nn2plot = [0,3,8,18]

layers2plot = np.arange(0,nLayers)

tr=0 

init2plot = [0]
cols_grad = cm.Blues(np.linspace(0,1,len(nn2plot)+2))
cols_grad=cols_grad[2:,:]
 
sf=0
dd=3

plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]
ylimits = [0,21]
plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  ax=plt.subplot(npx,npy, ll+1)
  lh=[]
  
  for nn in range(len(nn2plot)):
    for ii in init2plot:
    
      vals_all_init = np.zeros([nImageSets,nOri])
      
      for kk in range(nImageSets):
        
        vals= np.squeeze(all_fish_cov[tr,ii,kk,layers2plot[ll],sf,:,nn2plot[nn],dd])
       
       
        vals_all_init[kk,:] = vals

      meanvals = np.mean(vals_all_init,0)    
      errvals = np.std(vals_all_init,0)
  
      plt.plot(ori_axis,meanvals,color=cols_grad[nn,:])
      myline = mlines.Line2D(ori_axis,meanvals,color = cols_grad[nn,:])
      ax.add_line(myline)   
      lh.append(myline)
      
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==len(layers2plot)-1:
    plt.xlabel('Orientation (deg)')
    plt.xticks(np.arange(0,181,45))
    plt.xlim([np.min(ori_axis),np.max(ori_axis)])
    plt.ylabel('fisher infor')
  else:
    plt.xticks([])
  

  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  if ll==nLayers-1:
    plt.legend(lh,['%d PCs'%ncomp2do[nn] for nn in nn2plot],bbox_to_anchor = [-0.2, 2.5])
# finish up the entire plot   
plt.suptitle('%s\nFisher information (w covariance) between images %d deg apart'% (training_strs[tr], delta_vals[dd]))


#%% plot just oblique bias across layers, for different ncomponents overlaid. 

ncomp2do = np.arange(2,48,1)
nn2plot = [3,8,18]

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr=0
cols_grad = cm.Reds(np.linspace(0,1,len(nn2plot)+2))
cols_grad=cols_grad[2:,:]
nInits=1
nImageSets=4
# loop over type of FIB - where are we looking for FI peaks?
for nn in range(len(nn2plot)):
  
  # matrix to store anisotropy index for each layer    
  all_vals = np.zeros([nInits,nImageSets,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):
    # loop over network initializations
    for ii in range(nInits):
      # loop over image sets
      for kk in range(nImageSets):
        
        vals= all_fish_cov[tr,ii,kk,layers2plot[ww1],sf,:,nn2plot[nn],dd]
       
        base_discrim=  vals[baseline_inds]
        peak_discrim = vals[obl_inds]
        
        all_vals[ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
  # average over image sets
  vals = np.mean(np.mean(all_vals,0),0)
  errvals = np.std(np.mean(all_vals,0),0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_grad[nn,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_grad[nn,:])


# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['%d PC'%ncomp2do[nn] for nn in nn2plot],loc='upper left')

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Oblique bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Oblique bias\n%s'%(training_strs[tr]))  
fig.set_size_inches(10,7)
#figname = os.path.join(figfolder, 'FIBcov_pretrained_varyncomps.pdf')
#plt.savefig(figname, format='pdf',transparent=True)
#%% plot mean FI across orientations, for different ncomponents

ncomp2do = np.arange(2,48,1)
nn2plot = [3,8,18]


plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr=0
cols_grad = cm.Greys(np.linspace(0,1,len(nn2plot)+2))
cols_grad=cols_grad[2:,:]

# loop over type of FIB - where are we looking for FI peaks?
for nn in range(len(nn2plot)):
  
  # matrix to store anisotropy index for each layer    
  all_vals = np.zeros([nInits,nImageSets,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):
    # loop over network initializations
    for ii in range(nInits):
      # loop over image sets
      for kk in range(nImageSets):
        
        vals= all_fish_cov[tr,ii,kk,layers2plot[ww1],sf,:,nn2plot[nn],dd]
       
        all_vals[ii,kk,ww1] = np.mean(vals)
        
  # put the line for this FIB onto the plot      
  vals = np.mean(np.mean(all_vals,0),0)
  errvals = np.std(np.mean(all_vals,0),0)
#  errvals = np.std(np.reshape(aniso_vals,[nInits*nImageSets,np.size(layers2plot)]),0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_grad[nn,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_grad[nn,:])


# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['%d PC'%ncomp2do[nn] for nn in nn2plot],loc='upper left')

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Average FI')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('FI averaged over all orients\n%s'%(training_strs[tr]))  
#plt.suptitle('Each type of information bias\nafter using PCA')
fig.set_size_inches(10,7)

#%% plot each kind of Bias (FIB) for one network at a time 
# mean + STD error bars across network initializations
ncomp2do = np.arange(2,48,1)
nn=8

peak_inds=[card_inds, twent_inds,obl_inds]
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr=0
nInits = nInits_list[tr]
# loop over type of FIB - where are we looking for FI peaks?
for pp in range(np.size(peak_inds)):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nInits,nImageSets,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):
    for ii in range(nInits):
      for kk in range(nImageSets):
        
        all_vals= all_fish_cov[tr,ii,kk,layers2plot[ww1],sf,:,nn,dd]

        base_discrim=  all_vals[baseline_inds]
        peak_discrim = all_vals[peak_inds[pp]]
        
        # final value for this layer: difference divided by sum 
        aniso_vals[ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
  # put the line for this FIB onto the plot      
  vals = np.mean(np.mean(aniso_vals,0),0)
  errvals = np.std(np.mean(aniso_vals,0),0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = colors_main[pp+1,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=colors_main[pp+1,:])


# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline'])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Information Bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Each type of information bias - using %d components\n%s'%(ncomp2do[nn],training_strs[tr]))  
#plt.suptitle('Each type of information bias\nafter using PCA')
fig.set_size_inches(10,7)



#%%  Plot cumulative info versus ncomponents - avg over all orients

ncomp2do = np.arange(2,48,1)
nn2plot = np.arange(0,46,1)

layers2plot = np.arange(0,nLayers)

tr=0
init2plot = [0] 
sf=0
dd=3
ii=0
plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]

plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
ylimits = [0,19]

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  vals_all_init = np.zeros([nImageSets,len(ncomp2do[nn2plot])])
  
  for kk in range(nImageSets):
    
    vals= np.mean(all_fish_cov[tr,ii,kk,layers2plot[ll],sf,:,:,dd] ,axis=0)
   
    vals_all_init[kk,:] = vals[nn2plot]
    

  meanvals = np.mean(vals_all_init,0)    
  errvals = np.std(vals_all_init,0)

  plt.plot(ncomp2do[nn2plot],meanvals,color=colors_main[color_inds[tr],:])

  plt.title('%s' % (layer_labels[layers2plot[ll]]))
  
  if ll==len(layers2plot)-1:
    plt.xlabel('num comp')
    plt.ylabel('fisher infor')
  else:
    plt.xticks([])

  plt.axvline(10,color=[0.8, 0.8, 0.8])
 
plt.suptitle('%s\nFisher info w covariance (averaged over all orients) versus number of PCs' % (training_strs[tr]))

#%%  Plot cumulative dist versus ncomponents - avg within specified bins of orientation

ncomp2do = np.arange(2,48,1)
nn2plot = np.arange(0,46,1)

peak_inds=[card_inds, twent_inds,obl_inds, baseline_inds]
lstrings=['0 + 90', '67.5 + 157.5', '45 + 135','22.5 + 112.5']
color_inds = [1,2,3,0]
layers2plot = np.arange(0,nLayers)
tr=0 
init2plot = [0]
sf=0
dd=3
ii=0
plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]

plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
ylimits = [0,5]

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  ax=plt.subplot(npx,npy, ll+1)
  lh=[]
  for pp in range(len(peak_inds)):
  
    vals_all_init = np.zeros([nImageSets,len(ncomp2do[nn2plot])])
    
    for kk in range(nImageSets):
      
      vals= np.mean(all_fish_cov[tr,ii,kk,layers2plot[ll],sf,peak_inds[pp],:,dd] ,axis=0)
      
      vals_all_init[kk,:] = vals[nn2plot]

    meanvals = np.mean(vals_all_init,0)    
    errvals = np.std(vals_all_init,0)

    plt.plot(ncomp2do[nn2plot],meanvals,color=colors_main[color_inds[pp],:])
    myline = mlines.Line2D(ncomp2do[nn2plot],meanvals,color = colors_main[color_inds[pp],:])
    ax.add_line(myline)   
    lh.append(myline)
    
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==len(layers2plot)-1:
    plt.xlabel('num comp')
    plt.ylabel('fisher infor')
  else:
    plt.xticks([])
  if ll==nLayers-1:
    plt.legend(lh,lstrings,bbox_to_anchor = [1.01, 1.01])

  plt.axvline(10,color=[0.8, 0.8, 0.8])
  plt.ylim(ylimits) 
plt.suptitle('%s\nFisher info w covariance (averaged in each orient bin) versus number of PCs\n' % (training_strs[tr]))

