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
#import statsmodels.stats.multitest
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
nImageSets = 4
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
ncomp2do = [1,2,3,4,5,6,7,8,9,10,20,50,100,500,1000,8640]

#%% load the values for t-statistics (calculated outside this code then saved)
all_tsep = []
min_var_expl=99

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
           
      if 'Filt' in dataset_all:
        dataset = '%s_rand%d'%(dataset_all,kk+1)
      elif kk==0:
        dataset = dataset_all
      else:
        dataset = '%s%d'%(dataset_all,kk)
        
      if ii==0 and kk==0:
        info = load_activations.get_info(model,dataset)
        layer_labels = info['layer_labels']
        layers2load = info['layer_labels_full']
        nOri = info['nOri']
        ori_axis = np.arange(0, nOri,1)
        
      # find the exact number of the checkpoint 
      ckpt_dirs = os.listdir(os.path.join(root,'code','tsep',model,training_str,param_str,dataset))
      ckpt_dirs = [dd for dd in ckpt_dirs if 'eval_at_ckpt-%s'%ckpt_num[0:2] in dd and '_reduced_varyncomps' in dd]
      nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'_reduced')] for dir in ckpt_dirs]            
  
      save_path = os.path.join(root,'code','tsep',model,training_str,param_str,dataset,'eval_at_ckpt-%s_reduced_varyncomps'%nums[0],'Distance_tstat_vary_ncomps.npy')
      print('loading from %s\n'%save_path)
      # array is [nLayer x nSF x nOri x nDeltaValues] in size
      tsep = np.load(save_path)

      if kk==0 and tr==0 and ii==0:
        nLayers = info['nLayers']         
        nOri = np.shape(tsep)[2]      
        # initialize this ND array to store all Fisher info calculated values
        all_tsep = np.zeros([nTrainingSchemes, np.max(nInits_list), nImageSets, nLayers, nSF, nOri, len(ncomp2do), nDeltaVals])
        all_ncomps = np.zeros([nTrainingSchemes, np.max(nInits_list), nImageSets, nLayers])

      all_tsep[tr,ii,kk,:,sf,:,:,:] = np.squeeze(tsep);
      
      
      activ_dir = os.path.join(root,'activations',model,training_str,param_str,dataset,'eval_at_ckpt-%s_reduced'%nums[0])
      print('loading var expl from %s\n'%activ_dir)
      varexpl_thisset = []
      for ll in range(nLayers):
        fn2 = os.path.join(activ_dir,'allStimsVarExpl_%s.npy'%layers2load[ll])
        var_expl = np.load(fn2)
        varexpl_thisset.append(var_expl)
        
        ncomp2keep = np.where(np.cumsum(var_expl)>min_var_expl/100)
        
        all_ncomps[tr,ii,kk,ll] = ncomp2keep[0][0];
        
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
  
#%% parameters for calculating bias
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

#%%  Plot for one "ncomp" value at a time, for four example layers. Save.
# subset of layers
ncomp2do = [1,2,3,4,5,6,7,8,9,10,20,50,100,500,1000,8640]

nn=9

layers2plot = np.asarray([0,6,12,18])

tr2plot=[0] 

init2plot = [0]
cols_grad = cm.Blues(np.linspace(0,1,6))
cols_grad=cols_grad[2:,:]

ii=0

sf=0
dd=3
#ii=1
plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]
ylimits = [0,6]
plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  for tr in tr2plot:
  
    vals_all = np.zeros([nImageSets,nOri])
    
    for kk in range(nImageSets):
      
      vals= np.squeeze(all_tsep[tr,ii,kk,layers2plot[ll],sf,:,nn,dd])
      vals_all[kk,:] = vals

  # average over image sets
    meanvals = np.mean(vals_all,0)    
    errvals = np.std(vals_all,0)

    plt.errorbar(ori_axis,meanvals,errvals,ecolor=colors_main[color_inds[tr],:],color=[0,0,0])
#    plt.plot(ori_axis,meanvals,color=colors_main[color_inds[tr],:])
    
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  plt.xlabel('Orientation (deg)')
  plt.xticks(np.arange(0,181,45))
  plt.xlim([np.min(ori_axis),np.max(ori_axis)])
  plt.ylabel('t-statistic')


  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  plt.ylim(ylimits)
  plt.yticks([0,4])
# finish up the entire plot   
plt.suptitle('%s - top %d PCs\nt-statistic, %d deg apart'% (training_strs[tr], ncomp2do[nn], delta_vals[dd]))
figname = os.path.join(figfolder, 't-stat_pretrained_%dcomps.pdf'%ncomp2do[nn])
plt.savefig(figname, format='pdf',transparent=True)


#%% plot just cardinal bias across layers, for different ncomponents overlaid. Save.

ncomp2do = [1,2,3,4,5,6,7,8,9,10,20,50,100,500,1000,8640]
nn2plot = [4,9,10]
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


for nn in range(len(nn2plot)):
  
  # matrix to store anisotropy index for each layer    
  all_vals = np.zeros([nInits,nImageSets,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):
    # loop over network initializations
    for ii in range(nInits):
      # loop over image sets
      for kk in range(nImageSets):
        
        vals= all_tsep[tr,ii,kk,layers2plot[ww1],sf,:,nn2plot[nn],dd]
       
        base_discrim=  vals[baseline_inds]
        peak_discrim = vals[card_inds]
        
        all_vals[ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
  # averaging over image sets here
  vals = np.mean(np.mean(all_vals,0),0)
  errvals = np.std(np.mean(all_vals,0),0)

  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_grad[nn,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_grad[nn,:])

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['%d PC'%ncomp2do[nn] for nn in nn2plot],loc='upper right')

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Cardinal bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Cardinal bias\n%s'%(training_strs[tr]))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'tsep_pretrained_varyncomps.pdf')
plt.savefig(figname, format='pdf',transparent=True)


#%%  Plot for one "ncomp" value at a time, all layers.
ncomp2do = [1,2,3,4,5,6,7,8,9,10,20,50,100,500,1000,8640]
nn=9
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
ylimits = [1,9]
plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  for ii in init2plot:
  
    vals_all_init = np.zeros([nImageSets,nOri])
    
    for kk in range(nImageSets):
      
      vals= np.squeeze(all_tsep[tr,ii,kk,layers2plot[ll],sf,:,nn,dd])
     
      # average over image sets
      vals_all_init[kk,:] = vals

    meanvals = np.mean(vals_all_init,0)    
    errvals = np.std(vals_all_init,0)
   
  
    plt.errorbar(ori_axis,meanvals,errvals,ecolor=colors_main[color_inds[tr],:],color=[0,0,0])

#    plt.plot(ori_axis,meanvals,color=colors_main[color_inds[tr],:])
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==len(layers2plot)-1:
    plt.xlabel('Orientation (deg)')
    plt.xticks(np.arange(0,181,45))
    plt.xlim([np.min(ori_axis),np.max(ori_axis)])
    plt.ylabel('t-statistic')
  else:
    plt.xticks([])
  
  plt.ylim(ylimits)
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
      
# finish up the entire plot   
plt.suptitle('%s - top %d PCs\nt-stat for distances between images %d deg apart'% (training_strs[tr], ncomp2do[nn], delta_vals[dd]))



#%%  Overlay different numbers of components

ncomp2do = np.asarray([1,2,3,4,5,6,7,8,9,10,20,50,100,500,1000,8640])
nn2plot = [4,9,10]
layers2plot = np.arange(0,nLayers)
tr=0 
init2plot = [0]
cols_grad = cm.Blues(np.linspace(0,1,len(ncomp2do[nn2plot])+2))
cols_grad=cols_grad[2:,:]
sf=0
dd=3
plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]
ylimits = [0,10]
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
        
        vals= np.squeeze(all_tsep[tr,ii,kk,layers2plot[ll],sf,:,nn2plot[nn],dd])
       
        
        vals_all_init[kk,:] = vals
        
      # average over image sets
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
    plt.ylabel('t-statistic')
  else:
    plt.xticks([])
  
  plt.ylim(ylimits)
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
  if ll==nLayers-1:
    plt.legend(lh,['%d PCs'%ncomp2do[nn] for nn in nn2plot],bbox_to_anchor = [-0.2, 2.5])
# finish up the entire plot   
plt.suptitle('%s\nt-stat for distances between images %d deg apart'% (training_strs[tr], delta_vals[dd]))


#%% plot each kind of Fisher Information Bias (FIB) for one network at a time 
# mean + STD error bars across network initializations
ncomp2do = [1,2,3,4,5,6,7,8,9,10,20,50,100,500,1000,8640]
nn=9

plt.rcParams['figure.figsize']=[18,10]

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

# loop over type of FIB - where are we looking for FI peaks?
for pp in range(np.size(peak_inds)):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nInits,nImageSets,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):
  
    # loop over network initializations
    for ii in range(nInits):
      # loop over image sets
      for kk in range(nImageSets):
        
        all_vals= all_tsep[tr,ii,kk,layers2plot[ww1],sf,:,nn,dd]
       
        # take the bins of interest to get anisotropy
        base_discrim=  all_vals[baseline_inds]
        peak_discrim = all_vals[peak_inds[pp]]
        
        # final value for this layer: difference divided by sum 
        aniso_vals[ii,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
      
  
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

fig.set_size_inches(10,7)
# finish up the entire plot
plt.suptitle('Each type of information bias - using %d PCs\n%s'%(ncomp2do[nn],training_strs[tr]))  
#plt.suptitle('Each type of information bias\nafter using PCA')
fig.set_size_inches(10,7)


#%%  Plot cumulative t-stat versus ncomponents - avg over all orients

ncomp2do = np.asarray([1,2,3,4,5,6,7,8,9,10,20,50,100,500,1000,8640])
nn2plot = np.arange(0,16,1)
layers2plot = np.arange(0,nLayers)
tr=0 #
if tr==4 or tr==0:
  init2plot = [0]
else:
  init2plot = [0,1,2,3]
  
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
ylimits = [0,8]

npc=10

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  plt.subplot(npx,npy, ll+1)
  
  vals_all_init = np.zeros([nImageSets,len(ncomp2do[nn2plot])])
  
  for kk in range(nImageSets):
    
    vals= np.mean(all_tsep[tr,ii,kk,layers2plot[ll],sf,:,:,dd] ,axis=0)
    vals_all_init[kk,:] = vals[nn2plot]

  meanvals = np.mean(vals_all_init,0)    
  errvals = np.std(vals_all_init,0)

  xvals = ncomp2do[nn2plot]
  # correcting for the fact that the number of comps used was actually less than the maximum depending on layer.
  ncomp_max = all_ncomps[tr,ii,kk,ll]
  xvals[xvals>ncomp_max] = ncomp_max
  plt.plot(xvals,meanvals,color=colors_main[color_inds[tr],:])
  
  plt.title('%s' % (layer_labels[layers2plot[ll]]))
  max_dim = info['activ_dims'][ll]**2*info['output_chans'][ll]
  max_after_pca = min(max_dim, 8640)

  if ll==len(layers2plot)-1:
    plt.xlabel('num comp')

    plt.ylabel('t-statistic')

  plt.axvline(npc,color=[0.8, 0.8, 0.8])
  plt.ylim(ylimits)
  plt.xlim([0,50])
  plt.xticks([0,50])
# finish up the entire plot   
plt.suptitle('%s\nt-statistic (averaged over all orients) versus number of PCs\n(line is first %d PCs)'% (training_strs[tr],npc))

#%%  Plot cumulative t-stat versus ncomponents - avg within specified bins of orientation

ncomp2do = np.asarray([1,2,3,4,5,6,7,8,9,10,20,50,100,500,1000,8640])
nn2plot = np.arange(0,16,1)
peak_inds=[card_inds, twent_inds,obl_inds, baseline_inds]
lstrings=['0 + 90', '67.5 + 157.5', '45 + 135','22.5 + 112.5']
color_inds = [1,2,3,0]
layers2plot = np.arange(0,nLayers)
tr=0 
if tr==4 or tr==0:
  init2plot = [0]
else:
  init2plot = [0,1,2,3]
  
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
ylimits = [0,8]
npc=10
# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  ax=plt.subplot(npx,npy, ll+1)
  lh=[]
  for pp in range(len(peak_inds)):
  
    vals_all_init = np.zeros([nImageSets,len(ncomp2do[nn2plot])])
    
    for kk in range(nImageSets):
      
      vals= np.mean(all_tsep[tr,ii,kk,layers2plot[ll],sf,peak_inds[pp],:,dd] ,axis=0)
      
      vals_all_init[kk,:] = vals[nn2plot]

    meanvals = np.mean(vals_all_init,0)    
    errvals = np.std(vals_all_init,0)

    xvals = ncomp2do[nn2plot]
    # correcting for the fact that the number of comps used was actually less than the maximum depending on layer.
    ncomp_max = all_ncomps[tr,ii,kk,ll]
    xvals[xvals>ncomp_max] = ncomp_max
    plt.plot(xvals,meanvals,color=colors_main[color_inds[pp],:])  
    myline = mlines.Line2D(xvals,meanvals,color = colors_main[color_inds[pp],:])
    ax.add_line(myline)   
    lh.append(myline)
    
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))
  plt.xlim([0,np.max(ncomp2do[nn2plot])])
  
  if ll==len(layers2plot)-1:
    plt.xlabel('num comp')
    plt.ylabel('t-statistic')
  else:
    plt.xticks([])
  if ll==nLayers-1:
    plt.legend(lh,lstrings)

  plt.axvline(npc,color=[0.8, 0.8, 0.8])
  plt.ylim(ylimits)
  plt.xlim([0,50])
  plt.xticks([0,50])
plt.suptitle('%s\nt-statistic (averaged within orientation bins) versus number of PCs\n(line is first %d pcs)' % (training_strs[tr],npc))

