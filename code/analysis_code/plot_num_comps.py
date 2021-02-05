#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
import load_activations

#%% paths
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
#%% define parameters for what to load here

# loading all networks at once - 
# [random, trained upright images, trained 22 deg rot images, trained 45 deg rot images, pretrained]
training_strs=['scratch_imagenet_rot_0_stop_early','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_22_cos','scratch_imagenet_rot_45_cos','pretrained']
#training_strs=['pretrained','scratch_imagenet_rot_0_cos','scratch_imagenet_rot_45_cos']
#training_strs=['scratch_imagenet_rot_0_cos']

ckpt_strs=['0','400000','400000','400000','0']
nInits_list = [1,1,1,1,1]
color_inds=[0,1,2,3,1]

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
#dataset_str = ['CosGratings']
nTrainingSchemes = np.size(training_strs)

sf_labels=['broadband SF']

min_var_expl=99

#%%
# load activations for each training set of images (training schemes)

varexpl_all = []
for tr in range(nTrainingSchemes):
  training_str = training_strs[tr]
  ckpt_num = ckpt_strs[tr]
  dataset_all = dataset_str[0]
  nInits = nInits_list[tr]
  varexpl_thistr = []
  # different initializations with same training set
  for ii in range(nInits):
 
    param_str=param_strs[ii]
    varexpl_thisinit = []
    # different versions of the evaluation image set (samples)
    for kk in range(nImageSets):
           
      if 'Filt' in dataset_all:
        dataset = '%s_rand%d'%(dataset_all,kk+1)
      elif kk==0:
        dataset = dataset_all
      else:
        dataset = '%s%d'%(dataset_all,kk)
      
      if kk==0 and tr==0 and ii==0:

        info=load_activations.get_info(model, dataset)
        nLayers = info['nLayers'] 
        layers2load=info['layer_labels_full']  
        layer_labels = info['layer_labels']
        all_ncomps = np.zeros([nTrainingSchemes, np.max(nInits_list), nImageSets, nLayers])
        
        
      # find the exact number of the checkpoint 
      ckpt_dirs = os.listdir(os.path.join(root,'activations',model,training_str,param_str,dataset))
      ckpt_dirs = [dd for dd in ckpt_dirs if 'eval_at_ckpt-%s'%ckpt_num[0:2] in dd and '_reduced' in dd]
      nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'_reduced')] for dir in ckpt_dirs]  
      
      activ_dir = os.path.join(root,'activations',model,training_str,param_str,dataset,'eval_at_ckpt-%s_reduced'%nums[0])
      print('loading var expl from %s\n'%activ_dir)
      varexpl_thisset = []
      for ll in range(nLayers):
        fn2 = os.path.join(activ_dir,'allStimsVarExpl_%s.npy'%layers2load[ll])
        var_expl = np.load(fn2)
        varexpl_thisset.append(var_expl)
        
        ncomp2keep = np.where(np.cumsum(var_expl)>min_var_expl/100)
        
        all_ncomps[tr,ii,kk,ll] = ncomp2keep[0][0];
      varexpl_thisinit.append(varexpl_thisset)
    varexpl_thistr.append(varexpl_thisinit)
  varexpl_all.append(varexpl_thistr)
         
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

#%% plot num comp each layer in RAW

tr=4
nInits=1

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)

nunits = np.zeros([nLayers,1])
for ll in range(nLayers):

  nunits[ll] = info['activ_dims'][ll]**2*info['output_chans'][ll]

plt.plot(np.arange(0,np.size(layers2plot),1),nunits,marker='o',color='k',zorder=21-tr)
plt.axhline(8640,color=[0.8, 0.8, 0.8])

xlims = [-1, np.size(layers2plot)]
ylims = [-50,5000]
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Number of total units')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Size of raw activations')    
fig.set_size_inches(10,7)

#%% plot num comp each layer after pca

tr=4
nInits=1

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)


# loop over network initializations
for ii in range(nInits):
  # loop over random image sets
  for kk in range(nImageSets):
    
    ncomp = all_ncomps[tr,ii,kk,:]
  
    plt.plot(np.arange(0,np.size(layers2plot),1),ncomp,color=colors_main[color_inds[tr],:],zorder=21-tr)


xlims = [-1, np.size(layers2plot)]
ylims = [-50,5000]
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
#plt.ylim(ylims)
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Number of components')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Dimensionality after pca, keeping %d pct var\n%s - %s'%(min_var_expl,training_strs[tr],dataset_all))    
fig.set_size_inches(10,7)


#%%
tr=4
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
#ax=fig.add_subplot(1,1,1)
#handles = []
#layers2plot = np.arange(0,nLayers,1)
min_var_expl=90
for ll in range(nLayers):
  # loop over network initializations
  for ii in range(nInits):
    # loop over random image sets
    for kk in range(nImageSets):
      
      varexpl = varexpl_all[tr][ii][kk][ll]
#      nunits = info['activ_dims'][layers2plot[ll]]**2*info['output_chans'][layers2plot[ll]]  
      nunits = len(varexpl)
      
      plt.subplot(npx,npy,ll+1)
      plt.plot(np.arange(0,len(varexpl)), np.cumsum(varexpl)*100,color=colors_main[color_inds[tr],:],zorder=21-tr)

#      ncomp = all_ncomps[tr,ii,kk,ll]
      ncomp = np.where(np.cumsum(varexpl)*100>min_var_expl)[0][0]
      
      plt.title(layer_labels[ll])
      
      xlims = [0, nunits]
      ylims = [-10,110]
      
      plt.axhline(min_var_expl,color=[0.8, 0.8, 0.8])
      plt.axvline(ncomp,color=[0.8, 0.8, 0.8])
      plt.xlim(xlims)
      plt.xticks([0,nunits])
      plt.ylim(ylims)
      #plt.yticks([-0.5,0, 0.5,1])
      if ll==nLayers-1:
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative pct var')
      else:
        
        plt.yticks([])
      #plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Cumulative percent var explained\n%s - lines indicate %d percent var'%(training_strs[tr],min_var_expl))    
fig.set_size_inches(18,10)



