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

# loading all networks at once - 
# [random, trained upright images, trained 22 deg rot images, trained 45 deg rot images, pretrained]
training_strs=['pretrained','pretrained','pretrained','pretrained']
#training_strs=['scratch_imagenet_rot_45_cos']
ckpt_strs=['0','0','0','0']
#ckpt_strs=['400000']
nInits_list = [1,1,1,1]
#nInits_list = [1]
color_inds=[1,1,1,1]
#color_inds=[3]

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

#dataset_str=['FiltIms14AllSFCos_bwk100','FiltIms14AllSFCos_bwk215','FiltIms14AllSFCos']
dataset_str=['FiltIms14AllSFCos_bwk100','FiltIms14AllSFCos_bwk215','FiltIms14AllSFCos_bwk464','FiltIms14AllSFCos']
nTrainingSchemes = np.size(training_strs)

 # values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)
nDeltaVals = np.size(delta_vals)

bwk_vals =[100,215, 464,1000]
fwhm_vals = [36.8, 24.8, 16.8, 11.5]

#sf_labels=['broadband SF']
nSF=1
sf=0

#%% load the data (Fisher information calculated from each layer)
all_fisher = []

# load activations for each training set of images (training schemes)
for tr in range(nTrainingSchemes):
  training_str = training_strs[tr]
  ckpt_num = ckpt_strs[tr]
  dataset_all = dataset_str[tr]
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
       
      all_fisher[tr,ii,kk,:,sf,:,:] = np.squeeze(FI);
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

#%%  Plot Fisher information for 4 example layers - overlaying different bandwidths. 
layers2plot = [0,6,12,18]
tr2plot =[0,1,2,3] 
cols_grad = cm.Blues(np.linspace(0,1,len(tr2plot)+2))
cols_grad =cols_grad[2:,:]

init2plot = [0]
sf=0
dd=3

plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]

plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
ylims=[[0, 0.1],[0,0.1],[0,0.1],[0,0.7]]

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  ax=plt.subplot(npx,npy, ll+1)
  allh=[]
  FI_all_init = np.zeros([len(init2plot),nOri])
  
  for tr in range(len(tr2plot)):
    for ii in init2plot:
      
      fish= all_fisher[tr2plot[tr],ii,:,layers2plot[ll],sf,:,dd] 
      # correct for the different numbers of units in each layer
      nUnits = info['activ_dims'][layers2plot[ll]]**2*info['output_chans'][layers2plot[ll]]  
      fish = fish/nUnits  
      # average over image sets
      FI_all_init[ii,:] = np.mean(fish,axis=0)

    meanfish = np.mean(FI_all_init,0)    
#    errfish = np.std(FI_all_init,0)

    plt.plot(ori_axis,meanfish,color=cols_grad[tr,:])
      
    myline = mlines.Line2D(ori_axis,meanfish,color=cols_grad[tr,:])
    ax.add_line(myline)   
    allh.append(myline)
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

#  if ll==len(layers2plot)-1:
  plt.xlabel('Orientation (deg)')
  plt.xticks(np.arange(0,181,45))
  plt.xlim([np.min(ori_axis),np.max(ori_axis)])
  plt.ylabel('FI (a.u.)')
  plt.ylim(ylims[ll])
#  if ll==len(layer2plot)-1
#  plt.legend(allh, ['fwhm = %.1f deg'%fwhm_vals[bw] for bw in range(len(fwhm_vals))])
#  else:
#    plt.xticks([])
#    plt.yticks([])
  
  
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
      
# finish up the entire plot   
plt.suptitle('%s' % (training_strs[tr]))
figname = os.path.join(figfolder, 'FIlines_pretrained_varyOrientBW.pdf')
plt.savefig(figname, format='pdf',transparent=True)

#%%  Plot Fisher information for all layers - overlaying different bandwidths. 
layers2plot = np.arange(0,nLayers)

tr2plot =[0,1,2,3] 
cols_grad = cm.Blues(np.linspace(0,1,len(tr2plot)+2))
cols_grad =cols_grad[2:,:]

init2plot = [0]
sf=0
dd=3

plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'font.size': 10})
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[18,10]

plt.close('all')

plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)

# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  ax=plt.subplot(npx,npy, ll+1)
  allh=[]
  FI_all_init = np.zeros([len(init2plot),nOri])
  
  for tr in range(len(tr2plot)):
    for ii in init2plot:
      
      fish= all_fisher[tr2plot[tr],ii,:,layers2plot[ll],sf,:,dd] 
      # correct for the different numbers of units in each layer
      nUnits = info['activ_dims'][layers2plot[ll]]**2*info['output_chans'][layers2plot[ll]]  
      fish = fish/nUnits  
      # average over image sets
      FI_all_init[ii,:] = np.mean(fish,axis=0)

    meanfish = np.mean(FI_all_init,0)    
#    errfish = np.std(FI_all_init,0)

    plt.plot(ori_axis,meanfish,color=cols_grad[tr,:])
      
    myline = mlines.Line2D(ori_axis,meanfish,color=cols_grad[tr,:])
    ax.add_line(myline)   
    allh.append(myline)
  # finish up this subplot    
  plt.title('%s' % (layer_labels[layers2plot[ll]]))

  if ll==len(layers2plot)-1:
    plt.xlabel('Orientation (deg)')
    plt.xticks(np.arange(0,181,45))
    plt.xlim([np.min(ori_axis),np.max(ori_axis)])
    plt.ylabel('FI (a.u.)')
    
    plt.legend(allh, ['fwhm = %.1f deg'%fwhm_vals[bw] for bw in range(len(fwhm_vals))])
  else:
    plt.xticks([])
#    plt.yticks([])
  
  
  for xx in np.arange(0,181,45):
      plt.axvline(xx,color=[0.8, 0.8, 0.8])
      
# finish up the entire plot   
plt.suptitle('%s' % (training_strs[tr]))

#%% plot just cardinal bias across layers, for different bandwidths overlaid.
# which type of FIB to plot?
pp=0  # set to 0, 1 or 2 to plot [FIB-0, FIB-22, FIB-45]
tr2plot=[0,1,2,3] 

cols_grad = cm.Blues(np.linspace(0,1,len(tr2plot)+2))
cols_grad =cols_grad[2:,:]

ii=0
sf=0
dd=3
peak_inds=[card_inds, twent_inds,obl_inds]
lstrings=['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline']
ii=0
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

allh=[]
layers2plot = np.arange(0,nLayers,1)

alpha=0.01;
# matrix to store anisotropy index for each layer    
aniso_vals = np.zeros([len(tr2plot),nImageSets,np.size(layers2plot)])

# loop over network layers
for ww1 in range(np.size(layers2plot)):
  # loop over networks with each training set
  for tr in range(len(tr2plot)):

    ii=0
    for kk in range(nImageSets):

      # FI is nOri pts long
      vals= all_fisher[tr2plot[tr],ii,kk,layers2plot[ww1],sf,:,dd]
      
      # take the bins of interest to get bias
      base_discrim=  vals[baseline_inds]
      peak_discrim = vals[peak_inds[pp]]
      
      # final value for this FIB: difference divided by sum 
      aniso_vals[tr,kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))

# error bars are across 4 image sets
for tr in range(len(tr2plot)):    
  vals = np.squeeze(np.mean(aniso_vals[tr,:,:],0))
  errvals = np.squeeze(np.std(aniso_vals[tr,:,:],0)) 
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_grad[tr,:],zorder=21)

  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color=cols_grad[tr,:])
  ax.add_line(myline)   
  allh.append(myline)
  
# finish up the entire plot
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]
plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Information Bias')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.legend(allh, ['fwhm = %.1f deg'%fwhm_vals[bw] for bw in range(len(fwhm_vals))]) 

plt.suptitle('FIB: %s\npre-trained model with different orientation BW'%(lstrings[pp]))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'FIB_pretrained_varyOrientBW.pdf')
plt.savefig(figname, format='pdf',transparent=True)



#%% plot FI in cardinal and oblique bins, versus BW of orient filter
# which type of FIB to plot?
pp2plot=[0,2]  # set to 0, 1 or 2 to plot [FIB-0, FIB-22, FIB-45]
tr2plot=[0,1,2,3] 

#cols_grad = cm.Blues(np.linspace(0,1,len(tr2plot)+2))
#cols_grad =cols_grad[2:,:]

ii=0
sf=0
dd=3
peak_inds=[card_inds, twent_inds,obl_inds]
lstrings=['0 + 90', '67.5 + 157.5', '45 + 135']
bin_color_inds = [1,2,3]

ii=0
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

allh=[]
layers2plot = np.arange(0,nLayers,1)

alpha=0.01;

layers2plot =np.asarray([0,6,12,18])

for ll in range(len(layers2plot)):
  
  plt.subplot(2,2,ll+1)
  allh=[]
  # loop over networks with each training set
  for pp in range(len(pp2plot)):
  
    ii=0
    # matrix to store anisotropy index for each layer    
    fi_vals = np.zeros([len(tr2plot),nImageSets])
    for bw in range(len(tr2plot)):
      for kk in range(nImageSets):
    
        # FI is nOri pts long
        vals= all_fisher[tr2plot[bw],ii,kk,layers2plot[ll],sf,:,dd]
  
        # final value for this FIB: difference divided by sum 
        fi_vals[bw,kk] = np.mean(vals[peak_inds[pp]])
  
    # error bars are across 4 image sets
#    for bw in range(len(tr2plot)):    
    vals = np.squeeze(np.mean(fi_vals,1))
    errvals = np.squeeze(np.std(fi_vals,1)) 
    plt.errorbar(fwhm_vals,vals,errvals,color=colors_main[bin_color_inds[pp2plot[pp]],:],zorder=21)
  
    myline = mlines.Line2D(fwhm_vals,vals,color=colors_main[bin_color_inds[pp2plot[pp]],:])
    ax.add_line(myline)   
    allh.append(myline)
    
# finish up the entire plot
#  ylims = [-0.5,1]
  xlims = [10,40]
#  plt.plot(xlims, [0,0], 'k')
  plt.xlim(xlims)
#  plt.ylim(ylims)
#  plt.yticks([-0.5,0, 0.5,1])
  plt.ylabel('Fisher information')
  plt.xticks(fwhm_vals)
  plt.xlabel('FWHM of orient filter (deg)')
  
  plt.legend(allh, [lstrings[pp] for pp in pp2plot]) 
  plt.title(layer_labels[layers2plot[ll]])
  
plt.suptitle('Fisher info versus bandwith of orient filter\npre-trained model') 
fig.set_size_inches(14,14)
#figname = os.path.join(figfolder, 'FIB_pretrained_varyOrientBW.pdf')
#plt.savefig(figname, format='pdf',transparent=True)


#%% plot average FI across orientations - for different bandwidths overlaid.
#pp=0  # set to 0, 1 or 2 to plot [FIB-0, FIB-22, FIB-45]
tr2plot=[0,1,2,3] # plot just pre-trained and random model here.

cols_grad = cm.Blues(np.linspace(0,1,len(tr2plot)+2))
cols_grad =cols_grad[2:,:]

ii=0
sf=0
dd=3
#peak_inds=[card_inds, twent_inds,obl_inds]
#lstrings=['0 + 90 vs. baseline', '67.5 + 157.5 vs. baseline', '45 + 135 vs. baseline']
ii=0
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

allh=[]
layers2plot = np.arange(0,nLayers,1)

alpha=0.01;
# matrix to store anisotropy index for each layer    
aniso_vals = np.zeros([len(tr2plot),nImageSets,np.size(layers2plot)])

# loop over network layers
for ww1 in range(np.size(layers2plot)):
  # loop over networks with each training set
  for tr in range(len(tr2plot)):

    ii=0
    for kk in range(nImageSets):

      # FI is nOri pts long
      vals= all_fisher[tr2plot[tr],ii,kk,layers2plot[ww1],sf,:,dd]
      
      # final value for this FIB: difference divided by sum 
      aniso_vals[tr,kk,ww1] = np.mean(vals)

# error bars are across 4 image sets
for tr in range(len(tr2plot)):    
  vals = np.squeeze(np.mean(aniso_vals[tr,:,:],0))
  errvals = np.squeeze(np.std(aniso_vals[tr,:,:],0)) 
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_grad[tr,:],zorder=21)

  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color=cols_grad[tr,:])
  ax.add_line(myline)   
  allh.append(myline)
  
# finish up the entire plot
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]
plt.plot(xlims, [0,0], 'k')
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Average FI (a.u.)')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

plt.legend(allh, ['fwhm = %.1f deg'%fwhm_vals[bw] for bw in range(len(fwhm_vals))]) 
plt.suptitle('FI averaged over all orients\npre-trained model with different orientation BW')  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'FImean_pretrained_varyOrientBW.pdf')
plt.savefig(figname, format='pdf',transparent=True)
