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
import matplotlib.lines as mlines
from copy import deepcopy


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
# values of "delta" to use for fisher information
delta_vals = np.arange(1,10,1)
nDeltaVals = np.size(delta_vals)

#%% get the data ready to go...then can run any below cells independently.
root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))
figfolder = os.path.join(root, 'figures','FisherInfoPop')
plot_full=1
loopSF=0
nSamples = 4
model='vgg16'
#model='pixel'

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
#training_strs=['scratch_imagenet_rot_0_cos_stop_early']
training_strs=['scratch_imagenet_rot_0_stop_early_init_ones']
#training_strs=['scratch_imagenet_rot_22_cos']
#training_strs=['pixels']
#ckpt_strs=['400000']
ckpt_strs=['0']
#ckpt_strs=['00000']
#dataset_str=['FiltIms11Square_SF_0.25']
#dataset_str=['FiltIms11Cos']
dataset_str=['FiltIms14AllSFCos']
#dataset_str=['SquareGratings']
color_ind=2


nTrainingSchemes = np.size(training_strs)
if np.size(dataset_str)==1:
  dataset_str = np.tile(dataset_str,nTrainingSchemes)
  
all_fisher = []
all_deriv_sq = []
all_pooled_var = []

sf_vals = np.logspace(np.log10(0.02),np.log10(.4),6)*140/224
nSF=np.size(sf_vals)
if 'AllSF' in dataset_str[0] or loopSF==0:
  sf_labels=['broadband SF']
  sf2do=[0]
else:
  sf_labels=['%.2f cpp'%sf_vals[sf] for sf in range(nSF)]
  sf2do=np.arange(0,nSF)

tr=0
# load activations for each training scheme
for ii in range(nInits):
  training_str = training_strs[tr]
  ckpt_num = ckpt_strs[tr]
  dataset_all = dataset_str[tr]
  
  param_str = param_strs[ii]
  
  # different versions of the evaluation image set (samples)
  for kk in range(nSamples):
    
    for sf in sf2do:
    
      if kk==0 and 'FiltIms' not in dataset_all and loopSF==0:
        dataset = dataset_all
      elif 'FiltIms' in dataset_all and loopSF==0:
        dataset = '%s_rand%d'%(dataset_all,kk+1)
      elif loopSF==0:
        dataset = '%s%d'%(dataset_all,kk)
      else:
        dataset = '%s_SF_%.2f_rand%d'%(dataset_all,sf_vals[sf],kk+1)
          
      if ii==0 and kk==0:
        info = load_activations.get_info(model,dataset)
      
      if not plot_full:
        if 'pixel' in model:
          save_path = os.path.join(root,'code','fisher_info',model,dataset,'Fisher_info_pixels.npy')
        else:
          save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'Fisher_info_eval_at_ckpt_%s_all_units.npy'%(ckpt_num))
        print('loading from %s\n'%save_path)
        # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
        FI = np.load(save_path)
        
        save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'Deriv_sq_eval_at_ckpt_%s_all_units.npy'%(ckpt_num))
        print('loading from %s\n'%save_path)    
        # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
        d = np.load(save_path)
        
        save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'Pooled_var_eval_at_ckpt_%s_all_units.npy'%(ckpt_num))
        print('loading from %s\n'%save_path)    
        # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
        v = np.load(save_path)
        
      else:
        
        # find the exact number of the checkpoint 
        ckpt_dirs = os.listdir(os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset))
        ckpt_dirs = [dd for dd in ckpt_dirs if 'eval_at_ckpt-%s'%ckpt_num[0:2] in dd and '_full' in dd]
        nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'_full')] for dir in ckpt_dirs]
              
    
        save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'eval_at_ckpt-%s_full'%nums[0],'Fisher_info_all_units.npy')
        print('loading from %s\n'%save_path)
        # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
        FI = np.load(save_path)
        
        save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'eval_at_ckpt-%s_full'%nums[0],'Deriv_sq_all_units.npy')
        print('loading from %s\n'%save_path)    
        # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
        d = np.load(save_path)
        
        save_path = os.path.join(root,'code','fisher_info',model,training_str,param_str,dataset,'eval_at_ckpt-%s_full'%nums[0],'Pooled_var_all_units.npy')      
        print('loading from %s\n'%save_path)    
        # this thing is [nLayer x nSF x nOri x nDeltaValues] in size
        v = np.load(save_path)
     
      if kk==0 and sf==0:
        nLayers = info['nLayers']
         
        nOri = np.shape(FI)[2]
        
        fisher = np.zeros([nSamples, nLayers, nSF, nOri, nDeltaVals])
        deriv_sq = np.zeros([nSamples, nLayers, nSF, nOri, nDeltaVals])
        pooled_var = np.zeros([nSamples, nLayers, nSF, nOri, nDeltaVals])
      
      fisher[kk,:,sf,:,:] = np.squeeze(FI);
      deriv_sq[kk,:,sf,:,:] = np.squeeze(d);
      pooled_var[kk,:,sf,:,:] = np.squeeze(v);

  all_fisher.append(fisher)
  all_deriv_sq.append(deriv_sq)
  all_pooled_var.append(pooled_var)
  
#%% more parameters of interest 
layer_labels = info['layer_labels']
nOri = info['nOri']
ori_axis = np.arange(0, nOri,1)

#% define the orientation bins of interest
# will use these below to calculate anisotropy index
#b = np.arange(22.5,nOri,45)
b = np.arange(22.5,nOri,90)  # baseline
t = np.arange(67.5,nOri,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
c = np.arange(0,nOri,90) # cardinals
o = np.arange(45,nOri,90)  # obliques
#bin_size = 10
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

#%%  Plot selected layers to save
# plot average FISHER INFORMATION curves, overlay NETWORKS, broadband SFs 
# one delta values at a time.
# params for figure saving
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[14,10]

tr=0
dd=3
ylims=[[0, 0.05],[0,0.12],[0,0.08],[0,0.8]]
plt.close('all')

#layers2plot = np.arange(0,nLayers,1)
layers2plot = [0,6,12,18]
plt.figure()
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
sf = 0
# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

    plt.subplot(npx,npy, ll+1)
   
    all_fish= np.squeeze(all_fisher[tr][:,layers2plot[ll],sf,:,dd])
            
    plt.subplot(npx,npy, ll+1)
     # reshape this to [nSamples x nOri]
    fish = deepcopy(np.reshape(all_fish,[nSamples,180],order='F'))
    
    # correct for the different numbers of units in each layer
    nUnits = info['activ_dims'][layers2plot[ll]]**2*info['output_chans'][layers2plot[ll]]  
    print(nUnits)
    fish = fish/nUnits
    
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    if nSamples>1:
      errfish = np.std(fish,0)
    else:
      errfish=np.zeros(np.size(meanfish))
#    for nn in range(nSamples):
#      plt.plot(ori_axis,fish[nn,:],color=cols_sf[sf+nn,1,:])
 
    plt.errorbar(ori_axis,meanfish,errfish,ecolor=colors_main[color_ind,:],color=[0,0,0],zorder=100)

    # finish up this subplot    
    plt.title('%s' % (layer_labels[layers2plot[ll]]))

    plt.xlabel('Orientation (deg)')
    plt.xticks(np.arange(0,181,45))
    plt.xlim([np.min(ori_axis),np.max(ori_axis)])
    plt.ylabel('FI (a.u.)')
    
    plt.ylim(ylims[ll])
    
    for ii in np.arange(0,181,45):
        plt.axvline(ii,color=[0.8, 0.8, 0.8])
            
  
    
# finish up the entire plot   
plt.suptitle('%s -%s' % (training_strs[tr],dataset_all))
figname = os.path.join(figfolder, '%s_FisherInfo.pdf' % (training_str))
plt.savefig(figname, format='pdf',transparent=True)


#%% big plot of all layers together 
# plot average FISHER INFORMATION curves, overlay samples, broadband SFs (ONE NETWORK)
# one delta values at a time.
ylims = [0, 1]

dd=3

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()
blue_colors = cm.Blues(np.linspace(0,1,nSamples))

layers2plot = np.arange(0,nLayers,1)
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
sf = 0
# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

  FI_all_init = np.zeros([nInits,nOri])
  
  for ii in range(nInits):
    
    all_fish= np.squeeze(all_fisher[ii][:,layers2plot[ll],sf,:,dd])
            
    plt.subplot(npx,npy, ll+1)
     # reshape this to [nSamples x nOri]
    fish = deepcopy(np.reshape(all_fish,[nSamples,180],order='F'))
    
    # correct for the different numbers of units in each layer
    nUnits = info['activ_dims'][layers2plot[ll]]**2*info['output_chans'][layers2plot[ll]]  
    fish = fish/nUnits
    
    FI_all_init[ii,:] = np.mean(fish,axis=0)
    
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    if nSamples>1:
      errfish = np.std(fish,0)
    else:
      errfish=np.zeros(np.size(meanfish))
#    for nn in range(nSamples):
#      plt.plot(ori_axis,fish[nn,:],color=cols_sf[sf+nn,1,:])
 
    if nInits==1:
      plt.errorbar(ori_axis,meanfish,errfish,ecolor=colors_main[color_ind,:],color=[0,0,0])
#    plt.plot(ori_axis,meanfish,color=colors_main[color_ind,:])
    
  if nInits>1:
    # get mean and std, plot errorbars.
    meanfish = np.mean(FI_all_init,0)    
    errfish = np.std(FI_all_init,0)
   
    plt.errorbar(ori_axis,meanfish,errfish,ecolor=colors_main[color_ind,:],color=[0,0,0])
#    plt.plot(ori_axis,meanfish,color=[0,0,0])

  # finish up this subplot    
#    plt.ylim(ylims)
  plt.title('%s' % (layer_labels[layers2plot[ll]]))
  if ll==np.size(layers2plot)-1:
     plt.xlabel('Orientation (deg)')
    
#       plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
     plt.xticks(np.arange(0,181,45))
  else:
#       plt.yticks([])
     plt.xticks([])
  for xx in np.arange(0,181,45):
     plt.axvline(xx,color='k')
        
       
#    plt.ylabel('Fisher information (a.u.)')
#plt.legend(['init %d'%ii for ii in init_nums])    
# finish up the entire plot   
plt.suptitle('%s %s - %s\nFisher Information (delta=%d deg)\nBroadband SF' % (training_strs[tr],init_str,dataset_all, delta_vals[dd]))

#%% plot each kind of anisotropy for one network at a time - variability across network initializations

peak_inds=[card_inds, twent_inds,obl_inds]
#this_baseline = [oo for oo in ori_axis if oo not in card_inds and oo not in twent_inds and oo not in obl_inds]
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
#tr2plot = [1,2,3]
# loop over network training schemes (upright versus rot images etc)
for pp in range(np.size(peak_inds)):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nInits,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for ii in range(nInits):

      all_fish= np.squeeze(deepcopy(all_fisher[ii][:,layers2plot[ww1],sf,:,dd]))
      # first average over image sets
      all_fish = np.mean(all_fish,axis=0)
     
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      
#      base_discrim=  all_fish[this_baseline]
      peak_discrim = all_fish[peak_inds[pp]]
      
      # final value for this layer: difference divided by sum 
      aniso_vals[ii,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
    
  # put the line for this spatial frequency onto the plot      
  vals = np.mean(aniso_vals,0)
  errvals = np.std(aniso_vals,0)
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
plt.suptitle('Each type of information bias\n%s-%s\n%s - %s'%(training_strs[tr],init_str,dataset_all,sf_labels[sf]))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, '%s_AnisoEachType.pdf'%training_strs[tr])

#%% plot each kind of anisotropy for one network at a time - variability across image sets

ii=1
peak_inds=[card_inds, twent_inds,obl_inds]
#this_baseline = [oo for oo in ori_axis if oo not in card_inds and oo not in twent_inds and oo not in obl_inds]
plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
#tr2plot = [1,2,3]
# loop over network training schemes (upright versus rot images etc)
for pp in range(np.size(peak_inds)):
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      all_fish= np.squeeze(deepcopy(all_fisher[ii][kk,layers2plot[ww1],sf,:,dd]))
      
      # correct for the different numbers of units in each layer
#      nUnits = info['activ_dims'][layers2plot[ww1]]**2*info['output_chans'][layers2plot[ww1]]
#      all_fish = all_fish/nUnits
    
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      
#      base_discrim=  all_fish[this_baseline]
      peak_discrim = all_fish[peak_inds[pp]]
      
      # final value for this layer: difference divided by sum 
      aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
    
  # put the line for this spatial frequency onto the plot      
  vals = np.mean(aniso_vals,0)
  errvals = np.std(aniso_vals,0)
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
plt.suptitle('Each type of information bias\n%s-init %d\n%s - %s'%(training_strs[tr],init_nums[ii],dataset_all,sf_labels[sf]))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, '%s_AnisoEachType.pdf'%training_strs[tr])
#plt.savefig(figname, format='pdf',transparent=True)
#%% plot average FISHER INFORMATION curves, overlay SPATIAL FREQUENCIES (one network)
# one delta values at a time.

tr=0
dd=3

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
npx = np.ceil(np.sqrt(len(layers2plot)))
npy = np.ceil(len(layers2plot)/npx)
#sf2plot = np.arange(0,nSF,1);
sf2plot=[5]
# loop over layers, making a subplot for each
for ww1 in layers2plot:

  for sf in sf2plot:
  
    all_fish= np.squeeze(all_fisher[tr][:,ww1,sf,:,dd])
            
    plt.subplot(npx,npy,ww1+1)
     # reshape this to [nSamples x nOri]
    fish = np.reshape(all_fish,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    errfish = np.std(fish,0)
    plt.errorbar(ori_axis,meanfish,errfish,ecolor=colors_all[sf,1,:],color=[0,0,0])

#    plt.errorbar(ori_axis,meanfish,errfish,color=cols_sf[sf,1,:])
#    plt.plot(ori_axis,meanfish,color='k')
  # finish up this subplot    
  plt.title('%s' % (layer_labels[ww1]))
  if ww1==layers2plot[-1]:
      plt.xlabel('actual orientation of grating')
      plt.ylabel('Fisher information')
#      plt.legend(['%.2f cpp'%sf_vals[sf] for sf in sf2plot])
      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([])
  for ll in np.arange(0,181,45):
      plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s - %s\nFisher Information (delta=%d deg)\n%s' % (training_strs[tr],dataset_all, delta_vals[dd],sf_labels[sf]))

#%% plot average SQUARED DERIVATIVE (not div by variance yet) curves, overlay samples, broadband SFs (ONE NETWORK)
# one delta values at a time.

tr=0
dd=3

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf = 0
# loop over layers, making a subplot for each
for ww1 in layers2plot:

  
    all_vals= np.squeeze(all_deriv_sq[tr][:,ww1,sf,:,dd])
            
    plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
     # reshape this to [nSamples x nOri]
    vals = np.reshape(all_vals,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanvals = np.mean(vals,0)
    errvals = np.std(vals,0)
#    for nn in range(nSamples):
#      plt.plot(ori_axis,vals[nn,:],color=cols_sf[sf+nn,1,:])
 
   # get mean and std, plot errorbars.
    plt.errorbar(ori_axis,meanvals,errvals,ecolor=colors_main[color_ind,:],color=[0,0,0])

    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Deriv squared')
#        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s - %s\nDeriv squared (delta=%d deg)\n%s' % (training_strs[tr],dataset_all, delta_vals[dd],sf_labels[sf]))

#%% plot average POOLED VARIANCE curves, overlay samples, broadband SFs (ONE NETWORK)
# one delta values at a time.

tr=0
dd=3

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf = 0
# loop over layers, making a subplot for each
for ww1 in layers2plot:

  
    all_vals= np.squeeze(all_pooled_var[tr][:,ww1,sf,:,dd])
            
    plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
     # reshape this to [nSamples x nOri]
    vals = np.reshape(all_vals,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanvals = np.mean(vals,0)
    errvals = np.std(vals,0)
#    for nn in range(nSamples):
#      plt.plot(ori_axis,vals[nn,:],color=cols_sf[sf+nn,1,:])
 
   # get mean and std, plot errorbars.
    plt.errorbar(ori_axis,meanvals,errvals,ecolor=colors_main[color_ind,:],color=[0,0,0])

    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Pooled variance')
#        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   

plt.suptitle('%s - %s\nPooled variance (delta=%d deg)\n%s' % (training_strs[tr],dataset_all, delta_vals[dd],sf_labels[sf]))



#%% plot average FISHER INFORMATION curves, mean over samples, broadband SFs (ONE NETWORK)
# Overlay different delta values to compare shapes.


tr=1

dd2plot = np.arange(0,9,1)
deltacolors = cm.YlGn(np.linspace(0,1,np.size(dd2plot)))
plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf = 0

# loop over layers, making a subplot for each
for ww1 in layers2plot:

    for dd in range(np.size(dd2plot)):
      
        all_fish= np.squeeze(all_fisher[tr][:,ww1,sf,:,dd2plot[dd]])
                
        plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
         # reshape this to [nSamples x nOri]
        fish = np.reshape(all_fish,[nSamples,180],order='F')
       
        # get mean and std, plot errorbars.
        meanfish = np.mean(fish,0)
       
        plt.plot(ori_axis,meanfish, color=deltacolors[dd,:])
    
    # finish up this subplot    
    plt.title('%s' % (layer_labels[ww1]))
    if ww1==layers2plot[-1]:
        plt.xlabel('actual orientation of grating')
        plt.ylabel('Fisher information')
        plt.legend(['Delta=%d deg'%delta_vals[dd2plot[dd]] for dd in range(np.size(dd2plot))])
        plt.xticks(np.arange(0,181,45))
    else:
        plt.xticks([])
    for ll in np.arange(0,181,45):
        plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('%s -%s\nFisher Information\nBroadband SF' % (training_strs[tr], dataset_all))
  

#%%  saving plots of single layers at a time
# plot average FISHER INFORMATION curves, overlay NETWORKS, broadband SFs 
# one delta values at a time.
# params for figure saving
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[10,6]

tr2plot = [1,2,3]
dd=3

plt.rcParams.update({'font.size': 20})
plt.close('all')

#layers2plot = np.arange(0,nLayers,1)
layers2plot = [0,6,12,18]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
sf = 0
# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

    plt.figure()

    for tr in tr2plot:
      all_fish= np.squeeze(all_fisher[tr][:,layers2plot[ll],sf,:,dd])

      fish = np.reshape(all_fish,[nSamples,180],order='F')
     
      # get mean and std, plot errorbars.
      meanfish = np.mean(fish,0)
      errfish = np.std(fish,0)
      if tr==1 or tr==3:
        plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[2,tr,:],color=[0,0,0])
      else:
        plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[1,tr,:],color=[0,0,0])

    # finish up this subplot    
    plt.title('%s' % (layer_labels[layers2plot[ll]]))

    plt.xlabel('Orientation (deg)')
    
#        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
    plt.xticks(np.arange(0,181,45))
    
    for ii in np.arange(0,181,45):
        plt.axvline(ii,color='k')
        
    plt.yticks([])
    plt.ylabel('Fisher information (a.u.)')
    
    # finish up the entire plot   
    plt.title('%s -%s\n%s' % (training_strs[tr],dataset_all,layer_labels[layers2plot[ll]]))
    figname = os.path.join(figfolder, '%s_FisherInfo_%s.pdf' % ('CompareFI',layer_labels[layers2plot[ll]]))
    plt.savefig(figname, format='pdf',transparent=True)


#%% big plot of all layers
#plot average FISHER INFORMATION curves, overlay NETWORKS, broadband SFs
# one delta values at a time.
dd=3
tr2plot = np.arange(0,nTrainingSchemes,1)

plt.rcParams.update({'font.size': 10})
plt.close('all')
plt.figure()

layers2plot = np.arange(0,nLayers,1)
sf = 0
handles = []
# loop over layers, making a subplot for each
for ww1 in layers2plot:

  for tr in range(nTrainingSchemes):
  
    all_fish= np.squeeze(all_fisher[tr][:,ww1,sf,:,dd])
            
    plt.subplot(np.ceil(len(layers2plot)/4), 4, ww1+1)
     # reshape this to [nSamples x nOri]
    fish = np.reshape(all_fish,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    errfish = np.std(fish,0)
    
    plt.errorbar(ori_axis,meanfish,errfish,color=cols_sf[3,tr,:])

  # finish up this subplot    
  plt.title('%s' % (layer_labels[ww1]))
  if ww1==layers2plot[-1]:
      plt.xlabel('actual orientation of grating')
      plt.ylabel('Fisher information')
      plt.legend(training_strs)
      plt.xticks(np.arange(0,181,45))
  else:
      plt.xticks([])
  for ll in np.arange(0,181,45):
      plt.axvline(ll,color='k')
        
# finish up the entire plot   
plt.suptitle('Fisher Information (delta=%d deg)\nBroadband SF' % (delta_vals[dd]))


#%% plot Cardinal (V+H) anisotropy from 5-deg steps, broadband SF
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[10,6]

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr2plot = [0,1]
# loop over network training schemes (upright versus rot images etc)
for tr in tr2plot:
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      all_fish= np.squeeze(all_fisher[tr][kk,ww1,sf,:,dd])
    
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[card_inds]
      
      # final value for this layer: difference divided by sum 
      aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
    
  # put the line for this spatial frequency onto the plot      
  vals = np.mean(aniso_vals,0)
  errvals = np.std(aniso_vals,0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,tr,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,tr,:])


#  finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.legend(['%s'%training_strs[tr] for tr in tr2plot])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)

plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Anisotropy Index')

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\nV+H versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'Aniso_cardinals.pdf')
plt.savefig(figname, format='pdf',transparent=True)

#%% plot 45 deg anisotropy from 5-deg steps, broadband SF

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr2plot = [0]
# loop over network training schemes (upright versus rot images etc)
for tr in tr2plot:
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      all_fish= np.squeeze(all_fisher[tr][kk,ww1,sf,:,dd])
    
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[obl_inds]
      
      # final value for this layer: difference divided by sum 
      aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
    
  # put the line for this spatial frequency onto the plot      
  vals = np.mean(aniso_vals,0)
  errvals = np.std(aniso_vals,0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,tr,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,tr,:])


# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

#plt.legend(['%s'%training_strs[tr] for tr in tr2plot])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)

plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Anisotropy Index')

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\n45-deg versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'Aniso_45deg.pdf')
plt.savefig(figname, format='pdf',transparent=True)
#%% plot 22 deg anisotropy from 5-deg steps, broadband SF

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
handles = []
layers2plot = np.arange(0,nLayers,1)
sf=0
dd=3
tr2plot = [1,2,3]
# loop over network training schemes (upright versus rot images etc)
for tr in tr2plot:
  
  # matrix to store anisotropy index for each layer    
  aniso_vals = np.zeros([nSamples,np.size(layers2plot)])
  
  # loop over network layers
  for ww1 in range(np.size(layers2plot)):

    for kk in range(nSamples):

      all_fish= np.squeeze(all_fisher[tr][kk,ww1,sf,:,dd])
    
      # take the bins of interest to get anisotropy
      base_discrim=  all_fish[baseline_inds]
      peak_discrim = all_fish[twent_inds]
      
      # final value for this layer: difference divided by sum 
      aniso_vals[kk,ww1] = (np.mean(peak_discrim) - np.mean(base_discrim))/(np.mean(peak_discrim) + np.mean(base_discrim))
    
  # put the line for this spatial frequency onto the plot      
  vals = np.mean(aniso_vals,0)
  errvals = np.std(aniso_vals,0)
  myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),vals,color = cols_sf[3,tr,:])
  ax.add_line(myline)   
  handles.append(myline)
  plt.errorbar(np.arange(0,np.size(layers2plot),1),vals,errvals,color=cols_sf[3,tr,:])


# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

#plt.legend(['%s'%training_strs[tr] for tr in tr2plot])

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim(ylims)
plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('Anisotropy Index')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Broadband SF stimuli\n22-deg versus baseline (from 5 deg step discrim)')  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, 'Aniso_22deg.pdf')
plt.savefig(figname, format='pdf',transparent=True)


#%%  saving plots of single layers at a time
# plot average FISHER INFORMATION curves, overlay samples, broadband SFs (ONE NETWORK)
# one delta values at a time.
# params for figure saving
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42    
plt.rcParams['figure.figsize']=[10,6]

tr=0
dd=3

plt.rcParams.update({'font.size': 20})
plt.close('all')

#layers2plot = np.arange(0,nLayers,1)
layers2plot = [0,6,12,18]
npx = np.ceil(np.sqrt(np.size(layers2plot)))
npy = np.ceil(np.size(layers2plot)/npx)
sf = 0
plt.figure()
# loop over layers, making a subplot for each
for ll in range(np.size(layers2plot)):

    
    all_fish= np.squeeze(all_fisher[tr][:,layers2plot[ll],sf,:,dd])
            
    plt.subplot(npx,npy, ll+1)
     # reshape this to [nSamples x nOri]
    fish = np.reshape(all_fish,[nSamples,180],order='F')
   
    # get mean and std, plot errorbars.
    meanfish = np.mean(fish,0)
    errfish = np.std(fish,0)
#    for nn in range(nSamples):
#      plt.plot(ori_axis,fish[nn,:],color=cols_sf[sf+nn,1,:])
 
    if tr==0 or tr==1 or tr==3:
      plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[2,tr,:],color=[0,0,0])
    else:
      plt.errorbar(ori_axis,meanfish,errfish,ecolor=cols_sf[1,tr,:],color=[0,0,0])
    # finish up this subplot    
    plt.title('%s' % (layer_labels[layers2plot[ll]]))
#    if ww1==layers2plot[-1]:
    plt.xlabel('Orientation (deg)')
    
#        plt.legend(['Rand image set %d'%ss for ss in range(nSamples)])
    plt.xticks(np.arange(0,181,45))
    
    for ii in np.arange(0,181,45):
        plt.axvline(ii,color='k')
        
    plt.yticks([])
    plt.ylabel('Fisher information (a.u.)')
    
    # finish up the entire plot   
    plt.title('%s\n%s\n%s' % (training_strs[tr],dataset_all,layer_labels[layers2plot[ll]]))
figname = os.path.join(figfolder, '%s_FisherInfo.pdf' % (training_strs[tr]))
plt.savefig(figname, format='pdf',transparent=True)
