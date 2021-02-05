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

#%% define which network to load - uncomment one of these lines

#training_strs = ['scratch_imagenet_rot_0_stop_early']   # a randomly initialized, un-trained model
#training_strs = ['scratch_imagenet_rot_0_cos']  # model trained on upright images
#training_strs = ['scratch_imagenet_rot_22_cos']   # model trained on 22 deg rot iamges
#training_strs = ['scratch_imagenet_rot_22_cos']   # model trained on 45 deg rot images
training_strs = ['pretrained']   # a pre-trained model 

#%% define other basic parameters
set2load=[0]
nImageSets=len(set2load)
kk=0

sf2load = [3]

model='vgg16'
param_str='params1'
dataset_str=['CosGratingsVeryHigh']

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
  init_nums=[0]
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
     sf_vals = info['sf_vals']
     if 'VeryHigh' in dataset:
       sf_vals = np.logspace(np.log10(0.25), np.log10(0.5), 6)
     nSF = np.size(np.unique(info['sflist']))
     nLayers = info['nLayers']      
     layer_labels = info['layer_labels']    
     nOri = info['nOri']
     ori_axis = np.arange(0, nOri,1)
         
     # initialize these arrays (will be across all init of the network)    
     coords_all = []    
     fit_pars_all = [] 
     r2_all = []  
     prop_zero = np.zeros((nInits, nLayers))
     prop_const = np.zeros((nInits, nLayers))
     ntotal = np.zeros((nInits, nLayers))
     
     fastpars_all = []
     
     
  fn=os.path.join(save_path,'PropZeroUnits_eval_at_ckpt_%s0000.npy'%ckpt_num[0:2])
  print('loading from %s\n'%fn)
  prop_zero[ii,:] = np.squeeze(np.load(fn))
  
  fn=os.path.join(save_path,'PropConstUnits_eval_at_ckpt_%s0000.npy'%ckpt_num[0:2])
  print('loading from %s\n'%fn)
  prop_const[ii,:] = np.squeeze(np.load(fn))
  
  fn=os.path.join(save_path,'TotalUnits_eval_at_ckpt_%s0000.npy'%ckpt_num[0:2])
  print('loading from %s\n'%fn)
  ntotal[ii,:] = np.squeeze(np.load(fn))
     
  coords = []
  fit_pars = [] 
  r2 = []   
  fastpars = []
  r2_allsf = None
  pars_allsf = None
  # loop over layers and load fit parameters
  for ll in range(nLayers):
      
     # load coordinates of each network unit (spatial position and channel number)
     # [nUnits x 3] where third dim is [H,W,C]
     file_name =os.path.join(save_path,'%s_coordsHWC_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
     print('loading from %s\n'%file_name)
     coords.append(np.load(file_name))
    
     #load the fit parameters [nUnits x nSF x nPars]
     # in par dimension, [0,1,2,3,4] are [center, k, amplitude, baseline, FWHM size]
     file_name= os.path.join(save_path,'%s_fastpars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
     print('loading from %s\n'%file_name)
     fastpars.append(np.load(file_name).item())
      
     for sf in sf2load:
        # find the random seed(s) for the jitter that was used
        files=os.listdir(os.path.join(save_path))
        [jitt_file] = [ff for ff in files if '%s_sf%d_set%d_fit_jitter'%(layer_labels[0],sf,set2load[kk]) in ff and 'pars' in ff];  
        rand_seed_str = jitt_file[jitt_file.find('jitter')+7:jitt_file.find('jitter')+13]  
                
        # load fit r2 [nUnits x nSF x nImageSets] 
        file_name= os.path.join(save_path,'%s_sf%d_set%d_fit_jitter_%s_r2_each_sample_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],sf,set2load[kk],rand_seed_str,ckpt_num[0:2]))
        print('loading from %s\n'%file_name)
        this_r2=np.expand_dims((np.load(file_name)),2)
        
        # load the fit parameters [nUnits x nSF x nPars]
        # in par dimension, [0,1,2,3,4] are [center, k, amplitude, baseline, FWHM size]
        file_name= os.path.join(save_path,'%s_sf%d_set%d_fit_jitter_%s_pars_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],sf,set2load[kk],rand_seed_str,ckpt_num[0:2]))
        print('loading from %s\n'%file_name)
        this_pars = (np.load(file_name))
 
        if sf==sf2load[0]:
          r2_allsf = np.zeros((np.shape(this_r2)[0],nSF,np.shape(this_r2)[2]))
          pars_allsf = np.zeros((np.shape(this_pars)[0],nSF,np.shape(this_pars)[2]))
          
        r2_allsf[:,sf,:] = np.squeeze(this_r2,axis=1)
        pars_allsf[:,sf,:] = np.squeeze(this_pars)
        
     r2.append(r2_allsf)
     fit_pars.append(pars_allsf)

          
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
colors_all = np.concatenate((colors_all_1[np.arange(2,nsteps,1),:,:],
                                          colors_all_2[np.arange(2,nsteps,1),:,:],
                                          colors_all_3[np.arange(2,nsteps,1),:,:],
                                          colors_all_4[np.arange(2,nsteps,1),:,:],
                                          colors_all_2[np.arange(2,nsteps,1),:,:]),axis=1)

int_inds = [3,3,3,3]
colors_main = np.asarray([colors_all[int_inds[ii],ii,:] for ii in range(np.size(int_inds))])
colors_main = np.concatenate((colors_main, colors_all[5,1:2,:]),axis=0)
# plot the color map
#plt.figure();plt.imshow(np.expand_dims(colors_main,axis=0))
cols_sf = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
cols_sf = cols_sf[np.arange(2,8,1),:,:]



#%% load actual tfs from a single layer (slow)
ll=0
tr=0
ii=0
kk=0
fn = os.path.join(save_path,'%s_all_responsive_units_eval_at_ckpt_%s0000.npy'%(layer_labels[ll],ckpt_num[0:2]))
print('loading from %s\n'%fn)
tfs=np.load(fn)

#%% make plots of randomly selected units, no fitting
plt.close('all')
sf = 4

nUnits = np.shape(tfs)[1]
nUnitsPlot = 6

nOri=180

np.random.seed(124394)  
units2plot = np.random.choice(np.arange(0,np.shape(tfs)[1]), nUnitsPlot, replace='False')

npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)

  real_y = np.squeeze(tfs[ii,units2plot[uu],sf,0:nOri])

  plt.plot(ori_axis, real_y, color=colors_main[color_ind,:]) 
  plt.xticks([0,45,90,135,180])
  plt.xlabel('Grating Orientation')
  plt.ylabel('Unit response')

plt.suptitle('Gratings with %.2f cycles/pixel\nExample tuning curves for %s model, %s'%(sf_vals[sf],training_str,layer_labels[ll]))
figname = os.path.join(figfolder, 'Example_TFs_gratingsSF%.2f.pdf'%sf_vals[sf])
plt.savefig(figname, format='pdf',transparent=True)


#%% plot the proportion of units above r2 threshold, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
#layers2plot=[0]
sf=3

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
plt.suptitle('Prop of units with r2>%.2f\n%s:  Single phase gratings, sf=%.2f'%(r2_cutoff,training_str,sf_vals[sf]))  
fig.set_size_inches(10,7)
figname = os.path.join(figfolder, '%s_pct_units_vs_layer.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)
#%% plot the proportion of non-responsive units (zero or constant) as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
ax=plt.subplot(1,1,1)
layers2plot = np.arange(0,nLayers,1)

lh=[]
prop_vals = np.zeros((nInits, nLayers))
for ii in range(nInits):
  for ll in range(nLayers):
    prop_vals[ii,ll] = 1-np.shape(fastpars_all[ii][ll]['maxresp'])[1]/ntotal[ii][ll]

# put the line for this spatial frequency onto the plot      
meanvals = np.mean(prop_vals,axis=0)
sdvals = np.std(prop_vals,axis=0)
plt.plot(np.arange(0,np.size(layers2plot),1),meanvals,marker='o',color = colors_all[1,color_ind,:])
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),meanvals,color = colors_all[1,color_ind,:])
ax.add_line(myline)   
lh.append(myline)

prop_vals = prop_zero  
meanvals = np.mean(prop_vals,axis=0)
sdvals = np.std(prop_vals,axis=0)
plt.plot(np.arange(0,np.size(layers2plot),1),meanvals,marker='o',color = colors_all[3,color_ind,:])
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),meanvals,color = colors_all[3,color_ind,:])
ax.add_line(myline)   
lh.append(myline)

prop_vals = prop_const
meanvals = np.mean(prop_vals,axis=0)
sdvals = np.std(prop_vals,axis=0)
plt.plot(np.arange(0,np.size(layers2plot),1),meanvals,marker='o',color = colors_all[5,color_ind,:])
myline = mlines.Line2D(np.arange(0,np.size(layers2plot),1),meanvals,color = colors_all[5,color_ind,:])
ax.add_line(myline)   
lh.append(myline)

# finish up this subplot 
ylims = [-0.5,1]
xlims = [-1, np.size(layers2plot)]

plt.plot(xlims, [0,0], 'k')
plt.xlim(xlims)
plt.ylim([-0.2,1])
plt.yticks([0,0.5,1])
plt.ylabel('Proportion of units')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)
plt.legend(lh,['total non-responsive','zero resp','constant nonzero resp'])
# finish up the entire plot
plt.suptitle('Prop of non-responsive (for ANY tested spatial freq)\n%s, %s'%(training_str, dataset))  
fig.set_size_inches(10,7)
#figname = os.path.join(figfolder, '%s_pct_units_vs_layer.pdf' % (training_str))
#plt.savefig(figname, format='pdf',transparent=True)

#%% Plot histogram of tuning centers, for all layers in the network.
plt.rcParams.update({'font.size': 10})

# bins for the histograms
ori_bin_size=1
ori_bins = np.arange(-ori_bin_size/2,nOri+ori_bin_size,ori_bin_size)
bin_centers = ori_bins[0:np.size(ori_bins)-1]+ori_bin_size/2

#layers2plot=[0]
layers2plot=np.arange(0,nLayers)
sf=3

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
      
plt.suptitle('%s %s sf=%.2f\nFit Centers - All units r2>%.2f'%(training_str,dataset,sf_vals[sf],r2_cutoff));

#%% Plot K versus the center (scatter plot), all layers in the network
plt.rcParams.update({'font.size': 10})

layers2plot=np.arange(0,nLayers,1)
plt.rcParams['figure.figsize']=[14,10]
pp2plot=1 # index of k in the parameters array
ppname='k'
sf=3
maxpts=10000
ylims = [-5,150]
alpha=0.5
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
  
plt.suptitle('%s %s sf=%.2f\n%s vs center - All units r2>%.2f'%(training_str,dataset,sf_vals[sf],ppname,r2_cutoff));

#%% Plot amplitude versus the center (scatter plot), all layers in the network
plt.rcParams.update({'font.size': 10})

layers2plot=np.arange(0,nLayers,1)
plt.rcParams['figure.figsize']=[14,10]
pp2plot=2 # index of k in the parameters array
ppname='amp'
sf=4
maxpts=10000
ylims = [-5,150]
alpha=0.5
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
#  plt.ylim(ylims)
  plt.xlabel('Center (deg)')
  plt.ylabel(ppname)
  plt.xticks(np.arange(0,181,45))
  
  for xx in np.arange(45,180,45):
    plt.axvline(xx,color=[0.95, 0.95, 0.95],linewidth=2)
  
plt.suptitle('%s %s sf=%.2f\n%s vs center - All units r2>%.2f'%(training_str,dataset,sf_vals[sf],ppname,r2_cutoff));

#%% plot the distribution of FWHM, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
sf=3

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
plt.suptitle('FWHM (deg) averaged over units with r2>%.2f\n%s, %s sf=%.2f'%(r2_cutoff,training_str, dataset,sf_vals[sf]))  
fig.set_size_inches(10,7)
#%% plot the distribution of k, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
sf=4

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
#plt.ylim([-10,5000])
#plt.yticks([-0.5,0, 0.5,1])
plt.ylabel('k (a.u.)')
plt.xticks(np.arange(0,np.size(layers2plot),1),[layer_labels[ii] for ii in layers2plot],rotation=90)

# finish up the entire plot
plt.suptitle('Concentration parameter (k) averaged over units with r2>%.2f\n%s, %s sf=%.2f'%(r2_cutoff,training_str, dataset,sf_vals[sf]))  
  
fig.set_size_inches(10,7)

#%% plot the distribution of amplitudes, as a function of layer

plt.rcParams.update({'font.size': 14})
plt.close('all')
fig=plt.figure()
layers2plot = np.arange(0,nLayers,1)
sf=4

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
plt.suptitle('Amplitude of TF averaged over units with r2>%.2f\n%s, %s sf=%.2f'%(r2_cutoff,training_str, dataset,sf_vals[sf]))  

fig.set_size_inches(10,7)
#%% plot spatial position of all well-fit units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=3;
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

plt.suptitle('%s\n%s: sf=%.2f\n%s'%(training_str,dataset,sf_vals[sf],layer_labels[ll]))

#%% plot spatial position of all cardinal-tuned units

plt.close('all')
pp=0;
alpha=0.1
r2_here=0.4
nspat = info['activ_dims'][ll]
tr=0;
ll=0;
ii=0
sf2plot = [3]


for sf in sf2plot:
  plt.figure();
 
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
  yvals = nspat - coords_all[ii][ll][units_good,0]  # flip the y axis so orientations go clockwise from vertical
  
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
  
  plt.suptitle('%s\n%s: sf=%.2f\n%s'%(training_str,dataset,sf_vals[sf],layer_labels[ll]))

#%% plot spatial position of all oblique-tuned units
plot_jitt=1
plt.close('all')
plt.figure();
tr=0;
ll=0;
sf=3;
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

o1_units = np.where(np.logical_and(np.logical_and(centers>40, centers<50), amps>0))[0]
o2_units = np.where(np.logical_and(np.logical_and(centers>130, centers<140), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ii][ll][units_good,1]
yvals = nspat - coords_all[ii][ll][units_good,0]  # flip the y axis so orientations go clockwise from vertical

plt.subplot(1,2,1)
plt.plot(xvals[o1_units],yvals[o1_units],'.',color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('45-tuned units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.subplot(1,2,2)
plt.plot(xvals[o2_units],yvals[o2_units],'.',color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('135-tuned units')
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.suptitle('%s\n%s: sf=%.2f\n%s'%(training_str,dataset,sf_vals[sf],layer_labels[ll]))
figname = os.path.join(figfolder, 'SpatialDistTuning_Gratings_HighSF.pdf')

#%% plot spatial position of all units with max value at cardinals (no fitting)
plot_jitt=1
plt.close('all')
plt.figure();
ori1 = 45
ori2=135
tr=0;
ll=0;
sf=4;
pp=0;
alpha=0.1
r2_here=0.4
nspat = info['activ_dims'][ll]
#r2_here=r2_cutoff
ii=0
kk=0
rvals  = deepcopy(np.squeeze(np.mean(r2_all[ii][ll][:,sf,:],axis=1)))
rvals[np.isnan(rvals)] = -1000

maxvals = fastpars_all[ii][ll]['maxori'][kk,:,sf]
max_ori_in_range = np.where(np.logical_and(np.logical_and(maxvals>(ori1-5), maxvals<(ori1+5)), rvals>r2_here))[0]
inds2plot1 = max_ori_in_range
max_ori_in_range = np.where(np.logical_and(np.logical_and(maxvals>(ori2-5), maxvals<(ori2+5)), rvals>r2_here))[0]
inds2plot2 = max_ori_in_range

## values to plot  
#rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][ll][:,sf,:],axis=1)))
#rvals[np.isnan(rvals)] = -1000
#units_good = np.where(rvals>r2_here)[0]
#centers = deepcopy(np.squeeze(fit_pars_all[ii][ll][units_good,sf,0]))
#amps = deepcopy(np.squeeze(fit_pars_all[ii][ll][units_good,sf,2]))
#
#horiz_units = np.where(np.logical_and(np.logical_and(centers>85, centers<95), amps>0))[0]
#vert_units = np.where(np.logical_and(np.logical_or(centers>175, centers<5), amps>0))[0]
# coords goes [vertical pos, horizontal pos, output channel number]
xvals = coords_all[ii][ll][:,1]
yvals = nspat - coords_all[ii][ll][:,0]  # flip the y axis so orientations go clockwise from vertical

plt.subplot(1,2,1)
plt.plot(xvals[inds2plot1],yvals[inds2plot1],'.',color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('%d-preferring units'%ori1)
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.subplot(1,2,2)
plt.plot(xvals[inds2plot2],yvals[inds2plot2],'.',color=colors_main[color_ind,:],markersize=2,alpha=alpha)
plt.xlabel('horizontal unit position')
plt.ylabel('vertical unit position')
plt.axis('square')
plt.title('%d-preferring units'%ori2)
plt.xlim([0,info['activ_dims'][ll]])
plt.ylim([0,info['activ_dims'][ll]])

plt.suptitle('%s\n%s\n%s'%(training_str,dataset,layer_labels[ll]))

figname = os.path.join(figfolder, 'SpatialDistTuning_Gratings_HighSF.pdf')
#plt.savefig(figname, format='pdf',transparent=True)


#%% plot spatial positions of units with high slope regions near 90
plt.close('all')

alpha=1
ll=0
ii=0
kk=0
sf2plot = np.arange(0,nSF,1)
nspat = info['activ_dims'][ll]
nUnitsPlot=1000000

for sf in sf2plot:
  
  plt.figure();
  maxvals = fastpars_all[ii][ll]['maxsqslopeori'][kk,:,sf]
  slope_vals = fastpars_all[ii][ll]['maxsqslopeval'][kk,:,sf]
  
  plt.subplot(1,2,1)
  
  max_slope_in_range = np.where(np.logical_and(maxvals>85, maxvals<95))[0]
  highest_slope_vals = np.flipud(np.argsort(slope_vals))
  highest_slope_vals = highest_slope_vals[np.isin(highest_slope_vals,max_slope_in_range)]
  inds2plot = highest_slope_vals[0:nUnitsPlot]
  
  x_coords = coords_all[ii][ll][:,1]
  y_coords = nspat - coords_all[ii][ll][:,0] # flip the y axis so orientations go clockwise from vertical
  plt.plot(x_coords[inds2plot],y_coords[inds2plot],'.', color=colors_main[color_ind,:],markersize=2,alpha=alpha)
  plt.xlabel('horizontal unit position')
  plt.ylabel('vertical unit position')
  plt.axis('square')
  plt.title('units with highest slope near 90')
  plt.xlim([0,nspat])
  plt.ylim([0,nspat])
  
  plt.subplot(1,2,2)
  max_slope_in_range = np.where(np.logical_and(maxvals>40, maxvals<50))[0]
  highest_slope_vals = np.flipud(np.argsort(slope_vals))
  highest_slope_vals = highest_slope_vals[np.isin(highest_slope_vals,max_slope_in_range)]
  inds2plot = highest_slope_vals[0:nUnitsPlot]
  
  x_coords = coords_all[ii][ll][:,1]
  y_coords = nspat - coords_all[ii][ll][:,0] # flip the y axis so orientations go clockwise from vertical
  plt.plot(x_coords[inds2plot],y_coords[inds2plot],'.', color=colors_main[color_ind,:],markersize=2,alpha=alpha)
  plt.xlabel('horizontal unit position')
  plt.ylabel('vertical unit position')
  plt.axis('square')
  plt.title('units with highest slope near 45')
  plt.xlim([0,nspat])
  plt.ylim([0,nspat])
  
  plt.suptitle('%s\n%s: sf=%.2f\n%s'%(training_str,dataset,info['sf_vals'][sf],layer_labels[ll]))

#%% plot distribution of the maximum values of tuning functions, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')

npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
kk=0
sf2plot = np.arange(0,nSF,1)

for sf in sf2plot:
  plt.figure();
  layers2do = np.arange(0,nLayers,1)
  for ll in range(nLayers):
    plt.subplot(npx,npy,ll+1)
    
    maxvals = fastpars_all[ii][ll]['maxori'][kk,:,sf]
    maxvals = maxvals[maxvals!=0]
    plt.hist(maxvals,bins=np.arange(0,nOri,1),color=cols_sf[sf,:])
    plt.yticks([])
    if ll==len(layers2do)-1:
      plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
    else:
      plt.xticks([])
    
    plt.title(layer_labels[ll])
    
  plt.suptitle('Max of orient tuning functions\n%s, %s, sf=%.2f'%(training_str,dataset, sf_vals[sf]))

#%% plot distribution of the minimum values of tuning functions, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')

npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
kk=2
sf2plot = np.arange(0,nSF,1)

for sf in sf2plot:
  plt.figure();
  layers2do = np.arange(0,nLayers,1)
  for ll in range(nLayers):
    plt.subplot(npx,npy,ll+1)
    
    minvals = fastpars_all[ii][ll]['minori'][kk,:,sf]
    minvals = minvals[minvals!=0]
    plt.hist(minvals,bins=np.arange(0,nOri,1),color=cols_sf[sf,:])
    plt.yticks([])
    if ll==len(layers2do)-1:
      plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
    else:
      plt.xticks([])
    
    plt.title(layer_labels[ll])
    
  plt.suptitle('Minima of orient tuning functions\n%s, %s, sf=%.2f'%(training_str,dataset, sf_vals[sf]))

#%% plot distribution of the maximum slope regions of each unit's tuning function, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')
#plt.figure();
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
kk=2
sf2plot = np.arange(0,nSF,1)

for sf in sf2plot:
  plt.figure();
  layers2do = np.arange(0,nLayers,1)
  for ll in range(nLayers):
    plt.subplot(npx,npy,ll+1)
    
    maxvals = fastpars_all[ii][ll]['maxsqslopeori'][kk,:,sf]
    maxvals = maxvals[maxvals!=0]
    plt.hist(maxvals,bins=np.arange(0,nOri,1),color=cols_sf[sf,:])
    plt.yticks([])
    if ll==len(layers2do)-1:
      plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
    else:
      plt.xticks([])
    
    plt.title(layer_labels[ll])
    
  plt.suptitle('max slope regions of orient tuning functions\n%s, %s, sf=%.2f'%(training_str,dataset, sf_vals[sf]))

#%% plot mean resp vs the maximum slope regions of each unit's tuning function, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
kk=0
layers2do = np.arange(0,nLayers,1)
nUnitsPlot=10000
alpha=0.1

sf2plot = np.arange(0,nSF,1)
for sf in sf2plot:
  plt.figure();
 
  for ll in range(nLayers):
    plt.subplot(npx,npy,ll+1)
    
    maxvals = fastpars_all[ii][ll]['maxsqslopeori'][kk,:,sf]
    meanresp = fastpars_all[ii][ll]['meanresp'][kk,:,sf]
    inds2use = np.where(maxvals!=0)[0]
    inds2use= np.random.choice(inds2use,nUnitsPlot)
    plt.plot(maxvals[inds2use], meanresp[inds2use],'.',color=colors_main[color_ind,:],alpha=alpha)
  #  plt.yticks([])
    if ll==len(layers2do)-1:
      plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
    else:
      plt.xticks([])
    
    plt.title(layer_labels[ll])
    
  plt.suptitle('mean resp vs. orient where max slope occurs\n%s, %s, sf=%.2f'%(training_str,dataset, sf_vals[sf]))

#%% plot max slope vs the maximum slope regions of each unit's tuning function, for each layer
plt.rcParams['figure.figsize']=[18,10]
plt.close('all')
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
ii=0
kk=0
layers2do = np.arange(0,nLayers,1)
nUnitsPlot=10000
alpha=0.1

sf2plot = np.arange(0,nSF,1)
for sf in sf2plot:
  plt.figure();
 
  for ll in range(nLayers):
    plt.subplot(npx,npy,ll+1)
    
    maxvals = fastpars_all[ii][ll]['maxsqslopeori'][kk,:,sf]
    maxslope = fastpars_all[ii][ll]['maxsqslopeval'][kk,:,sf]
    inds2use = np.where(maxvals!=0)[0]
    inds2use= np.random.choice(inds2use,nUnitsPlot)
    plt.plot(maxvals[inds2use], maxslope[inds2use],'.',color=colors_main[color_ind,:],alpha=alpha)
  #  plt.yticks([])
    if ll==len(layers2do)-1:
      plt.xticks(np.arange(0,nOri+1,nOri/2), np.arange(0,181,90))  
    else:
      plt.xticks([])
    
    plt.title(layer_labels[ll])
    
  plt.suptitle('Max slope (squared) vs. orient where max slope occurs\n%s, %s, sf=%.2f'%(training_str,dataset, sf_vals[sf]))


#%% plot a couple example units which have high slope near the cardinal axes

plt.figure();
sf=4
ii=0
kk=0
nOri=180
orients = np.arange(0,nOri,1)
dat = np.transpose(tfs[kk,:,sf,0:nOri])
maxvals = fastpars_all[ii][ll]['maxsqslopeori'][kk,:,sf]
maxslope = fastpars_all[ii][ll]['maxsqslopeval'][kk,:,sf]

high_card_slope = np.where(np.logical_and(maxvals>85, maxvals<95))[0]

unit_colors = cm.Blues(np.linspace(0,1,5))
nUnitsPlot=10

inds2plot = np.random.choice(high_card_slope, nUnitsPlot)

#inds2plot = np.random.choice(np.arange(0,np.shape(activ_by_ori_all[ll])[1]), nUnitsPlot, replace='False')
npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)

for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)
  ydat=dat[:,inds2plot[uu]]
  plt.plot(orients,ydat,color=colors_main[color_ind,:])
#  plt.axvline(orients[maxvals[inds2plot[uu]]]+0.5)
  
  plt.plot(orients[maxvals[inds2plot[uu]]]+0.5, np.mean(ydat[maxvals[inds2plot[uu]]:maxvals[inds2plot[uu]]+2]), 'o')
  coord = coords_all[ii][ll][inds2plot[uu],:]
  plt.title('coords=%d,%d,%d'%(coord[0],coord[1],coord[2]))
meany = np.mean(dat,axis=1)
meany = meany - np.mean(meany)
#plt.plot(orients,meany,color='k')
plt.xticks(np.arange(0,181,45))
plt.suptitle('units with high slope near cardinals: %s, %s, eval on %s sf=%.2f'%(training_str, layer_labels[ll], dataset, info['sf_vals'][sf]))


#%% make plots of randomly selected well-fit units 
plt.close('all')
sf = 3
nUnits = np.shape(tfs)[1]
nUnitsPlot = 12
r2_here=r2_cutoff
nOri=180
rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][ll][:,sf,:],axis=1)))
cvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,0]))
avals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,2]))
#
rvals[np.isnan(rvals)] = -1000

# Now choose the units to plot, sorting by size
#units_good = np.where(np.logical_and(np.logical_and(np.logical_and(rvals>r2_here, ~np.isnan(cvals)), cvals>80), cvals<100))[0]
units_good = np.where(rvals>r2_here)[0]

units2plot = np.random.choice(units_good, nUnitsPlot, replace='False')

npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)

  real_y = np.squeeze(tfs[ii,units2plot[uu],sf,0:nOri])
  pars = fit_pars_all[ii][ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])

  plt.plot(ori_axis, real_y, color=colors_main[color_ind,:]) 
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.title('center=%d, size=%d, r2=%.2f'%(pars[0],pars[4], rvals[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('%s, %s: sf=%.2f\nExamples tuning curves, %s'%(training_str,dataset,sf_vals[sf],layer_labels[ll]))

#%% make plots of randomly selected well-fit units, with a specific size value
sf = 4
nUnits = np.shape(tfs)[1]
nUnitsPlot = 12
r2_here=r2_cutoff
nOri=180
rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][ll][:,sf,:],axis=1)))
cvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,0]))
avals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,2]))
kvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,1]))
#
rvals[np.isnan(rvals)] = -1000

# Now choose the units to plot, sorting by size
units_good = np.where(np.logical_and(rvals>r2_here, kvals>1000))[0]
#units_good = np.where(rvals>r2_here)[0]

units2plot = np.random.choice(units_good, nUnitsPlot, replace='False')

npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)

  real_y = np.squeeze(tfs[ii,units2plot[uu],sf,0:nOri])
  pars = fit_pars_all[ii][ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])

  plt.plot(ori_axis, real_y, color=colors_main[color_ind,:]) 
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.title('center=%d, size=%d, r2=%.2f'%(pars[0],pars[4], rvals[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('%s, %s: sf=%.2f\nExamples tuning curves, %s'%(training_str,dataset,info['sf_vals'][sf],layer_labels[ll]))


#%% make plots of randomly selected well-fit units, with a specific center value
sf = 4
nUnits = np.shape(tfs)[1]
nUnitsPlot = 12
r2_here=r2_cutoff
nOri=180
rvals = deepcopy(np.squeeze(np.mean(r2_all[ii][ll][:,sf,:],axis=1)))
cvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,0]))
avals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,2]))
kvals = deepcopy(np.squeeze(fit_pars_all[ii][ll][:,sf,1]))
#
rvals[np.isnan(rvals)] = -1000

# Now choose the units to plot, sorting by size
units_good = np.where(np.logical_and(np.logical_and(rvals>r2_here, cvals>85), cvals<95))[0]
#units_good = np.where(rvals>r2_here)[0]

units2plot = np.random.choice(units_good, nUnitsPlot, replace='False')

npx = np.ceil(np.sqrt(nUnitsPlot))
npy = np.ceil(nUnitsPlot/npx)

#% make the plot of these units of interest
plt.figure();
for uu in range(nUnitsPlot):
  plt.subplot(npx,npy,uu+1)

  real_y = np.squeeze(tfs[ii,units2plot[uu],sf,0:nOri])
  pars = fit_pars_all[ii][ll][units2plot[uu],sf,:]
  ypred = von_mises_deg(ori_axis, pars[0],pars[1],pars[2],pars[3])

  plt.plot(ori_axis, real_y, color=colors_main[color_ind,:]) 
  plt.plot(ori_axis, ypred,color=[0,0,0])
  plt.title('center=%d, size=%d, r2=%.2f'%(pars[0],pars[4], rvals[units2plot[uu]]))
  if uu!=nUnitsPlot-1:
    plt.xticks([])
plt.suptitle('%s, %s: sf=%.2f\nExamples tuning curves, %s'%(training_str,dataset,info['sf_vals'][sf],layer_labels[ll]))

