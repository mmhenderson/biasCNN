#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate the discriminability curve (orientation discriminability versus orientation)
for activations at each layer of network, within each spatial frequency. Save the result. 

Created on Sun Sep  9 15:30:56 2018

@author: mmhender
"""

import os
import numpy as np
from copy import deepcopy
import load_activations
import classifiers_custom as classifiers    
import matplotlib.pyplot as plt
import scipy.spatial
import scipy

#%% set up paths and decide what datasets to look at here

root = '/usr/local/serenceslab/maggie/biasCNN/';
os.chdir(os.path.join(root, 'code', 'analysis_code'))

dataset_all = 'FiltImsAllSFCos_rand1'

#dataset_all = 'FiltNoiseCos_SF_0.14'
#dataset_all = 'PhaseVaryingCosGratings'
#dataset_all = 'CircGratings'
#dataset_all = 'SpatFreqGratings'
#dataset_all = 'SquareGratings'

model='vgg16'
param_str='params1'
training_str='scratch_imagenet_rot_0_cos'
#training_str='untrained'
#training_str='pretrained'
#training_str = 'scratch_imagenet_rot_0_stop_early'

#ckpt_strs=['0']
#ckpt_strs = ['350000','400000','450000']
ckpt_strs = ['400000']

nCheckpoints = np.size(ckpt_strs)

sf_vals = [0.01, 0.02, 0.04, 0.08, 0.14, 0.25]
nSF= np.size(sf_vals)
nLayers=22
#%%
def load_activ_loop_sf(training_str):
  
  allw =[];
  
  save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset_all)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  
  for sf in range(len(sf_vals)):
    
     # searching for folders corresponding to this data set
    dataset_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str))
  
    good = [ii for ii in range(np.size(dataset_dirs)) if dataset_all in dataset_dirs[ii] and 'SF_%s'%sf_vals[sf] in dataset_dirs[ii]]
    assert(np.size(good)==1)
    # looping over the datasets
    for dd in good:
       
      # loop over checkpoints
      for tr in range(nCheckpoints):
     
        ckpt_str = ckpt_strs[tr]
     
        # also searching for all evaluations at different timepts (nearby, all are around the optimal point)
        ckpt_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dataset_dirs[dd]))
        nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in ckpt_dirs]
        
        # compare the first two characters
        
        good2 = [jj for jj in range(np.size(ckpt_dirs)) if 'reduced' in ckpt_dirs[jj] and not 'sep_edges' in ckpt_dirs[jj] and ckpt_str[0:2] in nums[jj][0:2]]
        assert(np.size(good2)==1)
        ckpt_dir = ckpt_dirs[good2[0]]
       
        ckpt_num= ckpt_dir.split('_')[2][5:]
        w, varexpl, info = load_activations.load_activ(model, dataset_dirs[dd], training_str, param_str, ckpt_num)
             
        allw.append(w)
      
    return allw, info

def load_activ_noloop(training_str):
  
  allw =[];
  
  save_path = os.path.join(root,'code','discrim_func',model,training_str,param_str,dataset_all)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
 
   # searching for folders corresponding to this data set
  dataset_dir = os.path.join(root, 'activations', model, training_str, param_str, dataset_all)

  # loop over checkpoints
  for tr in range(nCheckpoints):
 
    ckpt_str = ckpt_strs[tr]
 
    # also searching for all evaluations at different timepts (nearby, all are around the optimal point)
    ckpt_dirs = os.listdir(os.path.join(root, 'activations', model, training_str, param_str,dataset_dir))
    nums=[dir[np.char.find(dir,'-')+1:np.char.find(dir,'-')+7] for dir in ckpt_dirs]
    
    # compare the first two characters
    
    good2 = [jj for jj in range(np.size(ckpt_dirs)) if 'reduced' in ckpt_dirs[jj] and not 'sep_edges' in ckpt_dirs[jj] and ckpt_str[0:2] in nums[jj][0:2]]
    assert(np.size(good2)==1)
    ckpt_dir = ckpt_dirs[good2[0]]
   
    ckpt_num= ckpt_dir.split('_')[2][5:]
    w, varexpl, info = load_activations.load_activ(model, dataset_all, training_str, param_str, ckpt_num)
         
    allw.append(w)
      
     
    return allw, info
#%%
if 'PhaseVarying' in dataset_all:  
  allw, info = load_activ_loop_sf(training_str)
else:
  allw, info = load_activ_noloop(training_str)
  
nSF = info['nSF']
orilist = info['orilist']
sflist = info['sflist']
layer_labels = info['layer_labels']

#%% plot dissimilarity matrices 
plt.close('all')  
fig = plt.figure()
    
ll = 5

for sf in [0]:
#for sf in range(len(sf_vals)):
    
  plt.subplot(2,3,sf+1)
  
  if np.shape(allw)[0]>1:
    w = allw[sf][ll][0]
    myorilist = orilist
  else:
    w =allw[0][ll][0]
    inds = np.where(sflist==sf)[0]
    w = w[inds,:]
    myorilist = orilist[inds]
  disc = np.zeros([180,180])
  
  un = np.unique(myorilist)
  
  myinds_bool = np.ones(np.shape(myorilist))
  for ii in np.arange(0,np.size(un)):
      
      # find all gratings with this label
      inds1 = np.where(np.logical_and(myorilist==un[ii], myinds_bool))[0]    
#      assert(np.size(inds1)==48)
      for jj in np.arange(ii+1, np.size(un)):
          
          # now all gratings with other label
          inds2 = np.where(np.logical_and(myorilist==un[jj], myinds_bool))[0]    
#          assert(np.size(inds2)==48)
          dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])
            
          disc[ii,jj]=  dist
          disc[jj,ii] = dist
          
  plt.pcolormesh(disc)
  plt.colorbar()
      
  plt.axis('square')
     
  plt.title('%.2f cpp'%sf_vals[sf])
     
#  plt.xlabel('orientation 1')
#  plt.ylabel('orientation 2')
  
  plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
  plt.yticks(np.arange(0,180,45),np.arange(0,180,45))

plt.suptitle('%s - %s\n%s' % (training_str,dataset_all,layer_labels[ll]))
 
fig.set_size_inches(18,8)


#%% plot dissimilarity matrices 
plt.close('all')  
fig = plt.figure()
    
ll=12
sf=0

 
if np.shape(allw)[0]>1:
  w = allw[sf][ll][0]
  myorilist = orilist
else:
  w =allw[0][ll][0]
  inds = np.where(sflist==sf)[0]
  w = w[inds,:]
  myorilist = orilist[inds]
  
disc = np.zeros([180,180])
disc1 = np.zeros([180,180])
disc1[:]=np.nan
disc5 = np.zeros([180,180])
disc5[:]=np.nan
un = np.unique(myorilist)
disc_shifted = np.zeros([180,180])
disc_shifted1 = np.zeros([180,180])
disc_shifted1[:] = np.nan
disc_shifted5 = np.zeros([180,180])
disc_shifted5[:] = np.nan

shift_by=45
shifted_ticks  = np.mod(np.arange(0,180,45)-shift_by,180)

myinds_bool = np.ones(np.shape(myorilist))
for ii in np.arange(0,np.size(un)):
    
  
  # find all gratings with this label
  inds1 = np.where(np.logical_and(myorilist==un[ii], myinds_bool))[0]    
#      assert(np.size(inds1)==48)
  for jj in np.arange(ii+1, np.size(un)):
    
    ii2 = np.mod(ii+shift_by, 180)
    jj2 = np.mod(jj+shift_by, 180)
    ii2,jj2 = np.sort([ii2,jj2])
    
    # now all gratings with other label
    inds2 = np.where(np.logical_and(myorilist==un[jj], myinds_bool))[0]    
#          assert(np.size(inds2)==48)
    dist = classifiers.get_norm_euc_dist(w[inds1,:],w[inds2,:])
      
    disc[ii,jj]=  dist
    disc[jj,ii] = dist
    if np.abs(jj-ii)==5:
      disc5[ii,jj] = dist
    if np.abs(jj-ii)==1:
      disc1[ii,jj] = dist;
      
    disc_shifted[ii2,jj2]=  dist
    disc_shifted[jj2,ii2] = dist
    if np.abs(jj2-ii2)==5:
      disc_shifted5[ii2,jj2] = dist
    if np.abs(jj2-ii2)==1:
      disc_shifted1[ii2,jj2] = dist;
        
          
plt.subplot(2,3,1)  
plt.pcolormesh(disc)
plt.colorbar()
plt.axis('square')
plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
plt.yticks(np.arange(0,180,45),np.arange(0,180,45))
plt.title('full dissimilarity matrix')

plt.subplot(2,3,2)  
plt.pcolormesh(disc1)
plt.colorbar()
plt.axis('square')
plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
plt.yticks(np.arange(0,180,45),np.arange(0,180,45))
plt.title('just the 1-off diagonal (1-deg steps)')

plt.subplot(2,3,3)  
plt.pcolormesh(disc5)
plt.colorbar()
plt.axis('square') 
plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
plt.yticks(np.arange(0,180,45),np.arange(0,180,45))
plt.title('just the 5-off diagonal (5 deg steps)')


plt.subplot(2,3,4)  
plt.pcolormesh(disc_shifted)
plt.colorbar()
plt.axis('square')
plt.xticks(np.arange(0,180,45),shifted_ticks)
plt.yticks(np.arange(0,180,45),shifted_ticks)
plt.title('full dissimilarity matrix (shifted by %d deg)'%shift_by)

plt.subplot(2,3,5)  
plt.pcolormesh(disc_shifted1)
plt.colorbar()
plt.axis('square')
plt.xticks(np.arange(0,180,45),shifted_ticks)
plt.yticks(np.arange(0,180,45),shifted_ticks)
plt.title('just the 1-off diagonal (1-deg steps)')

plt.subplot(2,3,6)  
plt.pcolormesh(disc_shifted5)
plt.colorbar()
plt.axis('square') 
plt.xticks(np.arange(0,180,45),shifted_ticks)
plt.yticks(np.arange(0,180,45),shifted_ticks)
plt.title('just the 5-off diagonal (5 deg steps)')

plt.suptitle('%s - %s\n%s' % (training_str,dataset_all,layer_labels[ll]))
 
fig.set_size_inches(18,8)


#%% plot non-normalized euc. distance
plt.close('all')  
fig = plt.figure()
    
ll = 18


for sf in range(len(sf_vals)):
    
  plt.subplot(2,3,sf+1)
  
  if np.shape(allw)[0]>1:
    w = allw[sf][ll][0]
    myorilist = orilist
  else:
    w =allw[0][ll][0]
    inds = np.where(sflist==sf)[0]
    w = w[inds,:]
    myorilist = orilist[inds]
  disc = np.zeros([180,180])
      
  un = np.unique(myorilist)
  
  myinds_bool = np.ones(np.shape(myorilist))
  for ii in np.arange(0,np.size(un)):
      
      # find all gratings with this label
      inds1 = np.where(np.logical_and(myorilist==un[ii], myinds_bool))[0]    
#      assert(np.size(inds1)==48)
      for jj in np.arange(ii+1, np.size(un)):
          
          # now all gratings with other label
          inds2 = np.where(np.logical_and(myorilist==un[jj], myinds_bool))[0]    
#          assert(np.size(inds2)==48)
          dist = classifiers.get_euc_dist(w[inds1,:],w[inds2,:])
            
          disc[ii,jj]=  dist
          disc[jj,ii] = dist
          
  plt.pcolormesh(disc)
  plt.colorbar()
      
  plt.axis('square')
     
  plt.title('%.2f cpp'%sf_vals[sf])
     
#  plt.xlabel('orientation 1')
#  plt.ylabel('orientation 2')
  
  plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
  plt.yticks(np.arange(0,180,45),np.arange(0,180,45))

plt.suptitle('%s - %s\n%s' % (training_str,dataset_all,layer_labels[ll]))
 
fig.set_size_inches(18,8)

#%% plot response variance (first 5 units
plt.close('all')  
fig = plt.figure()
    
ll = 18


for sf in range(len(sf_vals)):
    
  plt.subplot(2,3,sf+1)
  
  if np.shape(allw)[0]>1:
    w = allw[sf][ll][0]
    myorilist = orilist
  else:
    w =allw[0][ll][0]
    inds = np.where(sflist==sf)[0]
    w = w[inds,:]
    myorilist = orilist[inds]
  disc = np.zeros([180,180])
      
  un = np.unique(myorilist)
  
  myinds_bool = np.ones(np.shape(myorilist))
  for ii in np.arange(0,np.size(un)):
      
      # find all gratings with this label
      inds1 = np.where(np.logical_and(myorilist==un[ii], myinds_bool))[0]    
#      assert(np.size(inds1)==48)
      for jj in np.arange(ii+1, np.size(un)):
          
          # now all gratings with other label
          inds2 = np.where(np.logical_and(myorilist==un[jj], myinds_bool))[0]    
#          assert(np.size(inds2)==48)
          var = np.var(np.concatenate((w[inds1,1:5],w[inds2,1:5]),axis=0), axis=0)
            
          disc[ii,jj]=  np.mean(var)
          disc[jj,ii] = np.mean(var)
          
  plt.pcolormesh(disc)
  plt.colorbar()
      
  plt.axis('square')
     
  plt.title('%.2f cpp'%sf_vals[sf])
     
#  plt.xlabel('orientation 1')
#  plt.ylabel('orientation 2')
  
  plt.xticks(np.arange(0,180,45),np.arange(0,180,45))
  plt.yticks(np.arange(0,180,45),np.arange(0,180,45))

plt.suptitle('Response Variance\n%s - %s\n%s' % (training_str,dataset_all,layer_labels[ll]))
 
fig.set_size_inches(18,8)
