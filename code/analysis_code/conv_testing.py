#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:07:15 2020

@author: mmhender
"""

import numpy as np
import matplotlib.pyplot as plt
import conv_ops
import get_manual_activ
import os
from PIL import Image
import scipy
import load_activations

root ='/mnt/neurocube/local/serenceslab/maggie/biasCNN/'

#%% pass a couple of gratings through a random netork

layer_names = ['conv1_conv1_1','conv1_conv1_2','pool1',
 'conv2_conv2_1','conv2_conv2_2','pool2',
 'conv3_conv3_1','conv3_conv3_2','conv3_conv3_3','pool3',
 'conv4_conv4_1','conv4_conv4_2','conv4_conv4_3','pool4',
 'conv5_conv5_1','conv5_conv5_2','conv5_conv5_3','pool5',
 'fc6']
 
#image_set = 'FiltIms14AllSFCos_rand1'
image_set = 'FiltIms11Cos_SF_0.25_rand1'
info = load_activations.get_info('vgg16', image_set)
ori2load = [0,1,44,45,46,89,90,91,134,135,136,179]
ori2load = np.arange(0,180,15)
nOri_load = np.size(ori2load)
nEx=1
ex2load = np.arange(1,nEx+1,1)

num_in_channels=   [3, 64,64,64, 128,128,128,256,256,256,256,512,512,512,512,512,512,512,512]
num_out_channels = [64,64,64,128,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096]

nLayers=np.size(layer_names)
#%% get constant weights
layer_weights = []
for ll in range(nLayers):    
    if 'conv' in layer_names[ll]:
      weights = np.ones([3,3,num_in_channels[ll],num_out_channels[ll]])      
    elif 'pool' in layer_names[ll]:      
      weights = []
    elif 'fc6' in layer_names[ll]:
      weights = np.ones([7,7,num_in_channels[ll],num_out_channels[ll]])
  
    layer_weights.append(weights)
 
#%% get random weights   
layer_weights = []
rand_seed=345672
np.random.seed(rand_seed) 
for ll in range(nLayers):    
  if 'conv' in layer_names[ll]:
    weights_rand = np.random.normal(size=[3,3,num_in_channels[ll],num_out_channels[ll]])      
  elif 'pool' in layer_names[ll]:      
    weights_rand = []
  elif 'fc6' in layer_names[ll]:
    weights_rand = np.random.normal(size=[7,7,num_in_channels[ll],num_out_channels[ll]])
  
  layer_weights.append(weights_rand)

#%% process images
    
_R_MEAN = 124
_G_MEAN = 117
_B_MEAN = 104
im2subtract = np.tile([124,117,104],[224,224,1])

activ_by_ori_all = []
activ_spat_all = []
for oo in range(nOri_load):
  
  for ee in range(nEx):
    
    image_file = os.path.join(root,'images','gratings',image_set,
                              'AllIms', 'FiltImage_ex%d_%ddeg.png'%(ex2load[ee], ori2load[oo]))
    print('loading from %s'%image_file)
    im = Image.open(image_file)
    im = np.reshape(np.array(im.getdata()),[224,224,3])
    im = im - im2subtract
  #  im = im - np.mean(np.ravel(im))
    activ = im
    
    for ll in range(nLayers):
  
      if 'conv' in layer_names[ll]:
        weights = layer_weights[ll]
        activ = conv_ops.conv(activ,weights,1)
        activ = conv_ops.relu(activ)
    
      elif 'pool' in layer_names[ll]:
        
        activ, sd = conv_ops.max_pool_sd(activ,2)
#        activ = conv_ops.max_pool(activ,2)
#        activ = conv_ops.min_pool(activ,2)
#        activ = conv_ops.mean_pool(activ,2)
        
      elif 'fc6' in layer_names[ll]:
        weights = layer_weights[ll]
        activ = conv_ops.conv(activ,weights,1)    
        activ = conv_ops.relu(activ)
    
      
      if oo==0 & ee==0:
        activ_by_ori = np.zeros([nOri_load, nEx, np.prod(np.shape(activ))])
        activ_by_ori_all.append(activ_by_ori)
        activ_spat = np.zeros([nOri_load, nEx, np.shape(activ)[0],np.shape(activ)[1], np.shape(activ)[2]])
        activ_spat_all.append(activ_spat)
        if ll==2:
          sd_by_ori_pool1 = np.zeros([nOri_load, nEx, np.prod(np.shape(activ))])
          
      if ll==2:
        sd_by_ori_pool1[oo,ee,:] = np.ravel(sd)
        
      activ_by_ori_all[ll][oo,ee,:] = np.ravel(activ)
        
      activ_spat_all[ll][oo,ee,:,:,:] = activ;
#%% plot activ versus ori for each layer
plt.close('all')
plt.figure();
npx = np.ceil(np.sqrt(nLayers))
npy = np.ceil(nLayers/npx)
lims = []
for ll in range(nLayers):
   plt.subplot(npx,npy,ll+1)
   resp = activ_by_ori_all[ll]
   # average over exemplar images and all units
   mean_resp = np.mean(np.mean(resp,axis=2),axis=1)
   mi=np.min(np.ravel(resp))
   ma=np.max(np.ravel(resp))
   lims.append([mi, ma])
#   err_resp = np.std(resp,axis=1)
#   plt.errorbar(ori2load, mean_resp, err_resp)
   plt.plot(ori2load, mean_resp)
   plt.xlabel('orientation')
   plt.xticks(ori2load,ori2load)
   plt.ylabel('response')
   plt.title('%s'%layer_names[ll])
 
#%% plot SD versus ori for pool1 layer
ll=2
plt.close('all')
plt.figure();

mean_vals = np.mean(np.mean(sd_by_ori_pool1,axis=2),axis=1)
err_vals = np.mean(np.std(sd_by_ori_pool1,axis=2),axis=1)
#plt.plot(ori2load, mean_vals)
plt.errorbar(ori2load, mean_vals,err_vals)
plt.xlabel('orientation')
plt.xticks(ori2load,ori2load)
plt.ylabel('STD of values in pooling region')
plt.title('%s'%layer_names[ll])

#%% plot activ versus SD of inputs for pool1 layer
from matplotlib import cm
ll=2
plt.close('all')
plt.figure();
cols = cm.Reds(np.linspace(0,1,nOri_load))
ee=0
for oo in range(nOri_load):
  
  sd_vals = sd_by_ori_pool1[oo,ee,:]
  act_vals = activ_by_ori_all[ll][oo,ee,:]
  
  plt.plot(np.ravel(sd_vals), np.ravel(act_vals),'.',color=cols[oo,:])
    
plt.legend(['%d deg'%oo for oo in ori2load])
plt.xlabel('STD of values in pooling region')
plt.ylabel('Activ of unit')
plt.title('%s'%layer_names[ll])

#%% plot spatial activation maps for some example layers
ll=2
plt.close('all')
plt.figure() 
npx = np.ceil(np.sqrt(nOri_load))
npy = np.ceil(nOri_load/npx)
 
for oo in range(nOri_load):
  plt.subplot(npx,npy,oo+1)
  # averaging over exemplar images
  act=np.mean(activ_spat_all[ll][oo,:,:,:,:],axis=0)
  # averaging over channels
  plt.pcolormesh(np.mean(act,axis=2))
  
  plt.axis('square')
  plt.xticks([])
  plt.yticks([])
#  plt.title('%d deg: %.2f'%(ori2load[oo],np.mean(np.ravel(act))))
  plt.title('%d deg'%(ori2load[oo]))
  plt.clim(lims[ll])
  plt.colorbar()
plt.suptitle('%s, avg all maps'%layer_names[ll])

#%%
## plot a dissimilarity matrix, all units of the same layer
#plt.figure();
#plt.title('distances between pairs of gratings\n%s'%layer_names[layer2plot])
#distmat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(activ_patterns));
#plt.pcolormesh(distmat)
#plt.axis('square')
#plt.xticks(np.arange(0,np.size(Orients))+0.5, Orients)
#plt.yticks(np.arange(0,np.size(Orients))+0.5, Orients)
#plt.colorbar()
#%% trying to see what it looks like to pass an image through a random netork

layer_names = ['conv1_conv1_1','conv1_conv1_2','pool1',
 'conv2_conv2_1','conv2_conv2_2','pool2',
 'conv3_conv3_1','conv3_conv3_2','conv3_conv3_3','pool3',
 'conv4_conv4_1','conv4_conv4_2','conv4_conv4_3','pool4',
 'conv5_conv5_1','conv5_conv5_2','conv5_conv5_3','pool5',
 'fc6']
 
num_out_channels = [64,64,64,128,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096]

plt.close('all')

activ = np.ones([224,224,3])
plt.figure();

plt.subplot(6,4,1)
plt.pcolormesh(activ[:,:,0])
plt.axis('square')
plt.title('original image')

for ll in range(np.size(layer_names)):
  
  if 'conv' in layer_names[ll]:
    weights_rand = np.random.normal(size=[3,3,np.shape(activ)[2],num_out_channels[ll]])
    activ = conv_ops.conv(activ,weights_rand,1)
    
  elif 'pool' in layer_names[ll]:
    
    activ = conv_ops.pool(activ,2)
    
  elif 'fc6' in layer_names[ll]:
    weights_rand = np.random.normal(size=[7,7,np.shape(activ)[2],num_out_channels[ll]])
    activ = conv_ops.conv(activ,weights_rand,1)    

    
  print('output of %s is size [%d by %d by %d]'%(layer_names[ll],np.shape(activ)[0],np.shape(activ)[1],np.shape(activ)[2]))


  plt.subplot(6,4,ll+1)
  plt.pcolormesh(activ[:,:,0])
  plt.axis('square')
  plt.title('%s'%layer_names[ll])

#%% trying to see how many pixels would get contaminated by edge artifacts at each layer.
    
layer_names = ['conv1_conv1_1','conv1_conv1_2','pool1',
 'conv2_conv2_1','conv2_conv2_2','pool2',
 'conv3_conv3_1','conv3_conv3_2','conv3_conv3_3','pool3',
 'conv4_conv4_1','conv4_conv4_2','conv4_conv4_3','pool4',
 'conv5_conv5_1','conv5_conv5_2','conv5_conv5_3','pool5',
 'fc6',
 'fc7',
 'fc8']
 
plt.close('all')

activ = np.ones([224,224,1])
plt.figure();

plt.subplot(6,4,1)
plt.pcolormesh(activ[:,:,0])
plt.axis('square')
plt.title('original')

for ll in range(np.size(layer_names)):
  
  if 'conv' in layer_names[ll]:
    
    activ = conv_ops.conv_mean(conv_ops.pad(activ,0,1),3,1)
    
  elif 'pool' in layer_names[ll]:
    
    activ = conv_ops.pool(activ,2)
    
  elif 'fc6' in layer_names[ll]:
    
    activ = conv_ops.conv_mean(activ,7,1)

  # take cross sections to check for nonzero edge pixels 
  center = np.int64(np.shape(activ)[1]/2)   
  vert_center = activ[:,center]
  horiz_center = activ[center,:]
  
  edges_vert = np.where(vert_center!=1)[0]
  edges_horiz = np.where(horiz_center!=1)[0]
  
  nedges_top = np.size(np.where(edges_vert<center)[0])
  nedges_bottom = np.size(np.where(edges_vert>center)[0])
  nedges_left = np.size(np.where(edges_horiz<center)[0])
  nedges_right = np.size(np.where(edges_horiz>center)[0])
  
  print('\nsize of %s:'%layer_names[ll])
  print(np.shape(activ))
  
  print('max edge contamination in %s:'%layer_names[ll])
  print(np.max([nedges_top,nedges_bottom,nedges_left,nedges_right]))

  plt.subplot(6,4,ll+2)
  plt.pcolormesh(activ[:,:,0])
  plt.axis('square')
  plt.title(layer_names[ll])
