#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:07:15 2020

@author: mmhender
"""

import numpy as np
import matplotlib.pyplot as plt

def pad(activ,pad_with,n2pad):
  # pad the array 
  
  activ_padded = np.concatenate((pad_with*np.ones([np.shape(activ)[0],n2pad]),activ,pad_with*np.ones([np.shape(activ)[0],n2pad])),axis=1);
  activ_padded = np.concatenate((pad_with*np.ones([n2pad,np.shape(activ_padded)[1]]),activ_padded,pad_with*np.ones([n2pad,np.shape(activ_padded)[1]])),axis=0);
  
  return activ_padded

def conv_mean(activ,kernel_size,stride):
  # pretend to do a convolution by taking the mean of elements in kernel area
  
  half_size = np.int8(np.floor(kernel_size/2))
  activ_conv = np.zeros([np.shape(activ)[0]-half_size*2,np.shape(activ)[1]-half_size*2])
  xx=-1  
  for ii in np.arange(half_size,np.shape(activ)[0]-half_size,stride):
    xx=xx+1
    yy=-1
    for jj in np.arange(half_size,np.shape(activ)[1]-half_size,stride):
      yy=yy+1
      vals = activ[ii-half_size:ii+half_size+1,jj-half_size:jj+half_size+1]
       
      activ_conv[xx,yy] = np.mean(vals)
       
  return activ_conv

def pool(activ,pool_size):
    # pool over pixels
    
    new_size = np.int64((np.ceil(np.shape(activ)[0]/pool_size),np.ceil(np.shape(activ)[1]/pool_size)))
    
    activ_pool = np.zeros(new_size)
    xx=-1  
    for ii in np.arange(pool_size,np.shape(activ)[0],pool_size):
      xx=xx+1
      yy=-1
      for jj in np.arange(pool_size,np.shape(activ)[1],pool_size):
        yy=yy+1
        vals = activ[ii-pool_size:ii,jj-pool_size:jj]
         
        activ_pool[xx,yy] = np.mean(vals)
         
    return activ_pool

    
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

activ = np.ones([224,224])
plt.figure();

plt.subplot(6,4,1)
plt.pcolormesh(activ)
plt.axis('square')
plt.title('original')

for ll in range(np.size(layer_names)):
  
  if 'conv' in layer_names[ll]:
    
    activ = conv_mean(pad(activ,0,1),3,1)
    
  elif 'pool' in layer_names[ll]:
    
    activ = pool(activ,2)
    
  elif 'fc6' in layer_names[ll]:
    
    activ = conv_mean(activ,7,1)

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
  plt.pcolormesh(activ)
  plt.axis('square')
  plt.title(layer_names[ll])
