#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:07:15 2020

@author: mmhender

Some operations for doing manual convolutions on images (using only numpy).

"""

import numpy as np

def relu(activ):
  #relu activation function
  
  return np.maximum(activ,0)

def pad(activ,pad_with,n2pad):
  # pad the array 
  
  activ_full = activ
  activ_padded_full = np.zeros([np.shape(activ)[0]+n2pad*2, np.shape(activ)[1]+n2pad*2, np.shape(activ)[2]])
  for pp in range(np.shape(activ)[2]):
    activ = activ_full[:,:,pp]
    activ_padded = np.concatenate((pad_with*np.ones([np.shape(activ)[0],n2pad]),activ,pad_with*np.ones([np.shape(activ)[0],n2pad])),axis=1);
    activ_padded = np.concatenate((pad_with*np.ones([n2pad,np.shape(activ_padded)[1]]),activ_padded,pad_with*np.ones([n2pad,np.shape(activ_padded)[1]])),axis=0);
    activ_padded_full[:,:,pp] = activ_padded
    
  return activ_padded_full

def conv(activ,weights,stride=1):
  # convolve weights with activ. 
  # zero pad so input is same as output.
  
  filter_height = np.shape(weights)[0]
  filter_width = np.shape(weights)[1]
  in_channels = np.shape(weights)[2]
  out_channels = np.shape(weights)[3]
  
  half_filter_size = int(np.floor(filter_height/2))
  
  activ_padded = pad(activ,0,half_filter_size)
  
  # weights is now [FilterHeight*FilterWidth*InChannels x OutChannels]
  weights = np.reshape(weights, [filter_height*filter_width*in_channels,out_channels])
  
  in_height = np.shape(activ)[0]
  in_width = np.shape(activ)[1]
  
  out_height = int(np.ceil(in_height/stride))
  out_width = int(np.ceil(in_height/stride))
  
  activ_conv = np.zeros([out_height,out_width,out_channels])
  
  # looping over patches of the image
  xi = -1
  for ii in np.arange(half_filter_size,in_height+half_filter_size,stride):
    xi=xi+1
    yi=-1
    for jj in np.arange(half_filter_size,in_width+half_filter_size,stride):
      yi=yi+1;
      dat = activ_padded[ii-half_filter_size:ii+half_filter_size+1,
                         jj-half_filter_size:jj+half_filter_size+1,:]
      # dat is [FilterHeight*FilterWidth*nChannels x 1]
      dat = np.transpose(np.reshape(dat, [filter_height*filter_width*in_channels,1]))
      
      # apply all filters to this image patch
      activ_conv[xi,yi,:] = np.matmul(dat,weights)
      
    
  return activ_conv

def max_pool(activ,pool_size):
    # pool over pixels
    
    new_size = np.int64((np.ceil(np.shape(activ)[0]/pool_size),np.ceil(np.shape(activ)[1]/pool_size), np.shape(activ)[2]))
    
    activ_pool = np.zeros(new_size)
    for pp in range(np.shape(activ)[2]):
      xx=-1  
      for ii in np.arange(pool_size,np.shape(activ)[0],pool_size):
        xx=xx+1
        yy=-1
        for jj in np.arange(pool_size,np.shape(activ)[1],pool_size):
          yy=yy+1
          vals = activ[ii-pool_size:ii,jj-pool_size:jj,pp]
           
          activ_pool[xx,yy,pp] = np.max(vals)
           
    return activ_pool


