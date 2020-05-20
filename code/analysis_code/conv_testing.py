#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:07:15 2020

@author: mmhender
"""

import numpy as np
import matplotlib.pyplot as plt
import conv_ops
import os
from PIL import Image
import scipy

root ='/mnt/neurocube/local/serenceslab/maggie/biasCNN/'

#%% pass a couple of gratings through a random netork

layer_names = ['conv1_conv1_1','conv1_conv1_2','pool1',
 'conv2_conv2_1','conv2_conv2_2','pool2',
 'conv3_conv3_1','conv3_conv3_2','conv3_conv3_3','pool3',
 'conv4_conv4_1','conv4_conv4_2','conv4_conv4_3','pool4',
 'conv5_conv5_1','conv5_conv5_2','conv5_conv5_3','pool5',
 'fc6']
 
image_set = 'CosGratings'
Orients = [0,1,44,45,46,89,90,91,134,135,136,179]
nOrients = np.size(Orients)
SF = 0.08
Contrast = 0.80
layer2plot = 12

num_in_channels=   [3, 64,64,64, 128,128,128,256,256,256,256,512,512,512,512,512,512,512,512]
num_out_channels = [64,64,64,128,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096]

layer_weights = []

plt.close('all')
plt.figure();

for ll in range(np.size(layer_names)):
  
  if 'conv' in layer_names[ll]:
    weights_rand = np.random.normal(size=[3,3,num_in_channels[ll],num_out_channels[ll]])
    
  elif 'pool' in layer_names[ll]:
    
    weights_rand = []
    
  elif 'fc6' in layer_names[ll]:
    
    weights_rand = np.random.normal(size=[7,7,num_in_channels[ll],num_out_channels[ll]])

  layer_weights.append(weights_rand)


for oo in range(np.size(Orients)):
  
  image_file = os.path.join(root,'images','gratings',image_set,
                            'SF_%0.2f_Contrast_%0.2f'%(SF,Contrast),
                            'Gaussian_phase0_ex1_%ddeg.png'%Orients[oo])
  print('loading from %s'%image_file)
  im = Image.open(image_file)
  im = np.reshape(np.array(im.getdata()),[224,224,3])
  
  activ = im
  
  for ll in range(np.size(layer_names)):
    
    if 'conv' in layer_names[ll]:
      weights_rand = layer_weights[ll]
      activ = conv_ops.conv(activ,weights_rand,1)
      
    elif 'pool' in layer_names[ll]:
      
      activ = conv_ops.pool(activ,2)
      
    elif 'fc6' in layer_names[ll]:
      weights_rand = layer_weights[ll]
      activ = conv_ops.conv(activ,weights_rand,1)    
  
      
#    print('output of %s is size [%d by %d by %d]'%(layer_names[ll],np.shape(activ)[0],np.shape(activ)[1],np.shape(activ)[2]))
  
    if ll==layer2plot:
      
      plt.subplot(int(np.ceil(np.sqrt(nOrients))),int(np.ceil(np.sqrt(nOrients))),oo+1)
      plt.pcolormesh(activ[:,:,0])
      plt.axis('square')
      plt.title('%d deg'%Orients[oo])
      
      if oo==0:
        activ_patterns = np.zeros([np.size(Orients), np.size(activ)])
      
      activ_patterns[oo,:] = np.ravel(activ)
     
      
plt.suptitle('%s, first map'%layer_names[layer2plot])

# plot a dissimilarity matrix, all units of the same layer
plt.figure();
plt.title('distances between pairs of gratings\n%s'%layer_names[layer2plot])
distmat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(activ_patterns));
plt.pcolormesh(distmat)
plt.axis('square')
plt.xticks(np.arange(0,np.size(Orients))+0.5, Orients)
plt.yticks(np.arange(0,np.size(Orients))+0.5, Orients)
plt.colorbar()
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
