#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:07:15 2020

@author: mmhender
"""

import numpy as np
#import matplotlib.pyplot as plt
import conv_ops
import os
from PIL import Image
#import scipy
import multiprocessing


root ='/mnt/neurocube/local/serenceslab/maggie/biasCNN/'

#%% pass gratings through a network with random initializations, save a matrix of the result
# this is a (slow) manual way to simulate evaluating the VGG16 network

layer_names = ['conv1_conv1_1','conv1_conv1_2','pool1',
 'conv2_conv2_1','conv2_conv2_2','pool2',
 'conv3_conv3_1','conv3_conv3_2','conv3_conv3_3','pool3',
 'conv4_conv4_1','conv4_conv4_2','conv4_conv4_3','pool4',
 'conv5_conv5_1','conv5_conv5_2','conv5_conv5_3','pool5',
 'fc6']
 
nLayers = np.size(layer_names)

image_set = 'CosGratings'
#Orients = np.arange(0,180,1)
Orients = np.arange(0,2,1)
nOrients = np.size(Orients)
sf_vals = np.logspace(np.log10(0.02),np.log10(.4),6)*140/224
sf_vals = sf_vals[1:2]
Contrast = 0.80
phase = 0
ex = 1

num_in_channels=   [3, 64,64,64, 128,128,128,256,256,256,256,512,512,512,512,512,512,512,512]
num_out_channels = [64,64,64,128,128,128,256,256,256,256,512,512,512,512,512,512,512,512,4096]

#%% first, make random weights for each layer of the network

layer_weights = []

for ll in range(np.size(layer_names)):
  
  if 'conv' in layer_names[ll]:
    weights_rand = np.random.normal(size=[3,3,num_in_channels[ll],num_out_channels[ll]])
    
  elif 'pool' in layer_names[ll]:
    
    weights_rand = []
    
  elif 'fc6' in layer_names[ll]:
    
    weights_rand = np.random.normal(size=[7,7,num_in_channels[ll],num_out_channels[ll]])

  layer_weights.append(weights_rand)



#%% define function for individual images

def proc_image(oo,sf):

    activ_this_image = [];

    image_file = os.path.join(root,'images','gratings',image_set,
                              'SF_%0.2f_Contrast_%0.2f'%(sf_vals[sf],Contrast),
                              'Gaussian_phase0_ex1_%ddeg.png'%Orients[oo])
    print('loading from %s'%image_file)
    im = Image.open(image_file)
    im = np.reshape(np.array(im.getdata()),[224,224,3])
    
    activ = im
    
    # process each layer with this image
    for ll in range(np.size(layer_names)):
      
      if 'conv' in layer_names[ll]:
        weights_rand = layer_weights[ll]
        activ = conv_ops.conv(activ,weights_rand,1)
        
      elif 'pool' in layer_names[ll]:
        
        activ = conv_ops.pool(activ,2)
        
      elif 'fc6' in layer_names[ll]:
        weights_rand = layer_weights[ll]
        activ = conv_ops.conv(activ,weights_rand,1)    

      activ = conv_ops.relu(activ)
      
      # flatten matrix and add this to my big array
      activ_flat = np.ravel(activ)
      
      activ_this_image.append(activ_flat)
    
    activ = activ_proxy[oo]
#    activ_proxy[oo] = activ_this_image
      
    return 
  
  
  #%% now, pass each image through the random convnet
  
nCores = 8

for sf in range(np.size(sf_vals)):

  activ_patterns_all_images = [None]*np.size(Orients)

  with multiprocessing.Pool(processes=nCores) as pool:
    with multiprocessing.Manager() as manager:
#      ns = manager.Namespace()
#      ns.activ_all_ims = [None]*np.size(Orients)
      activ_proxy = manager.list()
      for ll in range(np.size(Orients)):
        activ_proxy.append([])
     
      # making an iterable of tuples with the arguments to pass to function
      arg_list = [(Orients[ii],sf) for ii in range(nOrients)]
      
      # processing each orientation individually
      pool.starmap(proc_image,arg_list)
     
 #%%   
    
  folder2save = os.path.join(root,'activations','vgg16_simul','random_normal_weights')
  if not os.path.isdir(folder2save):
    os.mkdir(folder2save)
  fn2save = os.path.join(folder2save,'SF_%0.2f_allunits.npy'%sf_vals[sf])
  np.save(fn2save,activ_patterns_all_images)
