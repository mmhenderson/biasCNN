#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:07:15 2020

@author: mmhender
"""

import numpy as np
import matplotlib.pyplot as plt
#import conv_ops
import os
from PIL import Image
import scipy
#from scipy import spatial
from sklearn import decomposition
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
Orients = np.arange(0,180,1)
nOrients = np.size(Orients)
sf_vals = np.logspace(np.log10(0.02),np.log10(.4),6)*140/224
sf_vals = sf_vals[0:1]
Contrast = 0.80
phase = 0
ex = 1

folder2save = os.path.join(root,'activations','vgg16_simul','random_normal_weights')
fn2save = os.path.join(folder2save,'SF_%0.2f_allunits.npy'%sf_vals[0])
activ_patterns_all = np.load(fn2save)

wall = []
for ll in nLayers:
  
  
  activ =np.zeros([np.size(Orients), np.shape(activ_patterns_all[oo])[1]])
  for oo in np.size(Orients):
    
    
  
  wall.append(activ)

#%% reduce w pca

#pctVar = 95
#n_components_keep = 180;
#min_components_keep = 10 
#
#bigw =[]
#
#for ll in range(nLayers):
#  
#  allw = activ_patterns_all[ll]
#
#  pca = decomposition.PCA(n_components = np.min((n_components_keep, np.shape(allw)[1])))
#  print('\n STARTING PCA WITH %d COMPONENTS MAX\n'%(n_components_keep))
#  print('size of allw before reducing is %d by %d'%(np.shape(allw)[0],np.shape(allw)[1]))
#  weights_reduced = pca.fit_transform(allw)   
#
#  var_expl = pca.explained_variance_ratio_
#  
#  n_comp_needed = np.where(np.cumsum(var_expl)>pctVar/100)
#  if np.size(n_comp_needed)==0:
#    n_comp_needed = n_components_keep
#    print('need >%d components to capture %d percent of variance' % (n_comp_needed, pctVar))
#  else:
#    n_comp_needed = n_comp_needed[0][0]
#    print('need %d components to capture %d percent of variance' % (n_comp_needed, pctVar))
#    
#  if n_comp_needed<min_components_keep:
#    n_comp_needed = min_components_keep
#    
#  bigw.append(allw[:,0:n_comp_needed])  

#%% plot a dissimilarity matrix, all units of the same layer

bigw = activ_patterns_all

plt.close('all')
plt.figure();

layers2plot = range(nLayers)

for ll in layers2plot:
  
  plt.subplot(6,4,ll+1)
  activ_patterns = bigw[ll]  
  plt.title('%s'%layer_names[ll])
  distmat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(activ_patterns));
  plt.pcolormesh(distmat)
  plt.axis('square')
  
  spacing =45
  
  plt.xticks(np.arange(0,np.size(Orients),spacing)+0.5, Orients[np.arange(0,nOrients,spacing)])
  plt.yticks(np.arange(0,np.size(Orients),spacing)+0.5, Orients[np.arange(0,nOrients,spacing)])
  plt.colorbar()
  
plt.suptitle('Dissimilarity matrix')