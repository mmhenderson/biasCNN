#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate "tuning" based only on pixel values - control.
"""

import os
import numpy as np
import shutil
import sys
import load_activations
from PIL import Image
from copy import deepcopy

_R_MEAN = 124
_G_MEAN = 117
_B_MEAN = 104

def get_orient_tuning_fn(load_path,save_path,dataset_name,do_relu=0):
  
  #%% get info for the netwok
  info = load_activations.get_info('pixel', dataset_name)
  
  # extract some things from info  
  sflist = info['sflist']  
  nSF = np.size(np.unique(sflist))
  sf_vals = info['sf_vals'][0:nSF]

  # get the correct orientation vals to use
  if info['nPhase']==2:
    # make a full 360 deg space
    orilist = deepcopy(info['orilist'])
    orilist_orig=deepcopy(info['orilist'])
    phaselist=deepcopy(info['phaselist'])
    orilist[phaselist==1] = orilist[phaselist==1]+180
  else: 
    # use the regular 180 deg space
    orilist = info['orilist'] 
    orilist_orig=orilist
    
  exlist=info['exlist']
  sflist = info['sflist']

  ori_axis = np.unique(orilist);
  nOri = np.size(ori_axis)
  nIms = np.size(orilist)

  H=224
  W=224
  nPix=H*W
  
  print('dataset %s has %d spatial freqs, orients from 0-%d deg\n'%(dataset_name,nSF,nOri))
  
  
  # if the save directory already exists, clear it before beginning.
  if os.path.isdir(save_path):
      print('deleting contents of %s' % save_path)
      shutil.rmtree(save_path, ignore_errors = True)
      os.mkdir(save_path)    
  else:
      os.mkdir(save_path)   
    
  #%% first make the "activation matrix" which is just pixel values
 
  for ii in range(nIms):
    
    if 'FiltIms' in dataset_name:
      im_file = os.path.join(load_path,'AllIms','FiltImage_ex%d_%ddeg.png'%(exlist[ii]+1,orilist_orig[ii]))
    else:
      im_file = os.path.join(load_path,
                           'SF_%0.2f_Contrast_0.80'%(sf_vals[sflist[ii]]),
                           'Gaussian_phase%d_ex%d_%ddeg.png'%(phaselist[ii][0],exlist[ii]+1,orilist_orig[ii]))

    image = Image.open(im_file)
    
    if do_relu==0:
      if ii==0:
         w = np.zeros([nIms, nPix])
 
      #convert to grayscale    
      image = image.convert('L')
      
      # unwrap to a vector 
      w[ii,:] = image.getdata()
      
    else:
      if ii==0:
         w = np.zeros([nIms, nPix*3])
 
      # get image data
      im = np.float64(np.reshape(image.getdata(),[H,W,3]))
      
      # first - subtracting the mean so everything is centered at zero
      im = im - np.float64(np.swapaxes(np.tile(np.expand_dims(np.expand_dims(np.asarray([_R_MEAN,_G_MEAN,_B_MEAN]),1),2),[1,H,W]),2,0));
      
      # next apply a relu operation
      im = np.maximum(im,0)
      
      w[ii,:] = np.reshape(im,np.prod(np.shape(im)))
      
#    wlabs = np.tile(np.expand_dims(np.arange(0,W),axis=1),[H,1])
#    hlabs = np.expand_dims(np.repeat(np.arange(0,H),W),axis=1)
#    # the coords matrix goes [nUnits x 3] where columns are [H,W,C]
#    coords = np.concatenate((hlabs,wlabs),axis=1)      
#    assert np.array_equal(coords, np.unique(coords, axis=0))
    
  #%% get the tuning curves for every unit in this layer
  nUnits = np.shape(w)[1]
  tuning_curves = np.zeros([nUnits, nSF, nOri])
  for sf in range(nSF):
    for ii in range(nOri):
      
      # find all stims that have this orientation and spatial frequency
      inds2use = np.where(np.logical_and(sflist==sf, orilist==ori_axis[ii]))[0]
      
      # take mean over all stims with this feature value
      # [nUnits long]
      meandat = np.mean(w[inds2use,:],axis=0)
      
      # put this into the big tuning curve matrix
      #[nUnits x nSF x nOri]
      tuning_curves[:,sf,ii] = meandat
        
  #%% Save the result as a single file
  
  fn2save = os.path.join(save_path, 'AllUnitsOrientTuning_%s.npy'%info['layer_labels'][0])
  
  np.save(fn2save, tuning_curves)
  print('saving to %s\n' % (fn2save))
      
        
if __name__ == '__main__':
  
  load_path = sys.argv[1]
  save_path = sys.argv[2] # The path to save the reduced activation files to.
  dataset_name = sys.argv[3] # the name of the dataset
  model_name = sys.argv[4]
  
  print('load_path is %s'%load_path)
  print('save_path is %s'%save_path)
  print('dataset name is %s'%dataset_name)
  print('model name is %s'%model_name)
  
  if 'relu' in model_name:
    do_relu=1
  else:
    do_relu=0
    
  get_orient_tuning_fn(load_path,save_path,dataset_name,do_relu=do_relu)
