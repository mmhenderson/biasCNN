#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:50:10 2021

@author: mmhender
"""
import numpy as np
from matplotlib import cm

def get_cmaps_biasCNN():
  
  """ 
  Create color maps
  """
  
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
  colors_sf = np.moveaxis(np.expand_dims(cm.GnBu(np.linspace(0,1,8)),axis=2),[0,1,2],[0,2,1])
  colors_sf = colors_sf[np.arange(2,8,1),:,:]
  
  return colors_main, colors_sf

def get_fi_bin_pars(nOri=180, bin_size=20, plot=False):
    """
    Define some fixed parameters for calculating Fisher information bias
    """ 
    
    ori_axis = np.arange(0, nOri,1)
    # define the bins of interest
    b = np.arange(22.5,nOri,90)  # baseline
    t = np.arange(67.5,nOri,90)  # these are the orientations that should be dominant in the 22 deg rot set (note the CW/CCW labels are flipped so its 67.5)
    c = np.arange(0,nOri,90) # cardinals
    o = np.arange(45,nOri,90)  # obliques

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

    if plot==True:
        #%% visualize the bins 
        plt.figure();
        plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),baseline_inds))
        plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),card_inds))
        plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),obl_inds))
        plt.plot(ori_axis,np.isin(np.arange(0,np.size(ori_axis),1),twent_inds))
        plt.legend(['baseline','cardinals','obliques','22'])
        plt.title('bins for getting anisotropy index')
        
        
    return baseline_inds, card_inds, obl_inds, twent_inds


def nonpar_onetailed_ttest(dat1,dat2,niter,rand_seed):
  
  # test if dat2>dat1
  
  assert(len(dat1)==len(dat2))
  dat1 = np.expand_dims(np.squeeze(dat1),axis=1)
  dat2 = np.expand_dims(np.squeeze(dat2),axis=1)
  np.random.seed(rand_seed)    
  real = np.mean(dat2) - np.mean(dat1)  
  rand = np.zeros([niter,1])
  datcomb = np.append(dat1,dat2)
  for ii in range(niter):
    
    randorder = np.random.permutation(len(datcomb))
    datrand = np.reshape(datcomb[randorder],[len(dat1),2])
    rand[ii] = np.diff(np.mean(datrand,axis=0))
  
  p = np.mean(rand>real)

  return p