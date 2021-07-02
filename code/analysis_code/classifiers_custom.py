#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:49:48 2018

@author: mmhender

Various functions to calculate information content about 
stimulus from neural activation patterns. Used for biasCNN project.

"""

import numpy as np
  
def get_fisher_info(data, ori_labs, delta=1):
    """ calculate the fisher information across orientation space (estimate the 
    slope of each unit's tuning at each point, square, divide by variance, and sum)
    """
    ori_axis,ia = np.unique(ori_labs, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==ori_labs)
    fi = np.zeros([np.size(ori_axis),1])
    deriv2 = np.zeros([np.size(ori_axis),1])
    varpooled = np.zeros([np.size(ori_axis),1])
    
    max_ori = np.max(ori_axis)+1
    # steps left and right should sum to delta
    steps_left = np.int8(np.floor(delta/2))
    steps_right = np.int8(np.ceil(delta/2))
    
    # if data values are very big, adjust so they don't give infinity values.
    if np.any(np.isinf(data**2)):
      print('found large values in data, dividing all data values by median')
      data = data/np.percentile(data,50)
      
    for ii in np.arange(0,np.size(ori_axis)):
        
        # want to get the slope at this point. Take data from two orientations that are delta deg apart
        inds_left = np.where(ori_labs==np.mod(ori_axis[ii]-steps_left, max_ori))[0]
        inds_right = np.where(ori_labs==np.mod(ori_axis[ii]+steps_right, max_ori))[0]
        
        assert(np.size(inds_left)==np.size(inds_right) and not np.size(inds_left)==0)
        nreps = np.size(inds_left)
        
        dat1 = data[inds_left,:]
        dat2 = data[inds_right,:]
        # variance of respones within each unit, for these two stimulus
        # values of interest only.
        var1 = np.var(dat1,axis=0)
        var1 = np.transpose(np.expand_dims(var1,axis=1))
        var2 = np.var(dat2,axis=0)
        var2 = np.transpose(np.expand_dims(var2,axis=1))
        neach = np.tile(nreps,(2,1))
        pooled_var = np.sum(np.concatenate((var1,var2),0)*np.tile(neach-1,(1,np.shape(data)[1])),0)/np.sum(neach-1);
        # if any values are zero for variance - replace them with very small number here.
        pooled_var[pooled_var==0] = 10**(-12)
        varpooled[ii] = np.sum(pooled_var)
        
        # for Gaussian distributed noise:
        # J(theta) = f'(theta).^2 / variance(f(theta));
        diff2 = np.power(np.mean(dat1,0)-np.mean(dat2,0),2)
        if np.any(np.isinf(diff2)):
          print('warning: some squared slope vals are infinite')
          
        deriv2[ii] = np.sum(diff2)
        
        fi_allneurons = diff2/pooled_var
        
        fi[ii] = np.sum(fi_allneurons)

    # to be perfectly correct -when delta is odd, then the center of each comparison is technically 0.5 degrees off of integer orientation.
    if np.mod(delta,2):
      ori_axis = np.mod(ori_axis+0.5, max_ori)
    return ori_axis, fi, deriv2, varpooled


def get_fisher_info_cov(data, ori_labs, delta=1):
    """ calculate the fisher information across orientation space, using covariance matrix between all units.
    Note for this to work, have to have fewer units than number of trials each orientation.
    So usually want to do PCA first.
    
    """
    ori_axis,ia = np.unique(ori_labs, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==ori_labs)
    fi = np.zeros([np.size(ori_axis),1])
   
    max_ori = np.max(ori_axis)+1
    # steps left and right should sum to delta
    steps_left = np.int8(np.floor(delta/2))
    steps_right = np.int8(np.ceil(delta/2))
    
    # if data values are very big, adjust so they don't give infinity values.
    if np.any(np.isinf(data**2)):
      print('found large values in data, dividing all data values by median')
      data = data/np.percentile(data,50)
      
    for ii in np.arange(0,np.size(ori_axis)):
        
        # want to get the slope at this point. Take data from two orientations that are delta deg apart
        inds_left = np.where(ori_labs==np.mod(ori_axis[ii]-steps_left, max_ori))[0]
        inds_right = np.where(ori_labs==np.mod(ori_axis[ii]+steps_right, max_ori))[0]
        
        assert(np.size(inds_left)==np.size(inds_right) and not np.size(inds_left)==0)

        dat1 = data[inds_left,:]
        dat2 = data[inds_right,:]
        npts1 = np.shape(dat1)[0]
        npts2 = np.shape(dat2)[0]
    
        cov1 = np.cov(dat1,rowvar=False)
        cov2 = np.cov(dat2,rowvar=False)
                
        # get pooled covariance matrix
        # weighted average of two group covariance matrices
        cov_pooled = (cov1*(npts1-1) + cov2*(npts2-1))/(npts1+npts2-2)
        icov_pooled = np.linalg.inv(cov_pooled)
    
        deriv_cov = (cov1-cov2)/delta
        # get "derivative" estimate by subtracting

        deriv = (np.mean(dat1,0)-np.mean(dat2,0))/delta
        if np.any(np.isinf(deriv)):
          print('warning: some slope vals are infinite')
          
        # FI = f'(theta)^T * covariance.^(-1) * f'(theta)
#        fi_allneurons = np.transpose(deriv) @ icov_pooled @ deriv
        
#         FI = f'(theta)^T * Q.^(-1) * f'(theta) + 1/2 Trace(Q' Q^(-1) Q' Q^(-1)        
        fi_allneurons = np.transpose(deriv) @ icov_pooled @ deriv + 0.5*np.trace(deriv_cov @ icov_pooled @ deriv_cov @ icov_pooled)
        
        fi[ii] = fi_allneurons

    # to be perfectly correct -when delta is odd, then the center of each comparison is technically 0.5 degrees off of integer orientation.
    if np.mod(delta,2):
      ori_axis = np.mod(ori_axis+0.5, max_ori)
      
    return ori_axis, fi
 
def get_tstat_dist(data, ori_labs, delta=1):
  
  """ Calculate all pairwise distances between two point clouds, using 
  Euclidean distance in multidimensional space.
  Calculate the mean distance and divide by standard deviation over distances.
  """
  ori_axis,ia = np.unique(ori_labs, return_inverse=True)
  assert np.all(np.expand_dims(ia,1)==ori_labs)
  
  assert(np.shape(data)[0]==len(ori_labs))
  nOri = len(np.unique(ori_labs))
  nEx = int(np.shape(data)[0]/nOri)
  
  max_ori = np.max(ori_axis)+1
  # steps left and right should sum to delta
  steps_left = np.int8(np.floor(delta/2))
  steps_right = np.int8(np.ceil(delta/2))

  tsep = np.zeros([nOri,1])
  for ii in range(nOri):
  
    # want to get the slope at this point. Take data from two orientations that are delta deg apart
    inds_left = np.where(ori_labs==np.mod(ori_axis[ii]-steps_left, max_ori))[0]
    inds_right = np.where(ori_labs==np.mod(ori_axis[ii]+steps_right, max_ori))[0]
          
    dat1 = data[inds_left,:]
    dat2 = data[inds_right,:]

    distances =np.zeros([nEx, nEx])
    
    for ee1 in range(nEx):
      for ee2 in range(nEx):
        
        distances[ee1,ee2] = np.sqrt(np.sum((dat1[ee1,:]-dat2[ee2,:])**2))
    
    distances=np.ravel(distances)
    tsep[ii] = np.mean(distances)/(np.std(distances))
    
  return tsep
