#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:49:48 2018

@author: mmhender

Various functions to clasiffy stimulus or calculate information content about 
stimulus from neural activation patterns.

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


def get_norm_euc_dist(dat1,dat2):
    
    """Calculate the normalized euclidean distance (d') between the means of
    two clouds of data points.

    Args:
      dat1: [nPts1 x nWeights] (voxels, spike rates, neural network weights)
      dat2: [nPts2 x nWeights] 
     
    Returns:
       normEucDist (single value)
    """

    assert type(dat1)==np.ndarray
    assert type(dat2)==np.ndarray
    assert np.shape(dat1)[1]==np.shape(dat2)[1]
    
    npts1 = np.shape(dat1)[0]
    npts2 = np.shape(dat2)[0]

    var1 = np.var(dat1,0)
    var2 = np.var(dat2,0)
    
    pooled_var = (var1*(npts1-1)+var2*(npts2-1))/(npts1-1+npts2-1)
    
    mean1 = np.mean(dat1,0)
    mean2 = np.mean(dat2,0)
    
    sq = np.power(mean1-mean2,2)/pooled_var
    normEucDist = np.sqrt(np.sum(sq))

    return normEucDist

def get_euc_dist(dat1,dat2):
    
    """Calculate the euclidean distance (d') between the means of
    two clouds of data points.

    Args:
      dat1: [nPts1 x nWeights] (voxels, spike rates, neural network weights)
      dat2: [nPts2 x nWeights] 
     
    Returns:
       eucDist (single value)
    """

    assert type(dat1)==np.ndarray
    assert type(dat2)==np.ndarray
    assert np.shape(dat1)[1]==np.shape(dat2)[1]
    
    mean1 = np.mean(dat1,0)
    mean2 = np.mean(dat2,0)
    
    sq = np.power(mean1-mean2,2)
    eucDist = np.sqrt(np.sum(sq))

    return eucDist

#def get_mahal_dist(x1,dat2,cov):
#  
#    """Calculate the mahalanobis distance between one point and another cloud 
#    of data points.
#
#      Args:
#      x1: [1 x nWeights] (voxels, spike rates, neural network weights)
#      dat2: [nPts2 x nWeights] 
#     
#      Returns:
#      mahaDist (single value)
#    """
#    
#    assert type(x1)==np.ndarray
#    assert type(dat2)==np.ndarray
#    assert np.shape(x1)[1]==np.shape(dat2)[1]
#    
#    cov2 = np.cov(dat2,rowvar=False)
#    icov2 = np.linalg.inv(cov2)
#    
#    mean2 = np.expand_dims(np.mean(dat2,0),0)
#    
#    diff_vec = x1-mean2
#    
#    mahaDist = np.sqrt(diff_vec @ icov2 @ diff_vec.T)
#    
##    mahaDist2 = scipy.spatial.distance.mahalanobis(x1,mean2,icov2)
#    
#    return mahaDist