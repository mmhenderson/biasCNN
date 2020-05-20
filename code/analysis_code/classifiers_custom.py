#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:49:48 2018

@author: mmhender
"""

import numpy as np
import scipy
import sklearn

def ideal_observer_gaussian(trndat, tstdat, labels_train):
    
    """Compute the likelihood function in stimulus space for each trial in 
    tstdat, using data in trndat and labels to train the model.

    Args:
      trndat: [nTrialsTrn x nWeights] (voxels, spike rates, neural network weights)
      tstdat: [nTrialsTst x nWeights] 
      labels_train: [nTrialsTrn x 1] (labels for the TRAINING data)

    Returns:
       likelihoods: [nTrialsTst x nLabels] (likelihood function, evaluated at each unique location in labels vector)
       labels_predicted: [nTrialsTst x 1] the most likely label for each trial in test.
    """
 
    
    conds=np.unique(labels_train)
    nconds=len(conds)
    npred=np.shape(trndat)[1];
    
    ntrials_tst = np.shape(tstdat)[0]
   
    likelihoods=np.zeros([ntrials_tst,nconds])
    log_likelihoods=np.zeros([ntrials_tst,nconds])

    mus=np.zeros([nconds,npred])
    sigmas=np.zeros([nconds,npred])
   
    for cc in range(nconds):

        mus[cc,:] = np.mean(trndat[np.where(labels_train==conds[cc])[0], :], axis=0)
        sigmas[cc,:] = np.std(trndat[np.where(labels_train==conds[cc])[0], :], axis=0)
          
    lik_tmp = np.zeros([ntrials_tst, nconds, npred])
    ll_tmp = np.zeros([ntrials_tst, nconds, npred])
    
    for pp in range(npred):
          
        # compute likelihood of this cond for each test data pattern 
        lik = scipy.stats.norm.pdf(tstdat[:,pp][:,None],mus[:,pp],sigmas[:,pp])    
        
        lik_tmp[:,:,pp] = lik
        
        ll_tmp[:,:,pp] = np.log(lik)

    likelihoods = np.prod(lik_tmp,axis=2)
    log_likelihoods = np.sum(ll_tmp, axis=2)
            
        
#    vals1 = np.argmax(log_likelihoods, axis=1)
#    vals2 = np.argmax(likelihoods, axis=1)
#    print(len(np.where(vals1!=vals2)))
#    assert np.array_equal(np.argmax(log_likelihoods, axis=1), np.argmax(likelihoods, axis=1))
    column_inds = np.argmax(log_likelihoods, axis=1)
    labels_predicted = np.zeros(np.shape(column_inds))
    
    # make sure these map back into native label space
    un = np.unique(column_inds)
    for uu in range(len(un)):
       
        labels_predicted[np.where(column_inds==un[uu])] = conds[un[uu]]

    return likelihoods, log_likelihoods, labels_predicted
    
def class_norm_euc_dist(trndat, tstdat, labels_train):
    
    """Classify the data in trndat according to labels in labels_train, using
    the normalized Euclidean distance.

    Args:
      trndat: [nTrialsTrn x nWeights] (voxels, spike rates, neural network weights)
      tstdat: [nTrialsTst x nWeights] 
      labels_train: [nTrialsTrn x 1] (labels for the TRAINING data)

    Returns:
       normEucDist: [nTrialsTst x nLabels] (distance to each label category)
       labels_predicted: [nTrialsTst x 1] the most likely label for each trial in test.
    """

    conds=np.unique(labels_train)
    nconds=len(conds)
    npred=np.shape(trndat)[1];
    
    # first, go through each condition, get its mean, variance and number of
    # samples in training set
    meanrespeach = np.zeros([nconds,npred])
    vareach = np.zeros([nconds,npred])
    neach = np.zeros([nconds,1])
    for cc in range(nconds):
        # find the trials of interest in training set    
        meanrespeach[cc,:] = np.mean(trndat[np.where(labels_train==conds[cc])[0],:],axis=0)
        vareach[cc,:] = np.var(trndat[np.where(labels_train==conds[cc])[0],:],axis=0)
        neach[cc] = len(np.where(labels_train==conds[cc])[0])
        
    # use this to get the pooled variance for each voxel


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
        varpooled[ii] = np.sum(pooled_var)
        
        # J(theta) = f'(theta).^2 / variance(f(theta));
        diff2 = np.power(np.mean(dat1,0)-np.mean(dat2,0),2)
        deriv2[ii] = np.sum(diff2)
        
        fi_allneurons = diff2/pooled_var
        
        fi[ii] = np.sum(fi_allneurons)

    # to be perfectly correct -when delta is odd, then the center of each comparison is technically 0.5 degrees off of integer orientation.
    if np.mod(delta,2):
      ori_axis = np.mod(ori_axis+0.5, max_ori)
    return ori_axis, fi, deriv2, varpooled
  
def get_discrim_func(data, ori_labs, step_size=1):
    
    """ Get the discriminability between neighboring orientation bins, as a function of orientation.
    Assume that ori_labs spans a circular space, where the max and the min value are 1 deg apart.
    """
    
    ori_axis,ia = np.unique(ori_labs, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==ori_labs)
    disc = np.zeros([np.size(ori_axis),1])
    max_ori = np.max(ori_axis)+1
    
    steps_left = np.int8(np.floor(step_size/2))
    steps_right = np.int8(np.ceil(step_size/2))
    
    
    for ii in np.arange(0,np.size(ori_axis)):
        
        # find gratings at the orientations of interest
        inds_left = np.where(ori_labs==np.mod(ori_axis[ii]-steps_left, max_ori))[0]
        inds_right = np.where(ori_labs==np.mod(ori_axis[ii]+steps_right, max_ori))[0]
        
        assert(np.size(inds_left)==np.size(inds_right) and not np.size(inds_left)==0)
        
        if np.size(inds_left)==1:
          dist = get_euc_dist(data[inds_right,:],data[inds_left,:])
        else:
          dist = get_norm_euc_dist(data[inds_right,:],data[inds_left,:])

        disc[ii] = dist

    ori_axis = np.mod(ori_axis+0.5, max_ori)
    return ori_axis, disc
 
def get_discrim_func_binned(data, ori_labs, bin_size=5):
    
    """ Get the discriminability between neighboring orientation bins, as a function of orientation.
    Assume that ori_labs spans a circular space, where the max and the min value are 1 deg apart.
    """
    
    ori_axis,ia = np.unique(ori_labs, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==ori_labs)
    max_ori = np.max(ori_axis)+1      
    assert(np.mod(max_ori,bin_size)==0)
    n_bins = np.int64(np.size(ori_axis)/bin_size)
    
    # shift over so that bins are centered on cardinals
#    ori_shifted = ori_axis
    ori_shifted = np.mod(ori_axis-np.floor(bin_size/2), max_ori)
    ori_bins = np.reshape(ori_shifted, [bin_size,n_bins],order='F')
   
    disc = np.zeros([np.size(ori_axis),1])
    
    # this is still the same orientation axis as the non-binned version, we'll just put in duplicate values because the bins are non-overlapping.
    ori_axis = np.mod(ori_axis+0.5, max_ori)
    
    for bb in range(n_bins):
      
      # find gratings at the orientations of interest
      inds_left = np.where(np.isin(ori_labs, ori_bins[:,bb]))[0]
      
      if bb<n_bins-1:
        inds_right = np.where(np.isin(ori_labs, ori_bins[:,bb+1]))[0]
      else:
        inds_right = np.where(np.isin(ori_labs, ori_bins[:,0]))[0]
      assert(np.size(inds_left)==np.size(inds_right) and not np.size(inds_left)==0)
      
      if np.size(inds_left)==1:
          dist = get_euc_dist(data[inds_right,:],data[inds_left,:])
      else:
          dist = get_norm_euc_dist(data[inds_right,:],data[inds_left,:])

      # this same value will go into the array in multiple positions - there are in total bin_size positions to place it in. 
      center_ori = ori_bins[bin_size-1,bb]+0.5
      ori_to_use = np.mod(np.arange(center_ori - np.floor(bin_size/2), center_ori+np.ceil(bin_size/2), 1), max_ori)
      inds_real = np.where(np.isin(ori_axis, ori_to_use))
      
      disc[inds_real] = dist        
   
   
    return ori_axis, disc
  
  
def get_discrim_func_not_normalized(data, ori_labs):
    
    """ Get the discriminability between neighboring orientation bins, as a function of orientation.
    Assume that ori_labs spans a circular space, where the max and the min value are 1 deg apart.
    """
    
    ori_axis,ia = np.unique(ori_labs, return_inverse=True)
    assert np.all(np.expand_dims(ia,1)==ori_labs)
    disc = np.zeros([np.size(ori_axis),1])
    max_ori = np.max(ori_axis)+1
    
    for ii in np.arange(0,np.size(ori_axis)):
        
        # find gratings at the orientations of interest
        inds_left = np.where(ori_labs==np.mod(ori_axis[ii], max_ori))[0]
        inds_right = np.where(ori_labs==np.mod(ori_axis[ii]+1, max_ori))[0]
        
        assert(np.size(inds_left)==np.size(inds_right) and not np.size(inds_left)==0)
        
#        if np.size(inds_left)==1:
        dist = get_euc_dist(data[inds_right,:],data[inds_left,:])
#        else:
#            dist = get_norm_euc_dist(data[inds_right,:],data[inds_left,:])

        disc[ii] = dist

    ori_axis = np.mod(ori_axis+0.5, max_ori)
    return ori_axis, disc