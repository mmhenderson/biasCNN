#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:49:48 2018

@author: mmhender
"""

import numpy as np
import scipy

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
