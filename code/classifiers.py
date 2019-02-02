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
    
def norm_euc_dist(trndat, tstdat, labels_train):
    
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
    pooledvar = np.sum((vareach*np.tile(neach-1,[1,npred])),axis=0)/np.sum(neach-1)
    
    # now loop through test set trials, and find their normalized euclidean
    # distance to each of the training set condition
    normEucDistAll = np.zeros([np.shape(tstdat)[0],nconds])
    for cc in range(nconds):

        tiled_means = np.tile(meanrespeach[cc,:],[np.shape(tstdat)[0],1])
        tiled_var= np.tile(pooledvar, [np.shape(tstdat)[0],1])
        sumofsq = np.sum(np.power((tstdat-tiled_means)/tiled_var, 2),axis=1)
        normEucDistAll[:,cc] = np.sqrt(sumofsq)
            
    # finally, assign a label to each of your testing set trials, choosing from
    # the original labels in group
    column_inds = np.argmin(normEucDistAll, axis=1)
    labels_predicted = np.zeros(np.shape(column_inds))
    
    # make sure these map back into native label space
    un = np.unique(column_inds)
    for uu in range(len(un)):
       
        labels_predicted[np.where(column_inds==un[uu])] = conds[un[uu]]

    return normEucDistAll, labels_predicted, pooledvar
