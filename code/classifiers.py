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

def circ_encoder(trndat,trnlabs,tstdat,tstlabs):
    
    """Use encoding model with sin(theta) and cos(theta) as channels
    to predict values in a continuous circular space, spanning [0,180].
    
    Args:
        trndat: [nTrialsTrn x nFeatures]
        trnlabs: [nTrialsTrn x 1], 0-180
        tstdat: [nTrialsTst x nFeatures]
        tstlabs: [nTrialsTst x 1], 0-180
        
    Returns:
        predlabs: [nTrialsTst x 1], 0-180
        circ_corr: a single value for the circular correlation coefficient
        
    """
    assert type(trndat)==np.ndarray
    assert type(tstdat)==np.ndarray
    assert type(trnlabs)==np.ndarray
    assert type(tstlabs)==np.ndarray
    assert np.shape(trnlabs)[1]==1
    assert np.shape(tstlabs)[1]==1
    
    # convert my orientations into two components
    ang = trnlabs/180*2*np.pi;  # first convert to radians
    s = np.sin(ang);
    c = np.cos(ang);
    
    # put these two components into a matrix, two predictors
    trnX = np.concatenate((s,c), axis=1)
    
    # solve the linear system of equations
    w = np.matmul(np.linalg.pinv(trnX), trndat) # verified same thing as the matlab \ operator (trnX\trnDat)
    
    # make predictions on test data
#    x_hat = np.transpose(np.matmul(np.transpose(np.linalg.pinv(w)), np.transpose(tstdat)))  
    x_hat = np.matmul(tstdat, np.linalg.pinv(w))    # verified these two lines do the same thing, and equivalent to matlab's tstDat*pinv(w) (with some small error)
    s_hat = x_hat[:,0]
    c_hat = x_hat[:,1]
    ang_hat = np.arctan(s_hat/c_hat);
    
    # these are restricted to the range -pi to pi. We want them to go
    # from 0-2pi and be exactly where the original angles were. We'll
    # need to use our knowledge of the signs of the s and c components here
    inds = np.where(c_hat<0)[0]
    ang_hat[inds] = ang_hat[inds]+np.pi;
    ang_hat = np.mod(ang_hat,2*np.pi);
    ang_hat = ang_hat[:,np.newaxis]
    
    tstlabs_rad = tstlabs/180*2*np.pi
    circ_corr = circ_corr_coef(tstlabs_rad,ang_hat)
    
    # converting back to degrees here
    predlabs = ang_hat/(2*np.pi)*180

    return predlabs, circ_corr


def circ_regression(trndat,trnlabs,tstdat,tstlabs):
    
    """Use linear multivariate regression to predict values in a continuous
    circular space, spanning [0,180].
    
    Args:
        trndat: [nTrialsTrn x nFeatures]
        trnlabs: [nTrialsTrn x 1], 0-180
        tstdat: [nTrialsTst x nFeatures]
        tstlabs: [nTrialsTst x 1], 0-180
        
    Returns:
        predlabs: [nTrialsTst x 1], 0-180
        circ_corr: a single value for the circular correlation coefficient
        
    """
    assert type(trndat)==np.ndarray
    assert type(tstdat)==np.ndarray
    assert type(trnlabs)==np.ndarray
    assert type(tstlabs)==np.ndarray
    assert np.shape(trnlabs)[1]==1
    assert np.shape(tstlabs)[1]==1
    
    # convert my orientations into two components
    ang = trnlabs/180*2*np.pi;  # first convert to radians
    s = np.sin(ang);
    c = np.cos(ang);
    
    # put these two components into a matrix, two predictors
    trnX = np.concatenate((s,c), axis=1)
    
    # solve the linear system of equations
#    w = np.matmul(np.linalg.pinv(trnX), trndat)
    r = sklearn.linear_model.LinearRegression().fit(trndat,trnX)
    # make predictions on test data
    x_hat = r.predict(tstdat)    
#    x_hat = np.matmul(tstdat, np.linalg.pinv(w))
    
    s_hat = x_hat[:,0]
    c_hat = x_hat[:,1]
    ang_hat = np.arctan(s_hat/c_hat);
   
    # these are restricted to the range -pi to pi. We want them to go
    # from 0-2pi and be exactly where the original angles were. We'll
    # need to use our knowledge of the signs of the s and c components here
    inds = np.where(c_hat<0)[0]
    ang_hat[inds] = ang_hat[inds]+np.pi;
    ang_hat = np.mod(ang_hat,2*np.pi);
    ang_hat = ang_hat[:,np.newaxis]
    
    tstlabs_rad = tstlabs/180*2*np.pi
    
    circ_corr = circ_corr_coef(tstlabs_rad,ang_hat)
    
    # converting back to degrees here
    predlabs = ang_hat/(2*np.pi)*180

    return predlabs, circ_corr



def circ_corr_coef(x, y):
    """ calculate correlation coefficient between two circular variables
    Using Fisher & Lee circular correlation formula (code from Ed Vul)
    x, y are both in radians [0,2pi]
    
    """
   
    assert type(x)==np.ndarray
    assert type(y)==np.ndarray
    assert np.shape(x)==np.shape(y)
    if np.all(x==0) or np.all(y==0):
        raise ValueError('x and y cannot be empty or have all zero values')
    if np.any(x<0) or np.any(x>2*np.pi) or np.any(y<0) or np.any(y>2*np.pi):
        raise ValueError('x and y values must be between 0-2pi')
    n = np.size(x);
    assert(np.size(y)==n)
    A = np.sum(np.cos(x)*np.cos(y));
    B = np.sum(np.sin(x)*np.sin(y));
    C = np.sum(np.cos(x)*np.sin(y));
    D = np.sum(np.sin(x)*np.cos(y));
    E = np.sum(np.cos(2*x));
    Fl = np.sum(np.sin(2*x));
    G = np.sum(np.cos(2*y));
    H = np.sum(np.sin(2*y));
    corr_coef = 4*(A*B-C*D) / np.sqrt((np.power(n,2) - np.power(E,2) - np.power(Fl,2))*(np.power(n,2) - np.power(G,2) - np.power(H,2)));
   
    return corr_coef