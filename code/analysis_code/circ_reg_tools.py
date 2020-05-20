#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:49:48 2018

@author: mmhender
"""

import numpy as np
import scipy
import sklearn
from scipy.stats import pearsonr
from scipy.stats import chi2

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
   
    w =np.matmul(np.linalg.pinv(trndat), trnX)
    
    x_hat = np.matmul(tstdat,w)

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

def circ_corrcc(alpha, x):
    """Correlation coefficient between one circular and one linear random
    variable.
    
    Args:
        alpha: vector
            Sample of angles in radians

        x: vector
            Sample of linear random variable

    Returns:
        rho: float
            Correlation coefficient

        pval: float
            p-value

    Code taken from the Circular Statistics Toolbox for Matlab
    By Philipp Berens, 2009
    Python adaptation by Etienne Combrisson
    (copied from brainpipe toolbox)
    """
    if len(alpha) is not len(x):
        raise ValueError('The length of alpha and x must be the same')
    n = len(alpha)

    # Compute correlation coefficent for sin and cos independently
    rxs = pearsonr(x,np.sin(alpha))[0]
    rxc = pearsonr(x,np.cos(alpha))[0]
    rcs = pearsonr(np.sin(alpha),np.cos(alpha))[0]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2));

    # Compute pvalue
    pval = 1 - chi2.cdf(n*rho**2,2);
    
    return rho, pval