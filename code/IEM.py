#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:46:07 2018

@author: mmhender
"""

import numpy as np


def make_basis_function(xx, mu, basis_pwr = 8):
    """ Define a raised cosine function at each point in xx.
    Arguments:
        xx: vector of points to evaluate the function, in degrees (typically span 0:179)
        mu: center of the basis function, in degrees
        basis_pwr: power to raise the function, typically nchannels-1 (default = 8).
    Returns: 
        funct: a vector with same size as xx
 
    """
    
    xx_rad = xx*np.pi/180
    mu_rad = mu*np.pi/180
    funct = np.power((np.cos(xx_rad-mu_rad)), basis_pwr)
    funct = np.maximum(funct,0)
    return funct

def make_stim_mask(ori_labs, xx):
    """ Make a stimulus mask matrix.
    Arguments:
        ori_labs: a vector nTrials long, listing orientation on each trial in degrees.
        xx: nPts long vector; points at which the reconstruction will be evaluated (typically 0:179)
    Returns:
        stim_mask: a matrix of [nTrials x nPts], binary stimulus mask.
    """
  
    stim_mask = np.zeros([len(ori_labs),len(xx)]);
    for tt in range(len(ori_labs)):        
        myind = np.where(ori_labs[tt]==xx)[0];
        if len(myind)!=1:
            raise ValueError('error filling in stim mask. check that each element in ori_labs is in xx.')
        
        stim_mask[tt,myind]=1; # put "1" in the right orientation column
        
    return stim_mask
    
def get_recons(trndat, ori_labs_trn, tstdat, n_ori_chans = 9):
    """ Run the IEM, using all trials in trndat as training set and all trials in tstdat as testing set.
    Arguments:
        trndat: [nTrials (training) x nVoxels]
        ori_labs_trn: [nTrials (training) x 1]
        tstdat: [nTrials (testing) x nVoxels]
        n_ori_chans: number of channels in basis set. 9 is default.
    Returns:
        chan_resp: [nTrials (testing) x 180]; reconstruction for each trial in 1:180 degree space.
    
    """
     
    # check a couple things in the inputs
    if np.shape(trndat)[1]!=np.shape(tstdat)[1]:
        raise ValueError('number of features in training and testing sets must be equal')
    if np.shape(trndat)[1]<n_ori_chans:
        raise ValueError('must have at least as many features as channels')    
    if np.mod(180,n_ori_chans):
        raise ValueError('n_chans must be a factor of n_pts (180)')

    n_pts = 180      
    xx = np.arange(0,n_pts, 1)        
    basis_pwr = n_ori_chans-1
 
    stim_mask = make_stim_mask(ori_labs_trn, xx)
   
    chan_resp = np.zeros([np.shape(tstdat)[0], len(xx)]);   
  
    # start the loop from 1:20 (e.g. if num chans == 9). Each time shift the
    # basis set centers (mu's) to completely cover the space after all
    # iterations.
    n_steps = int(n_pts/n_ori_chans)
    
    for b in range(n_steps):

        basis_set = np.zeros([n_pts,n_ori_chans]); # making the basis set for this iteration.
        chan_center = np.arange(b, n_pts, n_steps); # my channel centers on this iteration of b
#        print(chan_center)
        for cc in range(n_ori_chans):
            basis_set[:,cc] = make_basis_function(xx,chan_center[cc], basis_pwr);
        
        # now generate the design matrix
        trnX = np.matmul(stim_mask,basis_set);
        trnX = trnX/np.max(np.max(trnX))
        
        if np.linalg.matrix_rank(trnX)!=np.shape(trnX)[1]:
            raise ValueError('rank deficient training set Design Matrix!')
            
        # then solve for the weights!
        w = np.matmul(np.linalg.pinv(trnX), trndat)
#        w = w/np.max(np.max(w))
        # compute the stimulus reconstruction for each trial, filling in 
        # the orientations corresponding to the current centers of the 
        # basis functions. 
        
        chan_resp[:,chan_center] = np.matmul(tstdat, np.linalg.pinv(w));
             
    return chan_resp

def run_crossval_IEM(dat,ori_labs,n_folds = 10, n_ori_chans = 9):
    """ Run the IEM on all data, using n_folds of cross-validation, leave out a random subset of trials.
    Arguments:
        dat: [nTrials (total) x nVoxels]
        ori_labs: [nTrials (total) x 1]
        n_folds: Number of cross-validation folds. Default is 10.
    Returns:
        chan_resp: [nTrials (total) x 180]; reconstruction for each trial in 1:180 degree space.
    
     """
    
    nTrialsTotal = np.shape(dat)[0]
    nPerCV = int(np.ceil(nTrialsTotal/n_folds))
    whichCV = np.tile(np.arange(0,n_folds), [1,nPerCV])
    
    # if the division isn't perfect, trim off a couple of trials from each fold
    if np.size(whichCV)>nTrialsTotal:
        whichCV = whichCV[0][0:nTrialsTotal]
    else:
        whichCV = whichCV[0]
        
    allinds=  np.arange(0,nTrialsTotal)
    np.random.shuffle(allinds)
    
    chan_resp = np.zeros([nTrialsTotal, 180])

    for ff in range(n_folds):
        
        trninds = allinds[np.where(whichCV!=ff)[0]]
        tstinds = allinds[np.where(whichCV==ff)[0]]
#        print('training set has %d trials, testing set has %d trials' % (np.size(trninds), np.size(tstinds)))
        # call get_recons for this fold
        chan_resp_tmp = get_recons(dat[trninds,:], ori_labs[trninds,:], dat[tstinds,:],n_ori_chans)

        chan_resp[tstinds,:] = chan_resp_tmp    
        
    return chan_resp

def shift_and_average(chan_resp, ori_labs, center_deg = 90):
    """ Shift all reconstructions to a common center and average them.
    Arguments:
        chan_resp: [nTrials x 180] reconstruction for each trial
        ori_labs: [nTrials x 1] real label for each trial, in degrees
        center_deg: integer 0:179, where you want to position the center of the recons (default 90).
    Returns:
        average_recons: [1 x 180], the average reconstruction over all trials.
    """
    
    recons_shift = np.zeros(np.shape(chan_resp));
    for t in range(np.shape(chan_resp)[0]):
        recons_shift[t,:] =  np.roll(chan_resp[t,:],center_deg - ori_labs[t])

    average_recons = np.mean(recons_shift, axis=0);
    
    return average_recons


def shift_flip_and_average(chan_resp, ori_labs, dir_labs, center_deg = 90):
    """ Shift all reconstructions to a common center, flip a specified subset, and average them.
    Arguments:
        chan_resp: [nTrials x 180] reconstruction for each trial
        ori_labs: [nTrials x 1] real label for each trial, in degrees
        dir_labs: [nTrials x 1] label all trials that need to be flipped with -1
        center_deg: integer 0:179, where you want to position the center of the recons (default 90).
    Returns:
        average_recons: [1 x 180], the average reconstruction over all trials.
    """

    recons_shift = np.zeros(np.shape(chan_resp));
    recons_shift_to90 = np.zeros(np.shape(chan_resp));
    
#    plt.figure();plt.pcolormesh(chan_resp);plt.colorbar();plt.title('before shift')
    
    for t in range(np.shape(chan_resp)[0]):
        
        # first center it at 90, then flip around the middle
        this_rec = np.expand_dims(np.roll(chan_resp[t,:],90 - ori_labs[t]), 0)

        if dir_labs[t]==-1:
            this_rec = np.fliplr(this_rec)
            
        recons_shift_to90[t,:] = this_rec
        # then center it wherever you really want it
        recons_shift[t,:] =np.roll(this_rec, center_deg - 90)
        
    test = np.roll(recons_shift_to90, center_deg - 90, axis=1)
    if not np.array_equal(test, recons_shift):
        print('oops')
#    else:
#        print('match')
        
#    plt.figure();plt.pcolormesh(recons_shift_to90);plt.colorbar();plt.title('after shift to 90')
#         
#    plt.figure();plt.pcolormesh(recons_shift);plt.colorbar();plt.title('after shift to 45')
      
    average_recons = np.mean(recons_shift, axis=0);
    
    return average_recons