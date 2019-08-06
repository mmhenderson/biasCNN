#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:18:37 2019

@author: johnserences, adapted from Ruben van Bergen's code (see reference below)  
"""

def gen_test_data(p, nvox = 100, ntrials = 288, nruns = 8, nchans = 8, sig_sim = 0.3, rho_sim = 0.05, span=180):
    """generate some simulated fMRI data to test the model
    
    Adapted from Matlab code by Ruben van Bergen, Donders Institute for Brain, Cognition &
    Behavior, 2015/11/02.
    
    Reference:
    Van Bergen, R. S., Ma, W.J., Pratte, M.S. & Jehee, J.F.M. (2015). 
    Sensory uncertainty decoded from visual cortex predicts behavior. Nature
    Neuroscience.    
        
    Default param values set to those in Ruben's demo matlab code
    Can overide, but in general these work well
    
    js note: i changed a few of these compared to Ruben's code, but in general 
    I am using his defaults.
    
    input (param, default value):
        nvox = 100, number of voxels in simulated data set
        ntrials = 288, number of trials
        nruns = 8, number of experimental runs in the scanner: make sure that 
            ntrials % nruns == 0
        nchans = 8, number of orientation channels to model 
        sig_sim = 0.3, regulates sig of noise
        rho_sim = 0.05, regulates cor structure of noise
        span=180, span of stimulus space. e.g. 180 for orientation and 360 for 
            spatial position
    
    output (in p dictionary): 
        samples, returns a simulated data set of size ntrials, nvox
        runNs, list of experimental runs that correspond to rows in 'samples'
        stimfeat, list of stimulus features (e.g. orientations, locations). 
            corresponds to rows in 'samples' 
    """

    # Simulate generative model parameters & resulting covariance matrix
    W = np.random.randn(nvox, nchans)

    # hardcode for now...could break this into two input params 
    tau_sim = np.random.randn(nvox,1)*0.035+0.7

    # cov matrix to generate noise in simulated data 
    cov_sim = (1-rho_sim)*np.diagflat(tau_sim**2) + rho_sim*(tau_sim*tau_sim.T) + (sig_sim**2) * (W @ W.T)

    # Generate noise based on covariance matrix multiplied
    # by randn noise of size ntrials, nvox
    Q = np.linalg.cholesky(cov_sim)
    noise = (Q @ np.random.randn(ntrials,nvox).T).T    

    # Generate simulated voxel responses based on their tuning profile + noise
    if span==180:
        s_sim = np.random.rand(ntrials, 1) * np.pi
    elif span==360:
        s_sim = np.random.rand(ntrials, 1) * 2 * np.pi
    else:
        print('span must be either 180 or 360')
        return
    
    # generate an x-axis to eval responses from 0...span. Note that this 
    # is in degrees and by changing span you can adapt to 180 or 360deg spaces
    xx = np.linspace(0, span-(span/nchans), nchans)

    # make the basis functions
    c_sim = make_basis_fucntion(xx, s_sim, nchans)

    # generate the simulated data samples
    p["samples"] = (W @ c_sim.T).T + noise

    # Generate info about run number and stimulus feature
    # to use in cross validation and modelling
    if ntrials % nruns != 0:
        print('Please make sure that ntrials is evenly divisible by nruns')
        return

    p["runNs"] = np.sort(np.repeat(np.arange(0,nruns),ntrials/nruns))
    p["stimfeat"] = s_sim * 180/np.pi

    return p


