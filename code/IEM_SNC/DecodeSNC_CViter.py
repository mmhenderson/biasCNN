# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:35:47 2019

Performs one iteration (training & testing) of the LORO-CV decoding procedure.

@author: jserences, tsheehan, adapted from Ruben van Bergen's code (see reference below)  
"""
import numpy as np
from DecodeSNC import DecodeSNC, fun_basis
from EstimateSNC import EstimateSNC
def DecodeSNC_CViter(samples, p):
    """
    Performs one iteration (training & testing) of the LORO-CV decoding procedure.
    
    Adapted from Matlab code by Ruben van Bergen, Donders Institute for Brain, Cognition &
    Behavior, 2015/11/02.
    
    Reference:
    Van Bergen, R. S., Ma, W.J., Pratte, M.S. & Jehee, J.F.M. (2015). 
    Sensory uncertainty decoded from visual cortex predicts behavior. Nature
    Neuroscience.
    """
   
    train_samples = samples[p["test_trials"]==0, :]
    test_samples = samples[p["test_trials"]==1, :]
    train_stimori = p["stimfeat"][p["test_trials"]==0]

    # TS 5/24/19:  should use same version as code depicts
    c_train = fun_basis(train_stimori/90*np.pi)

    c_train_inv = np.linalg.pinv(c_train.T)
    W_est  = train_samples.T @c_train_inv # estimated weights, matches matlab mrdivide, aka '/'
    noise_est = train_samples - (W_est @ c_train.T).T # residules of linReg (v + mu), TS 5/24/19: matches matlab
    
    # TS: No point in advancing if Estimating Noise Fails.
    cnt = 0
    not_complete = 1
    while not_complete:    
        print('starting estimation of SNC parameters...')
        Pest = EstimateSNC(noise_est, W_est, p) # fit model parameters by maximizing LL of training data
        if np.isnan(Pest['rho']):
            cnt+=1
            print('Failed:', cnt) 
            if cnt==20:
                print('Failed to Estimate SNC sucessfully after %d attempts' %cnt)
                return 20
        else:
            not_complete=0
    print('starting decoding for test set trials...')
    est, unc = DecodeSNC(test_samples, Pest, W_est) # get estimates for held out trials
    return est, unc, Pest 