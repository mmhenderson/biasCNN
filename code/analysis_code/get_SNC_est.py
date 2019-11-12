#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:18:25 2019

@author: mmhender
"""

import numpy as np
from DecodeSNC_CViter import DecodeSNC_CViter
import time
import multiprocessing as mp

#%%
def run_withinsf_acrossphase(allw, layers2run, info, nUnits2Use, savename=[]):
    """ run the SNC decoder - within each sf, type, and noise level separately, train/test across phase.
    """   
    
    SNC_all = [];
    
#    for ww1 in range(info['nLayers']):
    for ww1 in layers2run:
        
        tmp1 = []
        tmp2 = []
        
        for ww2 in range(info['nTimePts']):
            
#            print([ww1,ww2])
            
            all_est = np.zeros([np.shape(info['typelist'])[0], 1])
            all_unc = np.zeros([np.shape(info['typelist'])[0], 1])
            
            # looping through noise level, spatial frequency, type, and doing training/testing separately within each group.
            for nn in range(info['nNoiseLevels']):               
                for sf in range(info['nSF']):
                     for tt in range(info['nType']):
                       
                        # find my data
                        inds = np.where(np.logical_and(np.logical_and(info['typelist']==tt, info['sflist']==sf), info['noiselist']==nn))[0]
                       
                        thisdat = allw[ww1][ww2][inds,0:nUnits2Use]
                        
                        samples = thisdat
                        p = dict()
                        p["stimfeat"] = np.squeeze(info['orilist'][inds])
                        p["ninit"] = 10
                        
                       
                        cv_labs = info['phaselist'][inds]
                        unlabs = np.unique(cv_labs)
                        small_est = np.zeros([np.shape(cv_labs)[0], 1])
                        small_unc = np.zeros([np.shape(cv_labs)[0], 1])
                        
                        #%% Sequential method
                      
                        tstart = time.time()

                        # run the SNC code for each CV fold
                        for cv in range(np.size(unlabs)):
                            
                            print('Layer %d/%d: Noise %d/%d, SF %d/%d, Type %d/%d, CV %d/%d' %(ww1,info['nLayers'],nn,info['nNoiseLevels'],
                                                                                  sf,info['nSF'],tt,info['nType'],cv,np.size(unlabs)))
                           
                            p["test_trials"] = np.squeeze(cv_labs==unlabs[cv])
                            OUT = DecodeSNC_CViter(samples,p)
                            if type(OUT) is int:
                                print('CV %d Failed' % cv)
                                continue
                            else:
                                [est,unc,Pest] = OUT                           
 
                            small_est[p["test_trials"]==1] = np.expand_dims(est,1)
                            small_unc[p["test_trials"]==1] = np.expand_dims(unc,1)
                            
                        
                        tend = time.time()
                        print('time elapsed: %.2f sec' %(tend-tstart))
                        
                        #%% Parallel method (has a bug currently, dont use)
#                         set up stuff for parallel computing
#                        num_cpus = np.min([10,np.size(unlabs)])                    
#                        my_pool = mp.Pool(num_cpus)
#                                               
#                        tstart = time.time()
#                        
#                        # this function calls the SNC functions, will execute it in parallel.
#                        def wrapper_func(cv):
#                            print('Noise %d/%d, SF %d/%d, Type %d/%d, CV %d/%d' %(nn,info['nNoiseLevels'],
#                                                                                  sf,info['nSF'],tt,info['nType'],cv,np.size(unlabs)))
#                           
#                            p["test_trials"] = np.squeeze(cv_labs==unlabs[cv])
#                            OUT = DecodeSNC_CViter(samples,p)
#                            if type(OUT) is int:
#                                print('CV %d Failed' % cv)
#                                return OUT
#                            else:
#                                [est,unc,Pest] = OUT                           
#                                return est, unc, p["test_trials"]
#                          
#                        cv_list = range(np.size(unlabs))
#                        
#                        # execute my wrapper function in parallel here
#                        pooled_out = my_pool.map(wrapper_func, cv_list)
#                        
#                        # unpack the output of parallel execution
#                        for cv in range(np.size(unlabs)):
#                            OUT = pooled_out[cv]                            
#                            if type(OUT) is int:
#                                print('CV %d Failed' % cv)
#                                continue
#                            [est,unc, inds] = OUT
#                           
#                            small_est[inds==1] = np.expand_dims(est,1)
#                            small_unc[inds==1] = np.expand_dims(unc,1)
#                            
#                        
#                        tend = time.time()
#                        print('time elapsed: %.2f sec' %(tend-tstart))
#                        
                        #%%
                        all_est[inds,:] = small_est
                        all_unc[inds,:] = small_unc
#
            tmp1.append(all_est)
            tmp2.append(all_unc)
                        
        SNC_out = [tmp1,tmp2]
            
        SNC_all.append(SNC_out)
        
        if savename:
            this_savename = savename + '_' + info['layer_labels'][ww1]
            np.save(this_savename, SNC_out)
            
    return SNC_all

#%%
def run_withinnoise_acrossphase(allw, layers2run, info, nUnits2Use, savename):
    """ run the SNC decoder - within each noise level separately, train/test across phase.
    Collapsing over spatial frequency, type (e.g. all SF are equally represented in training set)
    """   
    
    SNC_all = [];
    
    for ww1 in range(info['nLayers']):
        
        tmp1 = []
        tmp2 = []
        
        for ww2 in range(info['nTimePts']):
            
#            print([ww1,ww2])
            
            all_est = np.zeros([np.shape(info['typelist'])[0], 1])
            all_unc = np.zeros([np.shape(info['typelist'])[0], 1])
            
            # looping through noise level, spatial frequency, type, and doing training/testing separately within each group.
            for nn in range(info['nNoiseLevels']):               
                
                for tt in range(info['nType']):
                  
                    # find my data
                    inds = np.where(np.logical_and(info['noiselist']==nn, info['typelist']==tt))[0]
                   
                    thisdat = allw[ww1][ww2][inds,0:nUnits2Use]                                    
                    
                    samples = thisdat
                    p = dict()
                    p["stimfeat"] = np.squeeze(info['orilist'][inds])
                    p["ninit"] = 10
                    
                   
                    cv_labs = info['phaselist'][inds]
                    unlabs = np.unique(cv_labs)
                    small_est = np.zeros([np.shape(cv_labs)[0], 1])
                    small_unc = np.zeros([np.shape(cv_labs)[0], 1])
                    
                    # Sequential method, not parallelized                  
                    tstart = time.time()

                    # run the SNC code for each CV fold
                    for cv in range(np.size(unlabs)):
                        
                        print('Noise %d/%d, Type %d/%d, CV %d/%d' %(nn,info['nNoiseLevels'],
                                                                              tt,info['nType'],cv,np.size(unlabs)))
                       
                        p["test_trials"] = np.squeeze(cv_labs==unlabs[cv])
                        OUT = DecodeSNC_CViter(samples,p)
                        if type(OUT) is int:
                            print('CV %d Failed' % cv)
                            continue
                        else:
                            [est,unc,Pest] = OUT                           
 
                        small_est[p["test_trials"]==1] = np.expand_dims(est,1)
                        small_unc[p["test_trials"]==1] = np.expand_dims(unc,1)
                        
                    
                    tend = time.time()
                    print('time elapsed: %.2f sec' %(tend-tstart))
                    
            
                    all_est[inds,:] = small_est
                    all_unc[inds,:] = small_unc
#            
                        
            tmp1.append(all_est)
            tmp2.append(all_unc)
                
        SNC_all.append([tmp1,tmp2])
        
        if savename:
            np.save(savename, SNC_all)
        
    return SNC_all

#%%
def run_withinnoise_acrosssf(allw, layers2run, info, nUnits2Use, savename):
    """ run the SNC decoder - within each noise level separately, train/test across spatial frequence.
    Collapsing over phase, type (e.g. all phases are equally represented in training set)
    """   
    
    SNC_all = []
    
    for ww1 in range(info['nLayers']):
        
        tmp1 = []
        tmp2 = []
        
        for ww2 in range(info['nTimePts']):
           
            all_est = np.zeros([np.shape(info['typelist'])[0], 1])
            all_unc = np.zeros([np.shape(info['typelist'])[0], 1])
            
            # looping through noise level, spatial frequency, type, and doing training/testing separately within each group.
            for nn in range(info['nNoiseLevels']):               
                  
                for tt in range(info['nType']):
                    
                    # find my data
                    inds = np.where(np.logical_and(info['noiselist']==nn, info['typelist']==tt))[0]
                   
                    thisdat = allw[ww1][ww2][inds,0:nUnits2Use]
                    
                    samples = thisdat
                    p = dict()
                    p["stimfeat"] = np.squeeze(info['orilist'][inds])
                    p["ninit"] = 10
                    
                    cv_labs = info['sflist'][inds]
                    unlabs = np.unique(cv_labs)
                    small_est = np.zeros([np.shape(cv_labs)[0], 1])
                    small_unc = np.zeros([np.shape(cv_labs)[0], 1])
                       
                    # Sequential method, not parallelized                  
                    tstart = time.time()

                    # run the SNC code for each CV fold
                    for cv in range(np.size(unlabs)):
                        
                        print('Noise %d/%d, Type %d/%d, CV %d/%d' %(nn,info['nNoiseLevels'],
                                                                              tt,info['nType'],cv,np.size(unlabs)))
                       
                        p["test_trials"] = np.squeeze(cv_labs==unlabs[cv])
                        OUT = DecodeSNC_CViter(samples,p)
                        if type(OUT) is int:
                            print('CV %d Failed' % cv)
                            continue
                        else:
                            [est,unc,Pest] = OUT                           
 
                        small_est[p["test_trials"]==1] = np.expand_dims(est,1)
                        small_unc[p["test_trials"]==1] = np.expand_dims(unc,1)
                        
                    
                    tend = time.time()
                    print('time elapsed: %.2f sec' %(tend-tstart))
                    
            
                    all_est[inds,:] = small_est
                    all_unc[inds,:] = small_unc
#            
            
            tmp1.append(all_est)
            tmp2.append(all_unc)
            
        SNC_all.append([tmp1,tmp2])
    
        if savename:
            np.save(savename, SNC_all)
        
    return SNC_all

#%%
def run_acrossnoise_acrossphase(allw, layers2run, info, trnNoise, nUnits2Use, savename):
    """ run the SNC decoder - using one noise level as the training set, and testing on each noise level.
    Train/test across phase.
    Collapsing over spatial frequency, type (e.g. all SF are equally represented in training set)
    """   
    
    SNC_all = [];
    
    for ww1 in layers2run:
#    for ww1 in range(info['nLayers']):
        
        tmp1 = []
        tmp2 = []
        
        for ww2 in range(info['nTimePts']):
            
            all_est = np.zeros([np.shape(info['typelist'])[0], 1])
            all_unc = np.zeros([np.shape(info['typelist'])[0], 1])
            

            for pp in range(info['nPhase']):
                
                for tt in range(info['nType']):
                    
                    print('Layer %s: Train Noise %d, Test all noise, Type %d/%d, CV %d/%d' %(info['layer_labels'][ww1], trnNoise,
                                                                          tt,info['nType'],pp,info['nPhase']))
                   
                    # training set is whichever noise level we have specified
                    trninds = np.where(np.logical_and(np.logical_and(info['phaselist']!=pp, info['noiselist']==trnNoise), info['typelist']==tt))[0]
                    # testing set is all noise levels at once, but never the same phase as the training set
                    tstinds = np.where(np.logical_and(info['phaselist']==pp,  info['typelist']==tt))[0]
                
                    assert not np.intersect1d(trninds,tstinds)
                    
                    thisdat_trn = allw[ww1][ww2][trninds,0:nUnits2Use]
                    thisdat_tst = allw[ww1][ww2][tstinds,0:nUnits2Use]
                    
                    samples = np.concatenate((thisdat_trn, thisdat_tst), 0)
                   
                    assert(np.all(np.shape(samples)==(np.size(trninds)+np.size(tstinds), nUnits2Use)))
                    
                    p = dict()
                    p["stimfeat"] = np.concatenate((np.squeeze(info['orilist'][trninds]), np.squeeze(info['orilist'][tstinds])), 0)
                    assert(np.shape(p["stimfeat"])[0]==np.size(trninds)+np.size(tstinds))
                    
                    p["ninit"] = 10
                
                    p["test_trials"] = np.squeeze(np.concatenate((np.zeros([np.size(trninds), 1]), np.ones([np.size(tstinds), 1])), 0))
                    assert(np.shape(p['test_trials'])[0]==np.size(trninds)+np.size(tstinds))
                    
                    tstart = time.time()
                    
                    OUT = DecodeSNC_CViter(samples,p)
                    if type(OUT) is int:
                        print('CV %d Failed' % pp)
                        continue
                    else:
                        [est,unc,Pest] = OUT                           
 
                    all_est[tstinds] = np.expand_dims(est,1)
                    all_unc[tstinds] = np.expand_dims(unc,1)
                    
                    tend = time.time()
                    print('time elapsed: %.2f sec' %(tend-tstart))
                    
            tmp1.append(all_est)
            tmp2.append(all_unc)
        
        SNC_out = [tmp1,tmp2]
        SNC_all.append(SNC_out)
        
        if savename:
            this_savename = savename + '_' + info['layer_labels'][ww1]
            np.save(this_savename, SNC_out)
            
    return SNC_all
