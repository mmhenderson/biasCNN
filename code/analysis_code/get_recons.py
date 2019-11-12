#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:18:25 2019

@author: mmhender
"""

import numpy as np
import IEM 


def run_withinsf_acrossphase(allw, info, nUnits2Use, savename=[]):
    """ run the IEM - within each sf, type, and noise level separately, train/test across phase.
    """   
    
    chan_resp_all = [];
    
    for ww1 in range(info['nLayers']):
        
        tmp = []
        
        for ww2 in range(info['nTimePts']):
            
            print([ww1,ww2])
            
            chan_resp = np.zeros([np.shape(info['typelist'])[0], 180])
            
            # looping through noise level, spatial frequency, type, and doing training/testing separately within each group.
            for nn in range(info['nNoiseLevels']):               
                for sf in range(info['nSF']):
                     for tt in range(info['nType']):
                       
                        # find my data
                        inds = np.where(np.logical_and(np.logical_and(info['typelist']==tt, info['sflist']==sf), info['noiselist']==nn))[0]
                       
                        thisdat = allw[ww1][ww2][inds,0:nUnits2Use]
                        
                        
                        chan_resp[inds,:] = IEM.run_crossval_IEM_specify_labels(thisdat,info['orilist'][inds],info['phaselist'][inds])
                      
            
            tmp.append(chan_resp)
            
        chan_resp_all.append(tmp)
        
        if savename:
            np.save(savename, chan_resp_all)
        
    return chan_resp_all

def run_withinnoise_acrossphase(allw, info, nUnits2Use, savename):
    """ run the IEM - within each noise level separately, train/test across phase.
    Collapsing over spatial frequency, type (e.g. all SF are equally represented in training set)
    """   
    
    chan_resp_all = [];
    
    for ww1 in range(info['nLayers']):
        
        tmp = []
        
        for ww2 in range(info['nTimePts']):
            
            print([ww1,ww2])
            
            chan_resp = np.zeros([np.shape(info['typelist'])[0], 180])
            
            # looping through noise level, spatial frequency, type, and doing training/testing separately within each group.
            for nn in range(info['nNoiseLevels']):               
                
                for tt in range(info['nType']):
                  
                    # find my data
                    inds = np.where(np.logical_and(info['noiselist']==nn, info['typelist']==tt))[0]
                   
                    thisdat = allw[ww1][ww2][inds,0:nUnits2Use]
                                    
                    chan_resp[inds,:] = IEM.run_crossval_IEM_specify_labels(thisdat,info['orilist'][inds],info['phaselist'][inds])
                  
            
            tmp.append(chan_resp)
            
        chan_resp_all.append(tmp)
    
        if savename:
            np.save(savename, chan_resp_all)
            
        
    return chan_resp_all


def run_withinnoise_acrosssf(allw, info, nUnits2Use, savename):
    """ run the IEM - within each noise level separately, train/test across spatial frequence.
    Collapsing over phase, type (e.g. all phases are equally represented in training set)
    """   
    
    chan_resp_all = [];
    
    for ww1 in range(info['nLayers']):
        
        tmp = []
        
        for ww2 in range(info['nTimePts']):
            
            print([ww1,ww2])
            
            chan_resp = np.zeros([np.shape(info['typelist'])[0], 180])
            
            # looping through noise level, spatial frequency, type, and doing training/testing separately within each group.
            for nn in range(info['nNoiseLevels']):               
                  
                for tt in range(info['nType']):
                    
                    # find my data
                    inds = np.where(np.logical_and(info['noiselist']==nn, info['typelist']==tt))[0]
                   
                    thisdat = allw[ww1][ww2][inds,0:nUnits2Use]
                                    
                    chan_resp[inds,:] = IEM.run_crossval_IEM_specify_labels(thisdat,info['orilist'][inds],info['sflist'][inds])
                  
            
            tmp.append(chan_resp)
            
        chan_resp_all.append(tmp)
    
        if savename:
            np.save(savename, chan_resp_all)
            
        
    return chan_resp_all

def run_acrossnoise_acrossphase(allw, info, trnNoise, nUnits2Use, savename):
    """ run the IEM - using one noise level as the training set, and testing on each noise level.
    Train/test across phase.
    Collapsing over spatial frequency, type (e.g. all SF are equally represented in training set)
    """   
    
    chan_resp_all = [];
    
    for ww1 in range(info['nLayers']):
        
        tmp = []
        
        for ww2 in range(info['nTimePts']):
            
            print([ww1,ww2])

            chan_resp = np.zeros([np.shape(info['typelist'])[0], 180])
            
            # looping through noise level, spatial frequency, type, and doing training/testing separately within each group.
            for nn in range(info['nNoiseLevels']):               

                for pp in range(info['nPhase']):
                    
                    for tt in range(info['nType']):
                    
                        trninds = np.where(np.logical_and(np.logical_and(info['phaselist']!=pp, info['noiselist']==trnNoise), info['typelist']==tt))[0]
                        tstinds = np.where(np.logical_and(np.logical_and(info['phaselist']==pp, info['noiselist']==nn), info['typelist']==tt))[0]
                    
                        assert not np.intersect1d(trninds,tstinds)
                        
                        chan_resp[tstinds,:] = IEM.get_recons(allw[ww1][ww2][trninds],info['orilist'][trninds],allw[ww1][ww2][tstinds])
                 
                
            tmp.append(chan_resp)
            
        chan_resp_all.append(tmp)
        
        if savename:
            np.save(savename, chan_resp_all)
            
    return chan_resp_all
