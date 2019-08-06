# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:35:45 2019

@author: jserences, tsheehan adapted from Ruben van Bergen's code (see reference below)
"""
import numpy as np
from invSNC import invSNC
from MatProdTrace import MatProdTrace

import warnings

#ignore by message
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in log")

def fun_minLL_norm(params, W, noise):
        """
        Computes the negative log likelihood of the noise parameters (or more
        accurately, a value proportional to the negative log likelihood up
        to a constant of addition). Also returns the partial derivatives
        of each of the parameters, which are required by minimize.py and
        other efficient optimization algorithms. 
        
        Adapted from Matlab code by Ruben van Bergen, Donders Institute for Brain, Cognition &
        Behavior, 2015/11/02.    
        
        Reference:
        Van Bergen, R. S., Ma, W.J., Pratte, M.S. & Jehee, J.F.M. (2015). 
        Sensory uncertainty decoded from visual cortex predicts behavior. Nature
        Neuroscience.
        """
        
        nvox = noise.shape[1]
        ntrials = noise.shape[0]
        
        tau = params[0:nvox] # voxel specific noise
        sig = params[nvox]   # channel specific noise
        rho = params[nvox+1] # global noise (proportion)
#                
#        if rho<0:
#            print('rho negative')
#        if any(tau<0):
#            print('tau negative')
        # omi -> inverse of omega (see eq. 7)
        omi, NormConst = invSNC(W, tau, sig, rho,want_ld=1) # inverse and logDet of noise cov-matrix          
        XXt = noise.T @ noise # empirical noise correlation
        negloglik = (1/ntrials) * MatProdTrace(XXt, omi) + NormConst # compute log-likelihood for this attempt

        #- If we encounter a degenerate solution (indicated by complex-valued likelihoods), 
        #  make sure that the likelihood goes to infinity.   
        if np.any(np.iscomplex(negloglik)):
            negloglik = np.inf
            
        # MMH 7/2019: adding an extra condition here to look for nans in the
        # NormConst variable. If any of the noise params (rho etc) go negative, 
        # the log function returns a nan. In matlab log of a negative number will
        # return an imaginary number so this wasn't an issue.
        if np.isnan(NormConst):          
#            assert (rho<0 or np.any(tau<0)), 'nonsensical value from function'
            negloglik = np.inf

        # compute derivative
        der = np.full(params.shape[0], np.nan)

        JI = 1-np.eye(nvox)
        R = np.eye(nvox)*float(1-rho) + rho
        ss = np.sqrt(ntrials)
        U = (omi @ noise.T) / ss
        dom = omi @ (np.eye(nvox)-((1/ntrials) * XXt) @ omi)
#        
#        if sig<0:
#            print('sigma negative')
        
        # TS: np.multiply critical! '*' gives matrix multiplication
        der[:nvox] = np.squeeze(2 * np.multiply(dom,R) @ tau)
        der[nvox] = 2 * sig * MatProdTrace(W.T @ omi, W) - np.sum(np.sum(np.power((U.T * float(np.sqrt(2*sig)) @ W),2))) 
        der[nvox+1] = np.sum(np.sum(np.multiply(dom,( np.multiply(np.outer(tau, tau),JI) ))) )     # TS need np.multiply      

        # also checking if sig went negative and gave us a nan for the sqrt function
        if np.any(np.isnan(der)):          
            assert (sig<0), 'nonsensical value from function'
      
        
        return float(negloglik), der