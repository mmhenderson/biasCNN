# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:31:52 2019

@author: jserences, tsheehan, adapted from Ruben van Bergen's code (see reference below)  
"""
import numpy as np
from minimizeNLCG import minimize
from fun_minLL_norm import fun_minLL_norm
def EstimateSNC(noise, W, p):
    """
    Estimates the parameters of the generative model
    
    Adapted from Matlab code by Ruben van Bergen, Donders Institute for Brain, Cognition &
    Behavior, 2015/11/02.
    
    Reference:
    Van Bergen, R. S., Ma, W.J., Pratte, M.S. & Jehee, J.F.M. (2015). 
    Sensory uncertainty decoded from visual cortex predicts behavior. Nature
    Neuroscience.
    """
    # noise, (V) random noise (not related to channels) 
    nvox = noise.shape[1]
    init = np.random.rand(nvox+2)
    init[-2]*=.15

    # JS: start loop over param estimate iterations - want to do this a lot to 
    # make sure that you find best global params
    sol = np.full((nvox+2, p["ninit"]), np.nan)
    
    n_init = p["ninit"]
    fvals = np.ones(n_init)*999 # actual Es should be neg... but just in case
    Es=[]
    good=[]
    for i in range(n_init):
        print('initialization %d of %d' %(i,n_init))
        try:
            # minimize.py for optimization, which is an implementation of the conjugate gradient method by Carl Rasmussen
            sol[:, i], fX, cc = minimize(fun_minLL_norm, np.squeeze(init), W, noise, length=1e12) 
            fvals[i]= fX[-1]
            good.append(i)
            print('    Final value after %d iters is fX=%.4f' %(cc,fX[-1]))
        except Exception as e:
            print('    Minimization Error: %s' % e)
            Es.append(i)
        finally:
            init = np.random.rand(nvox+2) # not using start values !!
            init[-2]*=.15
    
    best_fval = np.min(fvals)
    assert best_fval!=1000, 'No Good Iterations!'
    minI = np.argmin(fvals)
    
    best_sol = sol[:, minI]    
    nPest=dict()
    nPest['tau'] = best_sol[0:nvox]
    nPest['sig'] = best_sol[nvox]
    nPest['rho'] = best_sol[nvox+1]
    nPest['corr'] = np.corrcoef(sol[:,good].T) #TS, need transpose #Gives an estimate of the stability of the solution

    return nPest
    

    



