
# Adapted by TS 5/24/19

# % Given a weight matrix and a precision matrix, decodes orientation &
# % uncertainty estimates from activation patterns (rows) in 'samples'.
# %
# % Written by Ruben van Bergen, Donders Institute for Brain, Cognition &
# % Behavior, 2015/11/02.
# %
# % Reference:
# % Van Bergen, R. S., Ma, W.J., Pratte, M.S. & Jehee, J.F.M. (2015). 
# % Sensory uncertainty decoded from visual cortex predicts behavior. Nature
# % Neuroscience.

import numpy as np
from scipy.integrate import quadrature as quad # quadgk
from scipy.optimize import fmin # fminsearch
from invSNC import invSNC as invSNC
pi = np.pi
tol = 1e-12
def DecodeSNC(test_samples, Pest, W_est):
    # Pest - dict

    if type(Pest) is dict: # not quite sure this is the cleaning I want to do...
        Pest = invSNC(W_est,Pest['tau'], Pest['sig'], Pest['rho'])
        
    n_trials = test_samples.shape[0]
    est,unc  = np.zeros(n_trials),np.zeros(n_trials)
    
    for i in range(n_trials):
        def fun_Eth1(s):
            out = (fun_lik(s)/Integ)*np.exp(1j*s)
            return out
        def fun_lik(s):
            ll = np.exp(-fun_minLL(s)+mll)
            return np.array(np.hstack((ll)))
        def fun_minLL(s):
            bwc = np.tile(b,(len(s),1)).T - W_est@fun_basis(s).T
            negll = 0.5*MatProdDiag(bwc.T@Pest,bwc)
            return negll
        
        def globminsearch():
            # Finds the global minimum of the likelihood in s by doing a coarse
            # search in orientation space first. 
            inits = np.linspace(0,2*pi,200)
            fvals = fun_minLL(inits)
            minI = np.argmin(fvals)
            # Note here I reduce maxiter from 1e10 to 1e4 because sucessful runs were typically 30-70 iterations
            out = fmin(fun_minLL,inits[minI],maxiter=1e4,xtol=1e-10,full_output=True,disp=False)
            sol = out[0]
            mll = out[1]
            assert ~np.isnan(mll), 'Function Failed, check inputs to outer function'
            return (float(sol),mll)
        def MatProdDiag(M1,M2):
            M = np.multiply(M1,M2.T)
            out = np.sum(M,1)
            return out
        
        b = test_samples[i,:]
        _,mll = globminsearch()
        print('evaluating integral for trial %d of %d' % (i,n_trials))
        Integ,_ = quad(fun_lik,0,2*pi,tol=tol,maxiter=600) # numerically evalulate integral
        E1,_ = quad(fun_Eth1,0,2*pi,tol=tol,maxiter=600)
        
        est[i] = np.mod(np.angle(E1),2*pi)/pi*90
        unc[i] = np.sqrt(-2*np.log(np.abs(E1)))/pi*90
    return est, unc

def fun_basis(s):
    tuning_centers = np.arange(0,2*pi-.001,pi/4)
    c = np.maximum(0,np.power(np.cos( np.tile(s,(len(tuning_centers),1)).T -tuning_centers),5)) # differnet than np.max
    return c
    
    
    
    
    