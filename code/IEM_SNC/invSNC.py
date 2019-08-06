# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:37:29 2019

@author: jserences, adapted from Ruben van Bergen's code (see reference below)  
"""
import numpy as np
from scipy.sparse import csr_matrix # compressed sparse row matrix
from scipy.linalg import lu # pivoted LU decomposition of matrix

def invSNC(W, tau, sig, rho,want_ld=0):
    """
    This is a fast way of computing the inverse and log determinant of the 
    covariance matrix used in Van Bergen, Ma, Pratte & Jehee (2015), Nature 
    Neuroscience, in the form of 2 low-rank updates of a diagonal matrix of 
    variances. 
    
    TS: If you have no idea what is going on here, the following link explains what is being
    used to compute the logDeterminant:
    https://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix.html
    
    Adapted from Matlab code by Ruben van Bergen, Donders Institute for Brain, Cognition &
    Behavior, 2015/11/02.
    
    Reference:
    Van Bergen, R. S., Ma, W.J., Pratte, M.S. & Jehee, J.F.M. (2015). 
    Sensory uncertainty decoded from visual cortex predicts behavior. Nature
    Neuroscience.
    """

    nvox = len(tau)
    nchans = W.shape[1]
    alpha = 1/(1-rho)
    
    Fi = alpha * csr_matrix(np.diag(np.power(tau,-2)), dtype=np.float64)
    ti = 1 / tau
    
    # TS 5/27/19 outer product! 
    Di = Fi - (float(rho*alpha**2) * np.outer(ti,ti) ) / (1 + (rho*nvox*alpha))
    
    DiW = Di @ W
    WtDiW = W.T @ DiW
    
    # TS 5/23/19
    # A = (1 / sig**2*np.eye(nchans)+WtDiW) 
    A = (1 / float(sig**2)*np.eye(nchans)+WtDiW) 
    omi = Di - ((np.linalg.solve(A.T @ A, A.T) @ DiW.T).T @ DiW.T) # verified TS 5/27

    # JS: only called when also want to compute ld? if not this may cause problems...
    
    if want_ld: 
        try:
            ld = my_logdet(csr_matrix(np.eye(nchans)) + float(sig**2) * WtDiW, 'chol') + np.log(1 + rho*nvox*alpha) + nvox*np.log(1-rho) + 2*np.sum(np.log(tau))

        except np.linalg.LinAlgError:        
            # Note from Ruben: If the cholesky decomposition-based version of logdet failed, this may be an
            # indication that your optimization routine is searching in the
            # wrong region of the solution space. If the optimization 
            # finishes shortly after seeing this message, you shouldn't
            # trust the result. This may be due to a bad initialization of
            # the noise parameters.
#            print('Cholesky decomposition-based computation of log determinant failed. Trying with LU decomposition instead.')
            ld = my_logdet(csr_matrix(np.eye(nchans)) + float(sig**2) * WtDiW, 'lu') + np.log(1 + rho*nvox*alpha) + nvox*np.log(1-rho) + 2*np.sum(np.log(tau)) 
            
        except Exception as e: # else would run everytime there IS NOT an error
            # If the failure is not due to a violation of positive-definiteness, rethrow the exception.
            print(e)
        return omi, ld
    else:
        return omi

def my_logdet(X, method):
    """
    Compute the natural log det. 
    Can use Cholesky ('chol') method if positive definite,
    otherwise can use lu factorization 
    
    exp:
    X = np.random.rand(2,2)
    C = X @ X.T     # make positive definite
    v = my_logdet(C, 'lu')
    
    johnserences 03262019
    """
    
    if method == 'chol':
        d = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(X))), axis = 0)
       
    elif method == 'lu':
        P, L, U = lu(X)   #scipy.linalg.lu
        diag_upper = np.diag(U)
        c = np.linalg.det(P) * np.prod(np.sign(diag_upper))
        d = np.log(c) + np.sum(np.log(np.abs(diag_upper)), axis = 0)
    
    else:
        print("Specify a method for logdet - either 'chol' or 'lu'")
        d = np.NaN
        
    return d