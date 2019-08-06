# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:35:00 2019

@author: jserences
"""

# imports
import numpy as np


def cosd(x):
    """
    convert from deg to rad
    input: x in degrees
    output: d2r in radians
    
    js 03082019
    """
    
    d2r = np.cos( np.deg2rad(x) )
    return d2r

def make_basis_fucntion(xx,mu,n_chans):
    """
    make a raised cos basis function
    note that default is cos**( n_chans - (n_chans % 2) )
    
    input: 
        x: eval range, in rad
        mu: center, in rad
        n_chans: how many channels?
    
    output: 
        basis functions with mean mu evaluated over x
    
    js 03082019
    """
    if not 1 in mu.shape:
        print('adding new axis to p["stimfeat"]')
        mu = mu[:,np.newaxis]
        
    bf = np.power( (cosd(xx-mu) ), ( n_chans - (n_chans % 2) ) )
    
    return bf