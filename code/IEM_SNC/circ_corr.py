import numpy as np


def circ_corrcc_better(x,y):
    '''
    calculate correlation coefficient between two circular variables
    using fisher & lee circular correlation formula (code from Ed Vul)
    x, y are both in radians [0.2pi]
    '''
    xu = x.copy()
    yu = y.copy()
    if np.any(x>90): xu = xu.astype(float); xu/=90*np.pi
    if np.any(y>90): yu = yu.astype(float); yu/=90*np.pi
    n 	= len(x)
    A 	= np.sum( np.multiply( np.cos( xu ), np.cos( yu ) ) )
    B 	= np.sum( np.multiply( np.sin( xu ), np.sin( yu ) ) )
    C 	= np.sum( np.multiply( np.cos( xu ), np.sin( yu ) ) )
    D 	= np.sum( np.multiply( np.sin( xu ), np.cos( yu ) ) )
    E 	= np.sum( np.cos( 2*xu ) )
    Fl 	= np.sum( np.sin( 2*xu ) )
    G 	= np.sum( np.cos( 2*yu ) )
    H 	= np.sum( np.sin( 2*yu ) )
    corr_coef = 4*( A*B - C*D ) / np.sqrt( ( n**2 - E**2 - Fl**2 ) * ( n**2 - G**2 - H**2 ) )
    return corr_coef