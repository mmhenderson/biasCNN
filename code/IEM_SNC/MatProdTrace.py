import numpy as np
def MatProdTrace(A, B):
    """
    Computes the trace of a product of 2 matrices efficiently.
    implemented by TS
    """
    return np.inner(A.flatten(),B.T.flatten())
