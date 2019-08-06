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