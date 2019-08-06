# IEM_SNC
(python 3.x)

Code translated from van Bergen et al 2015
Started: JS
Completed: 5/28/19 TS 

Useful translation notes:

Matlab A'/B'    ->  Python A.T@np.pinv(B.T)

Matlab min(0,X) -> np.minimum(0,X)

Matlab A.*B     -> np.multipy(A,B)

Testing:

5/28/2019 | Converged with Matlab output for same dataset & showed reasonable estimates and uncertainties

6/1/2019  | Code performs better than expected on test dataset. Performance substantially better than IEM and follow similar patterns to that data (eg. IPS fails to predict across conditions)
