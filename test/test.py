import numpy as np
"""
We are going to compare the contour integral result with Matlab polyeig (https://www.mathworks.com/help/matlab/ref/polyeig.html)
In our circular contour, we detected the last two eigenvalues ordered by magnititude
"""
def matrix_func(z):
  M = np.diag(np.array((3,1,3,1)))
  C = np.array([[0.4,0,-0.3,0], [0,0,0,0],[-0.3,0,0.5,-0.2],[0,0,-0.2,0.2]])
  K = np.array([[-7 ,2, 4,  0], [2 ,-4, 2, 0], [4, 2, -9, 3], [0, 0, 3, -3]])
  return (z**2)*M+z*C+K

m = 4
N = 25
c = 0
R = 0.5
K = 2
l = 2
cfunc = lambda t: c+R*np.exp(1j*t)
Dcfunc = lambda t: 1j*R*np.exp(1j*t)
params = {'center':c, 'radius':R}
solver = nnlinear_holom_eigs_solver(m, N, l, K, Mfunc2, cfunc, Dcfunc, params, rankTol=1e-4, resTol=1e-6)
print(solver.eigvals())
"""
>>>eigvalues = array([-0.3465513 +5.99273009e-17j,  0.33529443+1.29575805e-17j]
>>>eigenvectors = array([[-0.43113448+6.04900281e-15j, -0.47098682+6.58366433e-15j,
        -0.48255692+6.74397948e-15j, -0.5022076 +7.01061520e-15j],
       [-0.43380516+5.91761249e-15j, -0.47222392+6.43638952e-15j,
        -0.48409843+6.59621570e-15j, -0.50339399+6.85969640e-15j]]))

"""
  
