from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv, eig, norm

class nnlinear_holom_eigs_solver(object):
    """
        Compute eigenvalues of holomorphic square matrix valued functions given by 
            T(z)v=0, where v nonzero in C^m and z in Gamma 
            Gamma ={z in C^m: f(z)=0}, e.g., {z: |z-c|=R, c in C^m and R positive number}
        The contour is assumed to be 2pi period smooth or any diffeomorphism. Thus using trapezoid 
        we can achieve an exponential convergence rate. By default l=1 and the user
        need not to input it since we adaptively compute the best l
        m : the size of the matrix
        l : initial guess of the number of eigenvalues inside the domain
        N : discretization size
        mat_func: the holomorphic function should be callable
        cont_func : the contour function in parametrized form, f(t)
        params : parameters for the contour function (radius and centre) for smooth transformation from a circle
        Dcont_func : the derivative of the contour function, i.e., f'(t)
        rankTol : treshold for singular values
        resTol : treshold for the residual (||func(z)v||<=epsilon)
    """
    def __init__(self, m, N, l, mat_func, cont_func, Dcont_func=None, cont_func_params={'center':0, 'radius':1}, rankTol=1e-6, resTol=1e-6):
        self.m = m
        self.l = l
        self.N = N
        self.mat_func = mat_func
        self.phi = cont_func
        self.Dphi = Dcont_func
        self.rankTol = rankTol
        self.resTol  = resTol 
        self.cont_func_params = cont_func_params

    def ContourIntegralEval(self, l):
        
        """
        Implement trapezoid to compute the integrals
        """
        t_points = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        #Maybe have the user input his/her own points
        #t = np.linspace(start, end, self.N, endpoint=False)
        
        Vhat = np.random.randn(self.m, l)
        A_0N = 0.0
        A_1N = 0.0
        for ti in t_points:
            MAT  = np.dot(inv(self.mat_func(self.phi(ti))), Vhat) 
            A_0N = A_0N + MAT*self.Dphi(ti)
            A_1N = A_1N + MAT*self.phi(ti)*self.Dphi(ti)

        A_0N = A_0N/(1j*N)
        A_1N = A_1N/(1j*N)
        V, SIGMA, Wh = svd( A_0N)
        W            = inv( Wh )
        SIGMA0       = SIGMA[np.abs(SIGMA) > self.rankTol]
        #k is always less or equal to l<=m is the number of eigenvalues inside the contour
        #if k=len(SIGMA0)=l then there may be more than l eigenvalues inside the contour
        #eigs close to the contour either inside or outside may lead to difficulties in the rank test
        return [SIGMA0, V, W, A_1N]

    def eigvals(self):
        """
        compute eigvalues and eigenvectors where l is adaptively computed.
        
        """
        LAMBDA = []
        VECTORS = []
        c = self.cont_func_params['center']
        R = self.cont_func_params['radius']
        l = self.l
        while True:
            SIGMA0, V, W, A_1N =  self.ContourIntegralEval(l)
            k = len(SIGMA0)
            if k < l:
                #the inverse of sigma0 might not exists also there might be cases where k=0...need to solve them
                SIGMA0_INV = np.diag( 1/SIGMA0 )#the inverse
                V0         = V[:, :k]
                W0         = W[:, :k]
                B          = np.dot( np.dot( V0.conj().T, A_1N), np.dot( W0, SIGMA0_INV ) )
                EIGVALS, EIGVECTS = eig( B )
                for pos, la_i in enumerate( EIGVALS ):
                    eigvect_i = np.dot( V0, EIGVECTS[:, pos] )
                    if norm( np.dot( self.mat_func(la_i), eigvect_i ), np.inf) <= self.resTol and np.abs(la_i - c) < R:
                        
                        VECTORS.append(eigvect_i)
                        LAMBDA.append(la_i)
                        
                break
                
            else: 
                l = l +1
        if len(LAMBDA)==0:
            print( 'No eigenvalues were found inside/outside the given countour!')
            return [0, 0]
        else:
            return (np.array( LAMBDA), np.array(VECTORS))
    
 
