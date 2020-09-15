from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv, eig, norm

class nnlinear_holom_eigs_solver(object):
    """
    Compute eigenvalues of holomorphic square matrix valued functions.
        The contour is assumed to be 2pi period smooth. Thus using trapezoid 
        we can achieve an exponential convergence rate. By default l=1 and the user
        need not to input it since we adaptively compute the best l
        m : the size of the matrix
        l : initial guess of the number of eigenvalues inside the domain
        N : discretization size
        func: the holomorphic function should be callable
        c   : the center of the circle
        R   : the radius
        rankTol : treshold for singular values
        resTol : treshold for the residual (||func(z)v||<=epsilon)
    """
    def __init__(self, m, N, func, c, R, l=1, rankTol=1e-6, resTol=1e-6):
        self.m = m
        self.l = l
        self.N = N
        self.MatFunc = func
        self.c = c
        self.R = R
        self.rankTol = rankTol
        self.resTol  = resTol 
    def ContourIntegralEval(self, l):
        
        """
        Implement trapezoid to compute the integrals
        """
        l = self.l
        ti = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        PHI = self.c + self.R*np.exp(1j*ti)
        VHAT = np.random.rand(self.m, l)#uniform distribution
        SUM0 = 0.0
        SUM1 = 0.0
        for i in range( self.N ):
            MAT  = np.dot( inv( self.MatFunc( PHI[i] ) ), VHAT)
            SUM0 = SUM0 + MAT*np.exp( 1j*ti[i] )
            SUM1 = SUM1 + MAT*np.exp( 2j*ti[i] )
        A_0N = self.R*SUM0/self.N
        A_1N = (self.R**2)*SUM1/self.N + self.c*A_0N
        V, SIGMA, Wh = svd( A_0N)
        W            = inv( Wh )
        INDICES = np.where( abs(SIGMA) > self.rankTol )[0]
        #k is always less or equal to l
        k = len( INDICES )
        return [k, SIGMA, V, W, A_1N]

    def eigvals(self):
        """
        compute eigvalues and eigenvectors where l is adaptively computed.
        
        """
        LAMBDA = []
        VECTORS = []
        l = self.l
        while True:
            k, SIGMA, V, W, A_1N =  self.ContourIntegralEval(l)
            print('l',l)
            print('k',k)
            print('SIGMA',SIGMA)
            if k < l:
                #the inverse of sigma0 might not exists also there might be cases where k=0...need to solve them
                SIGMA0 = np.diag( 1/SIGMA[range(k)] )#the inverse
                V0     = V[:, :k]
                W0     = W[:, :k]
                B      = np.dot( np.dot( V0.conj().T, A_1N), np.dot( W0, SIGMA0 ) )
                EIGVALS, EIGVECTS = eig( B )
                for pos, la_i in enumerate( EIGVALS ):
                    eigvect_i = np.dot( V0, EIGVECTS[:, pos] )
                    if norm( np.dot( self.MatFunc(la_i), eigvect_i ), np.inf) <= self.resTol and np.abs(la_i - self.c) < self.R:
                        
                        VECTORS.append(eigvect_i)
                        LAMBDA.append(la_i)
                        
                break
                
            else: 
                l = l +1
        if len(LAMBDA)==0:
            print( 'No eigenvalues were found within the given countour!')
            return 0
        else:
            return (np.array( LAMBDA), np.array(VECTORS))
    
   