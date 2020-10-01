from __future__ import division
import numpy as np
from scipy.linalg import svd, inv, eig, norm

class nnlinear_holom_eigs_solver(object):
    """
        Compute eigenvalues of holomorphic square matrix valued functions given by 
            T(z)v=0, where v nonzero in C^m and z in Gamma 
            Gamma ={z in C^m: f(z)=0} a closed smooth contour, e.g., {z: |z-c|=R, c in C^m and R positive number}
        The contour is assumed to be any diffeomorphism. Thus using trapezoid 
        we can achieve an exponential convergence rate. By default l=2 and the user
        need not to input it since we adaptively compute the best l
        Input
        ====
        m : the size of the matrix
        l : initial guess of the number of eigenvalues inside the domain
        N : discretization size
        mat_func: the holomorphic function should be callable
        cont_func : the contour function in parametrized form, f(t)
        Dcont_func : the derivative of the contour function, i.e., f'(t)
        quad_rule : quadrature rules. Default is trapz, trapezoid
        rankTol : treshold for singular values
        resTol : treshold for the residual (||func(z)v||<=epsilon)
    """
    def __init__(self, m, N, l, K, mat_func, cont_func, Dcont_func=None, quad_rule='trapz', rankTol=1e-4, resTol=1e-6):
        """
        If the derivative is not provided, thinking to implement a second order method to get the Jacobian (finite difference)
        if not Dcont_func:
            return second_order_derivative
        
        """
        self.m = m
        self.l = l
        self.N = N
        self.K = K
        self.mat_func = mat_func
        self.phi = cont_func
        self.Dphi = Dcont_func
        self.quad_rule = quad_rule
        self.rankTol = rankTol
        self.resTol  = resTol 
        
    def BlockMatrix(self, lst):
        """
        a block Hankel matrix, i.e.,
        B = 
        [A1 A2 ...A_k]
        [A2 A3 ...A_{K+1}]
        .
        .
        [A_K A_{K+1} ... A_{2K-1}]
        Input
        =====
        lst : a list containing matrices of dimension m by l
              [A1, A2, ..., A_K, A_{K+1}, ... A_{2K-1}] 
        
        Output
        =====
        return a matrix B of shape Km by Kl 
        
        """
        return np.vstack([np.hstack(lst[i:i+self.K]) for i in range(self.K)])
    def ContourIntegralEval(self, l):
        
        """
        Implement trapezoid to compute the integrals. Adding  other quadrature rules soon
        """
        if not self.Dcont_func:
            #return second_order_derivative
            pass
        t_points = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        Vhat = np.random.randn(self.m, l)
        blck_mat_lst = []
        for p in range(2*self.K):
            A_pN = 0
            for ti in t_points:
                MAT  = np.dot(inv(self.mat_func(self.phi(ti))), Vhat) 
                A_pN = A_pN + MAT*(self.phi(ti)**p)*self.Dphi(ti)
            A_pN = A_pN/(1j*self.N)
            blck_mat_lst.append(A_pN)
        
        B0 = self.BlockMatrix(blck_mat_lst[: -1])
        B1 = self.BlockMatrix(blck_mat_lst[1: ])
        
        V, sigma, Wh = svd(B0, full_matrices=False)
        W            = inv( Wh )
        sigma0       = sigma[np.abs(sigma) > self.rankTol]
        #k is always less or equal to l<=m is the number of eigenvalues inside the contour
        #eigs close to the contour either inside or outside may lead to difficulties in the rank test
        return [sigma0, V, W, B1]

    def eigvals(self):
        """
        compute eigvalues and eigenvectors where l is adaptively computed.
        
        """
        eigenvals = []
        eigenvecs = []
        l = self.l
        while True:
            sigma0, V, W, A_1N =  self.ContourIntegralEval(l)
            k = len(sigma0)
            if k < l*self.K:
                #the inverse of sigma0 might not exists also there might be cases where k=0...need to solve them
                sigma0_inv = np.diag( 1/sigma0 )#the inverse
                V0         = V[:self.K*self.m, :k]
                W0         = W[:self.K*l, :k]
                B          = np.dot( np.dot( V0.conj().T, A_1N), np.dot( W0, sigma0_inv ) )
                EIGVALS, EIGVECTS = eig( B )
                for pos, la_i in enumerate( EIGVALS ):
                    v_i = np.dot( V0[:self.m, :], EIGVECTS[:, pos] )
                    #add option of other norms
                    if norm( np.dot( self.mat_func(s_i), v_i ), np.inf) <= self.resTol:
                        eigenvecs.append(v_i)
                        eigenvals.append(s_i)
                break
                
            else: 
                l = l + 1
        #Neet to think of an efficient and elegant way
        if len(eigenvals) == 0:
            print( 'No eigenvalues were found inside/outside the given countour!')
            pass
        else:
            return (np.array(eigenvals), np.array(eigenvecs))
   
