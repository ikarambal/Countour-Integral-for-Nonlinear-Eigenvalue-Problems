from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, inv, eig, norm

class nnlinear_holom_eigs_solver(object):
    """
        Compute eigenvalues of holomorphic square matrix valued functions given by 
            T(z)v=0, where v nonzero in C^m and z in Gamma 
            Gamma ={z in C^m: f(z)=0} a closed contour, e.g., {z: |z-c|=R, c in C^m and R positive number}
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
    def __init__(self, m, N, l, K, mat_func, cont_func, Dcont_func, cont_func_params, rankTol=1e-4, resTol=1e-6):
        self.m = m
        self.l = l
        self.N = N
        self.mat_func = mat_func
        self.phi = cont_func
        self.Dphi = Dcont_func
        self.rankTol = rankTol
        self.resTol  = resTol 
        self.cont_func_params = cont_func_params
        self.K = K
    def BlockMatrix(self, lst):
        """
        a block Hankel matrix
        """
        return np.vstack([np.hstack(lst[i:i+self.K]) for i in range(self.K)])
    def ContourIntegralEval(self, l):
        
        """
        Implement trapezoid to compute the integrals
        """
        t_points = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        #Maybe have the user input his/her own points
        #t = np.linspace(start, end, self.N, endpoint=False)
        Vhat = np.random.randn(self.m, l)
        #A_0N = 0.0
        #A_1N = 0.0
        lst = []
        for p in range(2*self.K):
            A_pN = 0
            for ti in t_points:
                MAT  = np.dot(inv(self.mat_func(self.phi(ti))), Vhat) 
                A_pN = A_pN + MAT*(self.phi(ti)**p)*self.Dphi(ti)
                #A_1N = A_1N + MAT*self.phi(ti)*self.Dphi(ti)
            A_pN = A_pN/(1j*N) 
            lst.append(A_pN)
        
        B0 = self.BlockMatrix(lst[:-1])
        B1 = self.BlockMatrix(lst[1:])
        #A_0N = A_0N/(1j*N)
        #A_1N = A_1N/(1j*N)
        V, SIGMA, Wh = svd(B0, full_matrices=False)
        W            = inv( Wh )
        SIGMA0       = SIGMA[np.abs(SIGMA) > self.rankTol]
        #k is always less or equal to l<=m is the number of eigenvalues inside the contour
        #if k=len(SIGMA0)=l then there may be more than l eigenvalues inside the contour
        #eigs close to the contour either inside or outside may lead to difficulties in the rank test
        return [SIGMA0, V, W, B1]

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
            if k < l*self.K:
                #the inverse of sigma0 might not exists also there might be cases where k=0...need to solve them
                SIGMA0_INV = np.diag( 1/SIGMA0 )#the inverse
                V0         = V[:self.K*self.m, :k]
                W0         = W[:self.K*l, :k]
                B          = np.dot( np.dot( V0.conj().T, A_1N), np.dot( W0, SIGMA0_INV ) )
                EIGVALS, EIGVECTS = eig( B )
                print('k', k)
                print(B.shape)
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
    
m = 4
N = 25
c = 0
R = 0.5
K = 2
#rankTol = 1e-4
#resTol = 1e-3
l    = 2
def Mfunc2(z):
    M = np.diag(np.array((3,1,3,1)))
    C = np.array([[0.4,0,-0.3,0], [0,0,0,0],[-0.3,0,0.5,-0.2],[0,0,-0.2,0.2]])
    K = np.array([[-7 ,2, 4,  0], [2 ,-4, 2, 0], [4, 2, -9, 3], [0, 0, 3, -3]])
    return (z**2)*M+z*C+K
def  Mfunc(z):
  T2 = np.array([[0,8, 0],[0, 6,0],[0, 0,1]])
  T1 = np.array([[1, -6, 0], [2, -7, 0], [0, 0, 0]])
  T0 = np.eye(3)#
  return T0+z*T1+(z**2)*T2
cfunc = lambda t: c+R*np.exp(1j*t)
Dcfunc = lambda t: 1j*R*np.exp(1j*t)
params = {'center':c, 'radius':R}
#cont_func_params=**kwargs
solver = nnlinear_holom_eigs_solver(m, N, l, K, Mfunc2, cfunc, Dcfunc, params, rankTol=1e-4, resTol=1e-6)
print(solver.ContourIntegralEval(l))
# z, v= solver.eigvals()
# print(z)
# #=====ploting
# t = np.linspace(0, 2*np.pi, 300, endpoint=True)
# s = cfunc(t)
# x = s.real 
# y = s.imag
# plt.figure()
# plt.plot(x, y)
# plt.plot(z.real, z.imag, 'bo', markersize=6)
# plt.grid('on')
# plt.show()