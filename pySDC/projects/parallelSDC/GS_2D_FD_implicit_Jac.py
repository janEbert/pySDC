import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres
#from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
from pySDC.implementations.problem_classes.GrayScott_2D_FD import grayscott_fullyimplicit

from scipy.sparse.linalg import cg
import numpy as np

from mpi4py import MPI
# noinspection PyUnusedLocal
class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

class GS_jac(grayscott_fullyimplicit):

    def eval_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        #v = u.values.flatten()
        v0 = u.values[0,:,:].flatten()
        v1 = u.values[1,:,:].flatten()  
               
        #self.A.dot(v) + 1.0 / self.params.eps ** 2 * v * (1.0 - v ** self.params.nu)
        # noinspection PyTypeChecker
        dfdu00 = self.A + 1.0/self.params.eps ** 2 * sp.diags(1.0 - (self.params.nu + 1) * v0 ** self.params.nu, offsets=0)
        #dfdu01 = np.zeros(self.nv**2)
        #dfdu10 = np.zeros(self.nv**2)
        dfdu11 = self.A + 1.0/self.params.eps ** 2 * sp.diags(1.0 - (self.params.nu + 1) * v1 ** self.params.nu, offsets=0)                        
        
        #dfdu=np.zeros([4,self.params.nvars[1]**2,self.params.nvars[1]**2])
        dfdu=np.zeros([2*self.params.nvars[1]*self.params.nvars[1], 2*self.params.nvars[1]*self.params.nvars[1]])
        #print("f", dfdu00.shape)
        #print("s", dfdu[0,:,:].shape)
                
        dfdu[:self.params.nvars[1]*self.params.nvars[1], :self.params.nvars[1]*self.params.nvars[1]] += (dfdu00)
        dfdu[self.params.nvars[1]*self.params.nvars[1]:, self.params.nvars[1]*self.params.nvars[1]:] += (dfdu11)        

        #dfdu = dfdu.flatten()

        return dfdu 


    def solve_system_jacobian(self, dfdu, rhs, factor, u0, t):
        """
        Simple linear solver for (I-dtA)u = rhs

        Args:
            dfdu: the Jacobian of the RHS of the ODE
            rhs: right-hand side for the linear system
            factor: abbrev. for the node-to-node stepsize (or any other factor required)
            u0: initial guess for the iterative solver (not used here so far)
            t: current time (e.g. for time-dependent BCs)

        Returns:
            solution as mesh
        """
        #print(dfdu.shape)
        #print(rhs.shape)
        z = u0.values.flatten()
        z*=0
        #print(z)
        me = self.dtype_u(self.init)
        #me = me.values.flatten()
        #print(dfdu)
        #print(sp.eye(self.params.nvars[0]*self.params.nvars[1]*self.params.nvars[2]))
        
        
        
        #print("shape1", (self.params.nvars[0]*self.params.nvars[1]*self.params.nvars[2])**2)
        #print("shape2", dfdu.shape)
        
        mat = sp.eye((self.params.nvars[0]*self.params.nvars[1]*self.params.nvars[2])) - factor * dfdu

        counter = gmres_counter()
        
        t1 = MPI.Wtime()              
        #new = spsolve(mat, rhs.flatten()) #.values.flatten()) 
        ###nnn =      cg(mat, rhs.flatten(), x0=z, tol=self.params.lin_tol, maxiter=self.params.lin_maxiter, callback=counter)
        nnn =   gmres(mat, rhs.flatten(), x0=z, tol=self.params.lin_tol, maxiter=self.params.lin_maxiter, callback=counter)
        new = nnn[0]
        t2 = MPI.Wtime()             
        #print( "solve system ")    
        #print(counter.niter)
        #print( t2 - t1 );  
        me.values = new.reshape(self.params.nvars)
        self.newton_itercount += 1
        self.linear_count += counter.niter
        self.time_for_solve += t2-t1 
        if counter.niter== self.params.lin_maxiter:
            self.warning_count += 1
                 
        #print(self.newton_itercount)
        #print(me.values)

        return me
