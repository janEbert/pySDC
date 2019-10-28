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

        N=self.params.nvars[1]      

        v0 = u.values[0,:,:].flatten()
        v1 = u.values[1,:,:].flatten()  


        dfdu=np.zeros([2*self.params.nvars[1]*self.params.nvars[1], 2*self.params.nvars[1]*self.params.nvars[1]])
        
        dfdu00 = self.params.D0*self.A 
        dfdu11 = self.params.D1*self.A                        
        

        dfdu = sp.bmat([[dfdu00-sp.diags(u.values[1,:,:].flatten()**2 - 1.0 * self.params.f, offsets=0), sp.diags(-2*u.values[0,:,:].flatten() *u.values[1,:,:].flatten(), offsets=0)],[sp.diags(u.values[1,:,:].flatten()**2, offsets=0), dfdu11+sp.diags(2.0* u.values[0,:,:].flatten() * u.values[1,:,:].flatten() -1.0 *(self.params.f +self.params.k), offsets=0)]])   
              
        #dfdu[:self.params.nvars[1]*self.params.nvars[1], :self.params.nvars[1]*self.params.nvars[1]] += (dfdu00)
        #dfdu[self.params.nvars[1]*self.params.nvars[1]:, self.params.nvars[1]*self.params.nvars[1]:] += (dfdu11)        

 
        #dfdu +=    sp.diags( [u.values[1,:,:].flatten()**2  , np.append(-u.values[1,:,:].flatten()**2 - 1.0 * self.params.f,  2.0* u.values[0,:,:].flatten() * u.values[1,:,:].flatten() -1.0 *(self.params.f +self.params.k)), -2*u.values[0,:,:].flatten() *u.values[1,:,:].flatten()] , [-int(N*N), 0, int(N*N)])

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
        
        
        #print("########################################################################")
        #print(type(rhs))
        #exit()
        
        #if not(isinstance(rhs, np.ndarray)): #if (type(rhs.dtype) == <type 'np.ndarray'>):
        #    rhs = rhs.values
        
        nnn =   gmres(mat, rhs.flatten(), x0=z, tol=self.params.lin_tol, maxiter=self.params.lin_maxiter, callback=counter)   ## ruth
        #nnn =   gmres(mat, rhs.values.flatten(), x0=z, tol=self.params.lin_tol, maxiter=self.params.lin_maxiter, callback=counter)
        #print("################################### das hat geklappt")
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
