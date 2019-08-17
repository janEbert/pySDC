from __future__ import division
import timeit
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.implementations.datatype_classes.mesh import mesh

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


# noinspection PyUnusedLocal
class grayscott_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'Du', 'Dv', 'f', 'k', 'inner_maxiter', 'inner_tol'] #gray
        
        #essential_keys = ['nvars', 'nu', 'eps', 'inner_maxiter', 'inner_tol', 'radius'] #allen
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)


        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if problem_params['nvars'] % 4 != 0: #4
            raise ProblemError('the setup requires nvars = 2^p per dimension')


        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(grayscott_fullyimplicit, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        self.dx = 200.0 / self.params.nvars

        self.A = self.__get_A(self.params.nvars, self.params.Du, self.params.Dv, self.dx)
        

        self.xvalues = np.append( np.array([i * self.dx for i in range(int(self.params.nvars/2.))]), np.array([i * self.dx for i in range(int(self.params.nvars/2.))])) 

	#exit()
        self.inner_solve_counter = 0
        #self.newton_ncalls = 0
        
        self.newton_itercount = 0
        self.newton_ncalls = 0

    @staticmethod
    def __get_A(N, Du, Dv, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            nu (float): diffusion coefficient
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """
        

        
       
        values_d = np.array([-2.0,-2.0])    #diogonal elements  
        values_nd = np.array([1.0])       #lower and upper diagonal elements 
        values_periodic = np.array([1.0]) #additional elements because of periodic boundarys


        for i in range(N-2):
            values_nd = np.append(values_nd, [1.0])
            values_d = np.append(values_d, [-2.0])
	
        for i in range(int(N/2)):
            values_periodic = np.append(values_periodic, [0.0])
	    
	
        values_periodic[int(N/2)]=1.0
        values_nd[int(N/2-1)]=0.0
	
        values_d[0:int(N/2)] *= Du
        values_d[int(N/2+1):N] *=Dv
	 
        values_periodic[0:int(N/2)] *= Du
        values_periodic[int(N/2+1):N] *= Dv

        values_nd[0:int(N/2)] *= Du
        values_nd[int(N/2+1):N] *= Dv		
	
        B = sp.diags([values_periodic, values_nd, values_d, values_nd, values_periodic], [-int(N/2-1),-1, 0, 1, int(N/2-1)], format='csc')    

   
        
        B *= 1.0 / (dx ** 2)

        return B

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """
	#just for PFASST_Newton

        me = self.dtype_u(self.init)

        me.values[:] = u0.values


        Id = sp.eye(self.params.nvars)

        # start newton iteration
        n = 0
        res = 99
        

                
        while n < self.params.inner_maxiter:

            # form the function g with g(u) = 0
            g = me.values - factor * self.eval_f(me,0).values - rhs.values

            res = np.linalg.norm(g, np.inf)

            exit()
            if res < self.params.inner_tol:
                break

            # assemble dg
            dg = Id - factor * self.build_jacobian(me)


            start = timeit.timeit()


            # newton update: u1 = u0 - g/dg
            me.values -= spsolve(dg, g)
            
            end = timeit.timeit()

            n += 1


        self.newton_ncalls += 1
        self.inner_solve_counter += n
        
        self.newton_itercount += n
 
        return me    





    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        
        N=self.params.nvars
                
        f = self.dtype_f(self.init)

        f.values[0:int(N/2)]  = - u.values[0:int(N/2)] * u.values[int(N/2):N]**2 +  self.params.f * (1.0 - u.values[0:int(N/2)])
        f.values[int(N/2):N]  =   u.values[0:int(N/2)] * u.values[int(N/2):N]**2 - (self.params.f + self.params.k) * u.values[int(N/2):N]
	
        f.values += self.A.dot(u.values)



        return f

    def build_jacobian(self, u): 
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """


        
        N=self.params.nvars      
        
        
        dfdu =  self.A + sp.diags( [u.values[int(N/2):N]**2  , np.append(-u.values[int(N/2):N]**2 - 1.0 * self.params.f,  2.0* u.values[0:int(N/2)] * u.values[int(N/2):N] -1.0 *(self.params.f +self.params.k)), -2*u.values[0:int(N/2)] * u.values[int(N/2):N]]     , [-int(N/2), 0, int(N/2)])
             
                         
        return dfdu

    def eval_jacobian(self, u): 
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """


        
        N=self.params.nvars      
        
        
        dfdu =  self.A + sp.diags( [u.values[int(N/2):N]**2  , np.append(-u.values[int(N/2):N]**2 - 1.0 * self.params.f,  2.0* u.values[0:int(N/2)] * u.values[int(N/2):N] -1.0 *(self.params.f +self.params.k)), -2*u.values[0:int(N/2)] * u.values[int(N/2):N]]     , [-int(N/2), 0, int(N/2)])
             
               
                        
        return dfdu


    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        for i in range(int(self.params.nvars/2)):

            me.values[i] = 1 - 0.5 * np.power(np.sin(np.pi * self.xvalues[i] / 100), 100)
            me.values[int(i+(self.params.nvars/2))] = 0.25 * np.power(np.sin(np.pi * self.xvalues[i] / 100), 100)

  
	
	
	
	
        return me
