from __future__ import division

import numpy as np
import scipy.sparse as sp
from pySDC.implementations.datatype_classes.mesh import mesh

from scipy.sparse.linalg import spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

#from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_fullyimplicit
from pySDC.implementations.problem_classes.GrayScott_1D_FD_neu import grayscott_fullyimplicit

# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


# noinspection PyUnusedLocal
class grayscott_fullyimplicit_jac(grayscott_fullyimplicit):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC (Jacobi formulation)

    Attributes:
        Jf: Jacobi matrix of the collocation problem
        inner_solve_counter (int): counter for the number of linear solves
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(grayscott_fullyimplicit_jac, self).__init__(problem_params, dtype_u, dtype_f)

        self.Jf = None

        self.inner_solve_counter = 0

    def solve_system_jacobian(self, dfdu, rhs, factor, u0, t):
        """
        Linear solver for the Jacobian

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time

        Returns:
            dtype_u: solution u
        """

        me = self.dtype_u(self.init)

        me.values = spsolve(sp.eye(self.params.nvars, format='csc') - factor * dfdu, rhs)


        self.inner_solve_counter += 1
        print(len(me.values) ," ", self.inner_solve_counter)
        return me

    def solve_system(self, rhs, factor, u0, t):
        """
        Linear solver for the Jacobian

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time

        Returns:
            dtype_u: solution u
        """

        me = self.dtype_u(self.init)

        me.values = spsolve(sp.eye(self.params.nvars, format='csc') - factor * self.Jf, rhs.values)


        self.inner_solve_counter += 1
        print(len(me.values) ," ", self.inner_solve_counter)
        return me

    def eval_f_ode(self, u, t):
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

    #def eval_f(self, u, t):
        """
        Routine to evaluate the RHS of the linear system (i.e. Jf times e)

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        #f = self.dtype_f(self.init)

        #f.values = self.Jf.dot(u.values)

        #return f

    def eval_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """


        
        N=self.params.nvars      
        
        
        J = sp.diags( [u.values[int(N/2):N]**2  , np.append(-u.values[int(N/2):N]**2 - 1.0 * self.params.f,  2.0* u.values[0:int(N/2)] * u.values[int(N/2):N] -1.0 *(self.params.f +self.params.k)), -2*u.values[0:int(N/2)] * u.values[int(N/2):N]]     , [-int(N/2), 0, int(N/2)])
             
               
        self.Jf = self.A + J  
        return  self.A + J

    def build_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """


        
        N=self.params.nvars      
        
        
        J = sp.diags( [u.values[int(N/2):N]**2  , np.append(-u.values[int(N/2):N]**2 - 1.0 * self.params.f,  2.0* u.values[0:int(N/2)] * u.values[int(N/2):N] -1.0 *(self.params.f +self.params.k)), -2*u.values[0:int(N/2)] * u.values[int(N/2):N]]     , [-int(N/2), 0, int(N/2)])
             
               
        self.Jf = self.A + J                         
 

