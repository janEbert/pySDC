from __future__ import division

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

#from pySDC.implementations.problem_classes.sinGeneralizedFisher_1D_FD_implicit import generalized_fisher
from pySDC.implementations.problem_classes.logistic_1D_FD_implicit import logistic

# noinspection PyUnusedLocal
class logistic_jac(logistic): #generalized_fisher):


    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(logistic_jac, self).__init__(problem_params, dtype_u, dtype_f)
        
        self.Jf = None
        #self.Jfext = None

        self.inner_solve_counter = 0
    



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

      
        #print(factor)
        #print(self.Jf)
        #me.values = gmres(spsolve(sp.eye(self.params.nvars, format='csc') - factor * self.Jf, rhs.values, x0=u0, tol=self.params.lin_tol, maxiter=1000) )
        me.values = (spsolve(sp.eye(self.params.nvars, format='csc') - factor * self.Jf, rhs.values))
        #print("solve syst")

        self.inner_solve_counter += 1

        return me         


    def eval_f_ode(self, u, t):
        """
        Routine to evaluate the RHS of the ODE

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS of the ODE
        """
        
        
        
        

        #lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        #sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)


        f = self.dtype_f(self.init)
        f.values =   self.params.lambda0 * u.values * (1 - u.values)             #self.A.dot(u.values) + self.params.lambda0 ** 2 * u.values * (1 - u.values ** self.params.nu)
        
        
        return f
        



    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS of the linear system (i.e. Jf times e)

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        #lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        #sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)


        f = self.dtype_f(self.init)
        f.values = self.Jf.dot(u.values) 
        return f






    def build_jacobian(self, u):
        """
        Set the Jacobian of the ODE's right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        #print(u.values)
        J = sp.diags(self.params.lambda0 - 2.*self.params.lambda0*  u.values , offsets=0)  # sp.diags(self.params.lambda0 ** 2 - self.params.lambda0 ** 2 *(self.params.nu + 1) * u.values ** self.params.nu, offsets=0)

        self.Jf = J #self.A + J
        

        
