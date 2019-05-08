from __future__ import division

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError

from pySDC.implementations.problem_classes.sinGeneralizedFisher_1D_FD_implicit import generalized_fisher


# noinspection PyUnusedLocal
class generalized_fisher_jac(generalized_fisher):


    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(generalized_fisher_jac, self).__init__(problem_params, dtype_u, dtype_f)
        
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

      
        
        me.values = (spsolve(sp.eye(self.params.nvars, format='csc') - factor * self.Jf, rhs.values))


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
        # set up boundary values to embed inner points
        lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)
        #ul = (1 + (2 ** (self.params.nu / 2.0) - 1) *
        #      np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[0] + 2 * lam1 * t))) ** (-2 / self.params.nu)
        #ur = (1 + (2 ** (self.params.nu / 2.0) - 1) *
        #      np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[1] + 2 * lam1 * t))) ** (-2 / self.params.nu)

        #uext = np.concatenate(([ul], u.values, [ur]))

        f = self.dtype_f(self.init)
        f.values = self.A.dot(u.values) + self.params.lambda0 ** 2 * u.values * (1 - u.values ** self.params.nu)
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
        lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)
        #ul = (1 + (2 ** (self.params.nu / 2.0) - 1) *
        #      np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[0] + 2 * lam1 * t))) ** (-2 / self.params.nu)
        #ur = (1 + (2 ** (self.params.nu / 2.0) - 1) *
        #      np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[1] + 2 * lam1 * t))) ** (-2 / self.params.nu)


        #uext = np.concatenate(([ul], u.values, [ur]))

        f = self.dtype_f(self.init)
        f.values = self.Jf.dot(u.values) 
        return f

        #f = self.dtype_f(self.init)

        #f.values = self.Jf.dot(u.values)

        #return f




    def build_jacobian(self, u):
        """
        Set the Jacobian of the ODE's right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """
        #lam1 = self.params.lambda0 / 2.0 * ((self.params.nu / 2.0 + 1) ** 0.5 + (self.params.nu / 2.0 + 1) ** (-0.5))
        #sig1 = lam1 - np.sqrt(lam1 ** 2 - self.params.lambda0 ** 2)
        #ul = 0 #(1 + (2 ** (self.params.nu / 2.0) - 1) *
              #np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[0] + 2 * lam1 * t))) ** (-2 / self.params.nu)
        #ur = 1 #(1 + (2 ** (self.params.nu / 2.0) - 1) *
              #np.exp(-self.params.nu / 2.0 * sig1 * (self.params.interval[1] + 2 * lam1 * t))) ** (-2 / self.params.nu)


        #uext = np.concatenate(([ul], u.values, [ur]))
	
        #Jext = sp.diags(self.params.lambda0 ** 2 - self.params.lambda0 ** 2 *(self.params.nu + 1) * uext ** self.params.nu, offsets=0)	

	#self.Jfext = self.A + Jext

        J = sp.diags(self.params.lambda0 ** 2 - self.params.lambda0 ** 2 *(self.params.nu + 1) * u.values ** self.params.nu, offsets=0)

        self.Jf = self.A + J
        

        
