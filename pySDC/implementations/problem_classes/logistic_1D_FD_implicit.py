from __future__ import division

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Problem import ptype
from pySDC.core.Errors import ParameterError, ProblemError


# noinspection PyUnusedLocal
class logistic(ptype):
    """
    Example implementing the generalized Fisher's equation in 1D with finite differences

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u, dtype_f):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """
        print("inside log init")

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'lambda0', 'newton_maxiter', 'newton_tol', 'interval']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars'] + 1) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p - 1')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(logistic, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        #self.dx = (self.params.interval[1] - self.params.interval[0]) / (self.params.nvars + 1)
        #self.A = self.__get_A(self.params.nvars, self.dx)
        self.newton_counter = 0        
        
        self.inner_solve_counter = 0
        
        
        


    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        A = sp.diags(stencil, [-1, 0, 1], shape=(N , N ), format='lil')
        A *= 1.0 / (dx ** 2)

        return A

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

        u = self.dtype_u(u0)

        #nu = self.params.nu
        lambda0 = self.params.lambda0

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:

            g = u.values - factor*( lambda0 * u.values * (1 - u.values )) - rhs.values
            
                #factor * (self.A.dot(u.values) + lambda0 ** 2 * u.values * (1 - u.values ** nu)) - rhs.values

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            print(n, "res", res, (u.values))	 				

            if res < self.params.newton_tol and n>=1:
                break


	    
            dg = sp.eye(self.params.nvars) - factor * \
                (sp.diags(lambda0 - lambda0 * 2 * u.values , offsets=0))
                #(self.A + sp.diags(lambda0 ** 2 - lambda0 ** 2 * (nu + 1) * u.values ** nu, offsets=0))



            # newton update: u1 = u0 - g/dg
            #print("factor", factor)
            #print(u.values)
            u.values -= spsolve(dg, g)

            # increase iteration count
            n += 1

        # if n == self.params.newton_maxiter:
        #     raise ProblemError('Newton did not converge after %i iterations, error is %s' % (n, res))
        self.newton_counter += n        
        self.inner_solve_counter += n
        #print("zaehler", n)
        
        
        return u

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        f.values = self.params.lambda0 * u.values * (1 - u.values ) # self.A.dot(u.values) + self.params.lambda0 ** 2 * u.values * (1 - u.values ** self.params.nu)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """
        #print("in exact")
        me = self.dtype_u(self.init)

        me.values = 1./(1.+ np.exp([-t])) 
        #print(me.values)
        #print("in exact")
        return me
