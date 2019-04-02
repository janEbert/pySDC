import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_periodic_fullyimplicit 


# noinspection PyUnusedLocal
class AC_jac(allencahn_periodic_fullyimplicit):

    def eval_jacobian(self, u):
        """
        Evaluation of the Jacobian of the right-hand side

        Args:
            u: space values

        Returns:
            Jacobian matrix
        """

        # noinspection PyTypeChecker
        
        dfdu = self.A - 2.0 / self.params.eps ** 2 *sp.diags(u.values - 3.0*u.values**2 + 2.0 * u.values**3, offsets=0)-6.0 * self.params.dw *sp.diags(1.0 -2.0*u.values, offsets=0)

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

        me = self.dtype_u(self.init)
        me.values = spsolve(sp.eye(self.params.nvars) - factor * dfdu, rhs.values)
        self.newton_itercount += 1
        return me
