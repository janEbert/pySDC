import numpy as np

from pySDC.projects.parallelSDC.linearized_implicit_parallel import linearized_implicit_parallel


class linearized_implicit_fixed_parallel(linearized_implicit_parallel):
    """
    Custom sweeper class, implements Sweeper.py

    Generic implicit sweeper, expecting lower triangular matrix QI as input

    Attributes:
        D: eigenvalues of the QI
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'fixed_time_in_jacobian' not in params:
            params['fixed_time_in_jacobian'] = 0

        # call parent's initialization routine
        super(linearized_implicit_fixed_parallel, self).__init__(params)

        assert self.params.fixed_time_in_jacobian in range(self.coll.num_nodes + 1), \
            "ERROR: fixed_time_in_jacobian is too small or too large, got %s" % self.params.fixed_time_in_jacobian

	#print(self.coll.Qmat[1:, 1:])
        self.D, self.V = np.linalg.eig(self.coll.Qmat[1:, 1:])
        self.Vi = np.linalg.inv(self.V)
        #self.newton_itercount = 0
        #self.linear_count = 0
        #self.warning_count = 0


    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # form Jacobian at fixed time
        jtime = self.params.fixed_time_in_jacobian
        dfdu = P.eval_jacobian(L.u[jtime])

        # form collocation problem
        Gu = self.integrate()
        for m in range(M):
            Gu[m] -= L.u[m + 1] - L.u[0]
            if L.tau[m] is not None:
               Gu[m] += L.tau[m]

        # transform collocation problem forward
        Guv = []
        #initial = []
        for m in range(M):
            Guv.append(P.dtype_u(P.init, val=0))
            #initial.append(P.dtype_u(P.init, val=0))
            for j in range(M):
                Guv[m] += self.Vi[m, j] * Gu[j]
                #initial[m] += self.Vi[m,j] * 
                
        

        # solve implicit system with Jacobian (just this one, does not change with the nodes)
        uv = []
        for m in range(M):  # hell yeah, this is parallel!!
            #if m>0:
            #    neu = P.solve_system_jacobian(dfdu, Guv[m].values, L.dt * self.D[m], uv[m-1], L.time + L.dt * self.coll.nodes[m])
            #else:
            #    neu = P.solve_system_jacobian(dfdu, Guv[m].values, L.dt * self.D[m], L.u[m+1], L.time + L.dt * self.coll.nodes[m])
            #uv.append(neu)
            uv.append(P.solve_system_jacobian(dfdu, Guv[m].values, L.dt * self.D[m], L.u[0], L.time + L.dt * self.coll.nodes[m]))
            #print(uv[m].values)
                #P.solve_system_jacobian(dfdu, Guv[m].values, L.dt * self.D[m], L.u[m + 1], L.time + L.dt * self.coll.nodes[m]))
        
        
        # transform soultion backward
        for m in range(M):
            for j in range(M):
                L.u[m + 1] += self.V[m, j] * uv[j]

        # evaluate f
        for m in range(M):  # hell yeah, this is parallel!!
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None
