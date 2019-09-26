import numpy as np
from mpi4py import MPI
from pySDC.projects.parallelPFASST.linearized_implicit_parallel_MPI import linearized_implicit_parallel_MPI


class linearized_implicit_fixed_parallel_MPI(linearized_implicit_parallel_MPI):
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

        super(linearized_implicit_fixed_parallel_MPI, self).__init__(params)

        assert self.params.fixed_time_in_jacobian in range(self.coll.num_nodes + 1), \
            "ERROR: fixed_time_in_jacobian is too small or too large, got %s" % self.params.fixed_time_in_jacobian

        self.D, self.V = np.linalg.eig(self.coll.Qmat[1:, 1:])
        self.Vi = np.linalg.inv(self.V)
        self.newton_itercount = 0
        

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

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

        if L.tau[self.rank] is not None:
            Gu += L.tau[self.rank]
                       
        Gu -= L.u[self.rank + 1] - L.u[0]


        Gu_global = np.zeros( M * Gu.values.size, dtype='d')
   
        Guv = np.zeros( Gu.values.size, dtype=complex)#complex) #dtype='d')
        
        #Guv2 = Guv2.astype(complex)  
        for m in range(self.coll.num_nodes):
            if m == self.rank:
                self.params.comm.Reduce(self.Vi[m, self.rank] * Gu.values ,
                                        Guv, root=m, op=MPI.SUM)
            else:
                self.params.comm.Reduce(self.Vi[m, self.rank] * Gu.values,
                                        None, root=m, op=MPI.SUM)

        Guv = Guv.astype(complex)      

        

        t1 = MPI.Wtime()  
        uv = P.solve_system_jacobian(dfdu, Guv, L.dt * self.D[self.rank], L.u[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])
        t2 = MPI.Wtime()             


        U = np.zeros( Gu.values.size, dtype=complex)#.flattten()


        for m in range(self.coll.num_nodes):
            U = (self.V[m, self.rank] * uv.values)                 	                
            if m == self.rank:
                U+=L.u[self.rank + 1].values
                self.params.comm.Reduce(U.astype(float), L.u[self.rank + 1].values, root=m, op=MPI.SUM)                        
            else:
                self.params.comm.Reduce(U.astype(float), None, root=m, op=MPI.SUM)



        #U = U.reshape((np.sqrt(Gu_global.size/M)).astype(int) , (np.sqrt(Gu_global.size/M)).astype(int) )
        #if(self.rank==0):
        #    print(11,L.u[self.rank+1].values.flatten())
	#    print(22,U)
        #self.params.comm.Barrier()
	#exit()
        
        # evaluate f
        #for m in range(M):  # hell yeah, this is parallel!!
        L.f[self.rank + 1] = P.eval_f(L.u[self.rank + 1], L.time + L.dt * self.coll.nodes[self.rank])



        # indicate presence of new values at this level
        L.status.updated = True
        return None
