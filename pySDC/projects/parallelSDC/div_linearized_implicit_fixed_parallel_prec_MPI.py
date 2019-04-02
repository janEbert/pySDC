import numpy as np

from mpi4py import MPI
from pySDC.projects.parallelSDC.div_linearized_implicit_fixed_parallel_MPI import div_linearized_implicit_fixed_parallel_MPI


class div_linearized_implicit_fixed_parallel_prec_MPI(div_linearized_implicit_fixed_parallel_MPI):
    """
    Custom sweeper class, implements Sweeper.py

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
        super(div_linearized_implicit_fixed_parallel_prec_MPI, self).__init__(params)

        assert self.params.fixed_time_in_jacobian in range(self.coll.num_nodes + 1), \
            "ERROR: fixed_time_in_jacobian is too small or too large, got %s" % self.params.fixed_time_in_jacobian

	#print(self.QI[1:, 1:])
        self.D, self.V = np.linalg.eig(self.QI[1:, 1:])
        self.Vi = np.linalg.inv(self.V)


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
        Gu_ = self.integrate()

        i=0      
        for m in self.node_list: #
            Gu_[m] -= L.u[m + 1] - L.u[0]
            if L.tau[m] is not None:
                Gu_[m] += L.tau[m]
                    


        Guv = []
        for m in range(M):
            if m in self.node_list:
                Guv.append(np.zeros( Gu_[m].values.size, dtype='d'))
            else:
                Guv.append(None) 

                    
        dnk=np.zeros( Gu_[m].values.size, dtype='d')
        for m in range(M):
            U = np.zeros( Gu_[m].values.size, dtype='d')#complex) #P.dtype_u(P.init, val=0.0)

            if m in self.node_list:
                for j in self.node_list:
                    U = U + (self.Vi[m,j] * Gu_[j].values).flatten()           
                #print(self.rank, "rufe", m, self.rank)
                self.params.comm.Reduce(U, Guv[m], root=self.rank, op=MPI.SUM)  
                Guv[m] = Guv[m].reshape((np.sqrt(Guv[m].size)).astype(int) , (np.sqrt(Guv[m].size)).astype(int) )       
            else:
                for j in self.node_list:
                    U = U + (self.Vi[m,j] * Gu_[j].values).flatten()  
                root=0    
                if self.rank==0:
                    root=1    
                #print(self.rank, "sende", m, root)    
                self.params.comm.Reduce(U, dnk, root=root, op=MPI.SUM)                

  
        
        uv_g = []
        for m in range(M):
            if m in self.node_list:        
                uv_g.append(P.dtype_u(P.init, val=0))
            else:    
                uv_g.append(P.dtype_u(P.init, val=0))
                
        for m in self.node_list: # range(M):  # hell yeah, this is parallel!!

            #if m in self.node_list:
            t1 = MPI.Wtime()  
            uv_g[m].values = (P.solve_system_jacobian(dfdu, Guv[m], L.dt * self.D[m], L.u[0], L.time + L.dt * self.coll.nodes[m]).values)




        for m in range(M):
            U = np.zeros(Gu_[m].values.size, dtype='d')#.flatten() #complex) #P.dtype_u(P.init, val=0.0)
            K = np.zeros(Gu_[m].values.size, dtype='d')#.flatten() #complex) #P.dtype_u(P.init, val=0.0)
            if m in self.node_list:
                for j in self.node_list:
                    U = U + ((self.V[m,j] * uv_g[j].values.flatten()).astype(float))#.flatten()           
                self.params.comm.Reduce(U, K, root=self.rank, op=MPI.SUM)  
                L.u[m + 1].values += K.reshape((np.sqrt(Guv[m].size)).astype(int) , (np.sqrt(Guv[m].size)).astype(int) )       
            else:
                for j in self.node_list:
                    U = U + ((self.V[m,j] * uv_g[j].values.flatten()).astype(float))#.flatten()  
                root=0    
                if self.rank==0:
                    root=1    
                self.params.comm.Reduce(U, dnk, root=root, op=MPI.SUM) 




        for m in range(M): #self.node_list: #  # hell yeah, this is parallel!!
            if m in self.node_list:
                L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        L.status.updated = True

        return None        
