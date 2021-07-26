import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh#, imex_mesh

from mpi4py_fft import newDistArray


class heat(ptype):
    """
    Example implementing the nonlinear Schrödinger equation in 2-3D using mpi4py-fft for solving linear parts,
    IMEX time-stepping

    mpi4py-fft: https://mpi4py-fft.readthedocs.io/en/latest/

    Attributes:
        fft: fft object
        X: grid coordinates in real space
        K2: Laplace operator in spectral space
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: fft data type (will be passed to parent class)
            dtype_f: fft data type wuth implicit and explicit parts (will be passed to parent class)
        """

        print("IM INIT ###########################################################################################")

        if 'L' not in problem_params:
            problem_params['L'] = 1.0
        if 'comm' not in problem_params:
            problem_params['comm'] = None

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'spectral']

        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if not (isinstance(problem_params['nvars'], tuple) and len(problem_params['nvars']) > 1):
            raise ProblemError('Need at least two dimensions')

        self.nu = problem_params['nu']
        self.dt = problem_params['dt']

        # Creating FFT structure
        ndim = len(problem_params['nvars'])
        axes = tuple(range(ndim))
        self.fft = PFFT(problem_params['comm'], list(problem_params['nvars']), axes=axes, dtype=np.float, collapse=True)

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, problem_params['spectral'])

        # invoke super init, passing the communicator and the local dimensions as init
        super(heat, self).__init__(init=(tmp_u.shape, problem_params['comm'], tmp_u.dtype),
                                             dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        L = np.array([self.params.L] * ndim, dtype=float)

        # get local mesh
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = (X[i] * L[i] / N[i])
        #print("X", X)
        self.X = X #[np.broadcast_to(x, self.fft.shape(False)) for x in X]

        # get local wavenumbers and Laplace operator
        s = self.fft.local_slice()
        N = self.fft.global_shape()
        k = [np.fft.fftfreq(n, 1. / n).astype(int) for n in N[:-1]]
        k.append(np.fft.rfftfreq(N[-1], 1. / N[-1]).astype(int))
        K = [ki[si] for ki, si in zip(k, s)]
        Ks = np.meshgrid(*K, indexing='ij', sparse=True)
        Lp = 2 * np.pi / L
        for i in range(ndim):
            Ks[i] = (Ks[i] * Lp[i]).astype(float)
        K = [np.broadcast_to(k, self.fft.shape(True)) for k in Ks]
        K = np.array(K).astype(float)
        self.K2 = np.sum(K * K, 0, dtype=float)
        self.K2 *= self.nu

        # Need this for diagnostics
        self.dx = self.params.L / problem_params['nvars'][0]
        self.dy = self.params.L / problem_params['nvars'][1]



        self.model = problem_params['model']
        self.model_params = problem_params['model_params'] 
        self.subkey =  problem_params['subkey'] 
        self.rng_key = problem_params['rng_key'] 
        self.time_comm = problem_params['time_comm'] 
        self.time_rank = self.time_comm.Get_rank()
        self.space_comm = problem_params['comm']
        self.space_rank = self.space_comm.Get_rank()
        self.iters = 0
        self.size = 0

        #print(self.time_rank , self.space_rank)

        if problem_params['use_RL']:

            self.QD = np.ndarray(shape=self.K2.shape, dtype=float) 
                
            tmp = np.ndarray(shape=(self.K2.shape[0]*self.K2.shape[1],1),dtype=float, buffer= (-self.K2*self.dt*self.nu).flatten() ) 

            #print("tmp ", tmp.shape  )
            self.QD[:,:] = self.model(self.model_params, tmp)[:,self.time_rank].reshape(self.K2.shape[0], self.K2.shape[1])    
            #print(self.QD)

            #print("vorhersage ",self.model(self.model_params, tmp).shape)
            #for idx, x in np.ndenumerate(self.K2):
            #    if self.K2[idx]*self.dt*self.nu < 2000:
            #        self.QD[idx] = self.model(self.model_params, -x*self.dt*self.nu)[0][self.time_rank] #, rng=self.subkey
            #    else:
            #        self.QD[idx] = self.model(self.model_params, -2000)[0][self.time_rank]   
            #print("MAX", max(self.K2.flatten()*self.dt*self.nu))
            #print(self.time_rank, "min max", min(self.QD.flatten()), max(self.QD.flatten()))
            #assert max(self.K2.flatten()*self.dt*self.nu) < 2000, 'zu gross %s' %max(self.K2.flatten()*0.01*self.nu)  


    def multQI(self, x):
        f = self.dtype_u(self.init)
        f = x*self.QD 
        return f


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

        if self.params.spectral:
            f = -self.K2 * u

        else:

            u_hat = self.fft.forward(u)
            lap_u_hat = -self.K2 * u_hat
            f[:] = self.fft.backward(lap_u_hat, f)

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        self.iters+=1

        self.size = rhs.size

        if self.params.spectral:

            me = rhs / (1.0 + factor * self.K2)

        else:

            me = self.dtype_u(self.init)
            rhs_hat = self.fft.forward(rhs)
            rhs_hat /= (1.0 + factor * self.K2)
            me[:] = self.fft.backward(rhs_hat)

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t, see (1.3) https://arxiv.org/pdf/nlin/0702010.pdf for details

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """




        tmp_me = self.dtype_u(self.init, val=1.0) 
        u = self.dtype_u(self.init, val=1.0) 


        if self.params.spectral:
            tmp= np.sin(2*np.pi * self.X[0])*np.sin(2*np.pi * self.X[1]) * np.exp(-t * 2 *(2* np.pi)**2* self.nu)
            
            tmp_me[:] = self.fft.forward(tmp)


        else:

            tmp_me[:] = np.sin(2*np.pi * self.X[0])*np.sin(2*np.pi * self.X[1]) * np.exp(-t * 2 *(2* np.pi)**2 * self.nu)



        return tmp_me
        #return me
