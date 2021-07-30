import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

from mpi4py_fft import newDistArray



class advectiondiffusion_imex(ptype):
    """
    Example implementing the advection-diffusion equation in 2-3D using mpi4py-fft for solving linear parts,
    IMEX time-stepping

    mpi4py-fft: https://mpi4py-fft.readthedocs.io/en/latest/

    Attributes:
        fft: fft object
        X: grid coordinates in real space
        K2: Laplace operator in spectral space
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):



        if 'L' not in problem_params:
            problem_params['L'] = 1.0
        if 'init_type' not in problem_params:
            problem_params['init_type'] = 'circle'
        if 'comm' not in problem_params:
            problem_params['comm'] = None


        if 'c' not in problem_params:
            problem_params['c'] = 1.0

        self.c = problem_params['c']
        self.nu = 1

        #imex needs imex mesh full implicit run just normal mesh
        self.imex= problem_params['imex']
        if not self.imex:
            dtype_f=mesh

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'L', 'spectral']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        if not (isinstance(problem_params['nvars'], tuple) and len(problem_params['nvars']) > 1):
            raise ProblemError('Need at least two dimensions')

        # Creating FFT structure
        ndim = len(problem_params['nvars'])
        self.dim = ndim
        axes = tuple(range(ndim))
        self.fft = PFFT(problem_params['comm'], list(problem_params['nvars']), axes=axes, dtype=np.float, collapse=True)

        # get test data to figure out type and dimensions
        tmp_u = newDistArray(self.fft, problem_params['spectral'])

        # invoke super init, passing the communicator and the local dimensions as init
        super(advectiondiffusion_imex, self).__init__(init=(tmp_u.shape, problem_params['comm'], tmp_u.dtype),
                                             dtype_u=dtype_u, dtype_f=dtype_f, params=problem_params)

        L = np.array([self.params.L] * ndim, dtype=float)

        # get local mesh
        X = np.ogrid[self.fft.local_slice(False)]
        N = self.fft.global_shape()
        for i in range(len(N)):
            X[i] = (X[i] * L[i] / N[i])
        #print("X", X)
        self.X = [np.broadcast_to(x, self.fft.shape(False)) for x in X]

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

        self.K1 = np.sum(K , 0, dtype=float)
        

        # Need this for diagnostics
        self.dx = self.params.L / problem_params['nvars'][0]
        self.dy = self.params.L / problem_params['nvars'][1]

        self.dt = problem_params['dt']
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
        self.num_nodes=3


        #if implicit optimize Q_\Delta just for (-\dt \Laplace)  
        if problem_params['use_RL']:
            if not self.imex:  
                self.QD = np.ndarray(shape=self.K2.shape, dtype=float)    
                tmp = np.ndarray(shape=(self.K2.flatten().size,1),dtype=float, buffer= ((-self.nu*self.K2-self.K1)*self.dt).flatten() )
                self.QD[:,:] = self.model(list(self.model_params), tmp, rng=self.subkey)[:,self.time_rank].reshape(self.K2.shape)
            else:
                self.QD = np.ndarray(shape=self.K2.shape, dtype=float)    
                tmp = np.ndarray(shape=(self.K2.flatten().size,1),dtype=float, buffer= (-self.nu*self.K2*self.dt).flatten() )
                self.QD[:,:] = self.model(list(self.model_params), tmp, rng=self.subkey)[:,self.time_rank].reshape(self.K2.shape)

            #print("MAX", max(self.K2.flatten()*self.dt*self.nu))

    def multQI(self, x):
        f = self.dtype_u(self.init)

        f = x*self.QD 
        return f


    def eval_f(self, u, t):

        f = self.dtype_f(self.init)

        if not self.imex:
            f = -self.nu*self.K2 * u + self.K1 * u*(-1j)
        else:

            f.impl = -self.nu*self.K2 * u
            f.expl = self.K1 * u*(-1j)


        return f

    def solve_system(self, rhs, factor, u0, t):


        self.iters+=1

        self.size = rhs.size


        if not self.imex:
            me = rhs / (1.0 + factor * (self.nu*self.K2-self.K1*(-1j))) #1j
        else:
            me = rhs / (1.0 + factor * self.nu*self.K2)

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


        f=1

        if self.dim ==2:
            tmp= np.sin(f*2*np.pi * (self.X[0] -self.c*t ) )* np.sin(f*2*np.pi * (self.X[1]-self.c *t)) * np.exp(-t * self.dim *(f*2* np.pi)**2* self.nu)
        else:
            tmp= np.sin(f*2*np.pi * (self.X[0] -self.c*t ) )* np.sin(f*2*np.pi * (self.X[1]-self.c *t)) * np.exp(-t * self.dim *(f*2* np.pi)**2* self.nu)* np.sin(f*2*np.pi * (self.X[2]-self.c *t))
        tmp_me[:] = self.fft.forward(tmp)



        return tmp_me
