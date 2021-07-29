import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
N = np.array([128, 128, 128], dtype=int)
fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1, 2), dtype=np.float, grid=(-1,))
