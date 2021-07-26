#mpirun -n 6 python heat_pySDC_with_mpi4pyfft.py -n 2

from argparse import ArgumentParser
import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.parallelSDC.generic_implicit_MPI import generic_implicit_MPI
from pySDC.implementations.problem_classes.1D_Heat_MPIFFT import heat
#from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers


num_nodes = 3

def build_model(M):
    scale = 1e-3
    glorot_normal = jax.nn.initializers.variance_scaling(
        scale, "fan_avg", "truncated_normal")
    normal = jax.nn.initializers.normal(scale)
    (model_init, model_apply) = stax.serial(
        stax.Dense(64, glorot_normal, normal),
        stax.Relu,
        # stax.Dense(256),
        # stax.Relu,
        stax.Dense(64, glorot_normal, normal),
        stax.Relu,
        stax.Dense(M, glorot_normal, normal),
    )
    return (model_init, model_apply)




def load_model(path):
    with open(path, 'rb') as f:
        weights = jnp.load(f, allow_pickle=True)
    with open(str(path) + '.steps', 'rb') as f:
        steps = jnp.load(f, allow_pickle=True)
    return weights, steps

def run_simulation(spectral=None, ml=None, nprocs_space=None, sweeper_class=None, use_RL = None, MIN3=None):
    """
    A test program to do SDC, MLSDC and PFASST runs for the 2D NLS equation

    Args:
        spectral (bool): run in real or spectral space
        ml (bool): single or multiple levels
        num_procs (int): number of parallel processors
    """


    seed = 0
    eval_seed = seed
    if eval_seed is not None:
        eval_seed += 1

    rng_key = jax.random.PRNGKey(0)
    model_path = "models/dp_model_2021-07-18T18-22-56.855290.npy" #dp_model_2021-06-24T09-55-45.128649.npy" #dp_model_2021-05-13T16-27-27.359957.npy" #complex_model_2021-06-29T12-51-32.544928.npy"

    model_init, model = build_model(num_nodes)
    rng_key, subkey = jax.random.split(rng_key)
   
    params, _ = load_model(model_path)

    #rng_key = jax.random.PRNGKey(0)
    #rng_key, subkey = jax.random.split(rng_key)



    #rng_key, subkey = jax.random.split(rng_key)

    #model_path = "complex_model_2021-06-29T12-51-32.544928.npy" #"dp_model_2021-06-24T09-55-45.128649.npy"

    #params, _ = load_model(model_path)


    #_, model = build_model(3, train=False)



    #action = model(params, -1j, rng=subkey)


    #model = jax.jit(model) 


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


    world_rank = comm.Get_rank()
    world_size = comm.Get_size()


    assert world_size%nprocs_space == 0, 'Total number of processors must be multiple of number of processors in space'  

    # split world communicator to create space-communicators
    if nprocs_space is not None:
        color = int(world_rank / nprocs_space)
    else:
        color = int(world_rank / 1)
    space_comm = comm.Split(color=color)
    space_comm.Set_name('Space-Comm')
    space_size = space_comm.Get_size()
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    if nprocs_space is not None:
        color = int(world_rank % nprocs_space)
    else:
        color = int(world_rank / world_size)
    time_comm = comm.Split(color=color)
    time_comm.Set_name('Time-Comm')
    time_size = time_comm.Get_size()
    time_rank = time_comm.Get_rank()





    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-12
    level_params['dt'] = 0.1
    #assert level_params['dt'] == 0.1, 'Zeitschritt muss in Problemklasse angepasst werden'
   
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [num_nodes]

    if sweeper_class == 'generic_implicit_MPI':
        assert sweeper_params['num_nodes'][0] == time_size, 'Need %s processors in time' % sweeper_params['num_nodes']   

    sweeper_params['initial_guess'] = 'spread' #zero'


    sweeper_params['comm'] = time_comm #MPI.COMM_WORLD

    # initialize problem parameters
    problem_params = dict()
    #if ml:
    #    problem_params['nvars'] = [(128, 128), (32, 32)]
    #else:


    problem_params['nvars'] = [(64, 64)]
    problem_params['nu'] = 0.001
    problem_params['spectral'] = spectral
    problem_params['comm'] = space_comm
    problem_params['time_comm'] = time_comm
    problem_params['model'] = model
    problem_params['model_params'] = params
    problem_params['subkey'] = subkey
    problem_params['rng_key'] = rng_key
    problem_params['dt'] = level_params['dt']

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30 if rank == 0 else 99
    # controller_params['predict_type'] = 'fine_only'

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    description['problem_class'] = heat
    description['sweeper_class'] = sweeper_class #generic_imex_MPI # imex_1st_order
    if sweeper_class == generic_implicit_MPI:
        if use_RL:
            sweeper_params['QI'] = ['RL'] 
            problem_params['use_RL'] = True 

        else:
            if MIN3:
                sweeper_params['QI'] = ['MIN3'] 
                problem_params['use_RL'] = False 
            else:
                sweeper_params['QI'] = ['MIN'] 
                problem_params['use_RL'] = False 

    else:
        sweeper_params['QI'] = ['LU']  
        problem_params['use_RL'] = False 





    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    #description['space_transfer_class'] = fft_to_fft




    # set time parameters
    t0 = 0.0
    Tend = 0.5

    f = None
    if rank == 0:
        f = open('step_7_B_out.txt', 'a')
        out = f'Running with ml={ml} and num_procs={world_size}...'
        f.write(out + '\n')
        print(out)

    # instantiate controller

    MPI.COMM_WORLD.Barrier()    
    wt = MPI.Wtime()
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    MPI.COMM_WORLD.Barrier()    
    run_time = MPI.Wtime()
    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    MPI.COMM_WORLD.Barrier()    
    wt = MPI.Wtime() - wt
    wt2 = MPI.Wtime() - run_time

    if False:
        plt.subplot(221)
        Z = P.fft.backward(uinit).reshape(32, 32)
        plt.imshow(Z, interpolation='bilinear')
        plt.title('initial')
        plt.colorbar()

    if False:
        plt.subplot(222)
        Z2 = P.fft.backward(uend).reshape(128, 128)
        plt.imshow(Z2, interpolation='bilinear')
        plt.title('numeric')
        plt.colorbar()

    uex = P.u_exact(Tend)

    if False:
        plt.subplot(223)
        Z3 = P.fft.backward(uex).reshape(128, 128)
        plt.imshow(Z3, interpolation='bilinear')
        plt.title('exact')
        plt.colorbar()
        plt.show()

    err =  np.linalg.norm(uex - uend, np.inf) #abs(uex - uend) #np.linalg.norm(uex - uend, np.inf)
    abw = np.linalg.norm(uinit - uend, np.inf) #abs(uinit - uend) #np.linalg.norm(uinit - uend, np.inf)


    if False:
        np.set_printoptions(precision=16)

        print(np.max(P.fft.backward(uinit).real))

        print("##### shape", uend.shape)
        print("uex, uend ", np.max(   (P.fft.backward(uex)).real) , np.max((P.fft.backward(uend)).real))

        print("uinit, uend " , np.max(   (P.fft.backward(uinit)).real), np.max((P.fft.backward(uend)).real))


    #print(world_rank, err, P.iters, P.size)

    print(world_rank, P.iters )

    if rank == 0:
        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        niters = np.array([item[1] for item in iter_counts])
        out = f'   Min/Mean/Max number of iterations: ' \
              f'{np.min(niters):4.2f} / {np.mean(niters):4.2f} / {np.max(niters):4.2f}'
        f.write(out + '\n')
        print(out)
        #out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
        #f.write(out + '\n')
        #print(out)
        #out = '   Position of max/min number of iterations: %2i -- %2i' % \
        #      (int(np.argmax(niters)), int(np.argmin(niters)))
        #f.write(out + '\n')
        #print(out)
        #out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        #f.write(out + '\n')
        #print(out)

        out = f'Error: {err:6.4e}'
        f.write(out + '\n')
        print(out)

        abw2 = f'Abw: {abw:6.4e}'
        f.write(abw2 + '\n')
        print(abw2)

        #timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
        #out = f'Time to solution: {timing[0][1]:6.4f} sec.'
        #f.write(out + '\n')
        #print(out)


        print('Mean number of iterations %s' %( np.mean(niters)))

        print("TIME", wt, wt2 )
        f.write('\n')
        print()
        f.close()


def main():
    """
    Little helper routine to run the whole thing

    Note: This can also be run with "mpirun -np 2 python B_pySDC_with_mpi4pyfft.py"
    """
    #parser = ArgumentParser()
    #parser.add_argument("-n", "--nprocs_space", help='Specifies the number of processors in space', type=int)
    #args = parser.parse_args()


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()


    n_space = int(world_size/num_nodes)



    MPI.COMM_WORLD.Barrier()    
    if rank ==0: print("############ RL")
    MPI.COMM_WORLD.Barrier()
    run_simulation(spectral=True, ml=False, nprocs_space=n_space, sweeper_class = generic_implicit_MPI, use_RL = True)
    MPI.COMM_WORLD.Barrier()
    if rank ==0: print("############ MIN")
    MPI.COMM_WORLD.Barrier()
    run_simulation(spectral=True, ml=False, nprocs_space=n_space, sweeper_class = generic_implicit_MPI, use_RL = False)
    MPI.COMM_WORLD.Barrier()
    if rank ==0: print("############ MIN3")
    MPI.COMM_WORLD.Barrier()   
    run_simulation(spectral=True, ml=False, nprocs_space=n_space, sweeper_class = generic_implicit_MPI, use_RL = False, MIN3=True)
    MPI.COMM_WORLD.Barrier()  
    if rank ==0: print("############ LU")
    MPI.COMM_WORLD.Barrier()    
    run_simulation(spectral=True, ml=False, nprocs_space=n_space, sweeper_class = generic_implicit, use_RL = False)
    MPI.COMM_WORLD.Barrier()    





if __name__ == "__main__":
    main()
