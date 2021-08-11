from argparse import ArgumentParser
import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.projects.parallelSDC.generic_imex_MPI import generic_imex_MPI
from pySDC.projects.parallelSDC.generic_implicit_MPI import generic_implicit_MPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.AdvectionDiffusion_MPIFFT import advectiondiffusion_imex
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft

from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import numpy as np

from .model_utils import load_model

num_nodes = 3

mins   = [0,0,0,0,0,0]
means  = [0,0,0,0,0,0]
maxima = [0,0,0,0,0,0]


def build_model(M, train):
    model_arch = [
        ('Dense', (128,)),
        ('Dropout', ()),
        ('Relu',),
        ('Dense', (256,)),
        ('Relu',),
        ('Dense', (128,)),
        ('Dropout', ()),
        ('Relu',),
        ('Dense', (M,)),
    ]

    (model_init, model_apply) = _from_model_arch(model_arch, train=train)

    return (model_init, model_apply, model_arch)


def build_opt(lr, params):

    lr = optax.cosine_onecycle_schedule(30000, 2 * lr, 0.3, 2e7)

    (opt_init, opt_update, opt_get_params) = optimizers.adam(lr)
    opt_state = opt_init(params)
    return (opt_state, opt_update, opt_get_params)


def run_simulation(nprocs_space=None, sweeper_class=None, use_RL = None, index = None, imex=None, QI=None):


    #load the RL model
    rng_key = jax.random.PRNGKey(0)
    rng_key, subkey = jax.random.split(rng_key)
    model_path = "models/dp_model_2021-07-28T11-40-55.669033.npy" 

    params, model = load_model(model_path)

    #setup communicators
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
    level_params['restol'] = 1E-15
    level_params['dt'] = 0.01 
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [num_nodes]

    if sweeper_class == 'generic_imex_MPI':
        assert sweeper_params['num_nodes'][0] == time_size, 'Need %s processors in time' % sweeper_params['num_nodes']   

    sweeper_params['initial_guess'] = 'zero'


    sweeper_params['comm'] = time_comm 

    # initialize problem parameters
    problem_params = dict()

    problem_params['nvars'] = [(32,32,32)]
    problem_params['spectral'] = True
    problem_params['comm'] = space_comm
    problem_params['time_comm'] = time_comm
    problem_params['model'] = model
    problem_params['model_params'] = params
    problem_params['subkey'] = subkey
    problem_params['rng_key'] = rng_key
    problem_params['dt'] = level_params['dt']    
    problem_params['imex'] = imex
    problem_params['use_RL'] = use_RL
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30 if rank == 0 else 99
    # controller_params['predict_type'] = 'fine_only'

    # fill description dictionary for easy step instantiation

    sweeper_params['QI'] = [QI] 
    sweeper_params['QE'] = ['PIC'] 
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    description['problem_class'] = advectiondiffusion_imex
    description['sweeper_class'] = sweeper_class #generic_imex_MPI # imex_1st_order

    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters





    # set time parameters
    t0 = 0.0
    Tend = 0.1



    # instantiate controller
    MPI.COMM_WORLD.Barrier()    
    wt = MPI.Wtime()

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    MPI.COMM_WORLD.Barrier()    
    run_time = MPI.Wtime()
    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    MPI.COMM_WORLD.Barrier()    
    wt = MPI.Wtime() - wt
    wt2 = MPI.Wtime() - run_time

    uex = P.u_exact(Tend)
    #print("shape", uex.shape)
    err = abs(uex - uend)
    abw = abs(uend-uinit)




    if rank == 0:
        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        niters = np.array([item[1] for item in iter_counts])
        out = f'   Min/Mean/Max number of iterations: ' \
              f'{np.min(niters):4.2f} / {np.mean(niters):4.2f} / {np.max(niters):4.2f}'
        #f.write(out + '\n')
        print(out)
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        #f.write(out + '\n')
        print(out)

        out = f'Error: {err:6.4e}'
        #f.write(out + '\n')
        print(out)

        timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
        out = f'Time to solution: {timing[0][1]:6.4f} sec.'
        #f.write(out + '\n')
        print(out)

        print('Mean number of iterations %s' %( np.mean(niters)))

        print("TIME", wt, wt2 )

        if index >=0:
            mins[index] = np.min(niters) 
            means[index] = np.mean(niters) 
            maxima[index] = np.max(niters)



def plot():
    print("mins = ", mins)
    print("means = ", means)
    print("maxima = ", maxima)
    labels = ['RL+0', 'RL+RL', 'Opt+0', 'Opt+Opt', 'LU+0', 'LU+LU']


    x = np.arange(len(labels))  # the label locations
    width = 0.6  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width/2, mins, width/2, label='Min', color='green')
    rects2 = ax.bar(x + width/2, maxima, width/2, label='Mean', color='red')
    rects3 = ax.bar(x, means, width/2, label='Max', color='orange')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('#iterations', fontsize=20)
    ax.yaxis.set_tick_params(labelsize=12)
    #ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.legend(fontsize=12)
    #ax.tick_params(axis='both', which='minor', labelsize=20)


    #ax.bar_label(rects1, padding=0, fontsize=12)
    #ax.bar_label(rects2, padding=0, fontsize=12)
    #ax.bar_label(rects3, padding=0, fontsize=12)

    fig.tight_layout()

    plt.show()


def main():
    """
    Little helper routine to run the whole thing

    Note: This can also be run with "mpirun -np 2 python B_pySDC_with_mpi4pyfft.py"
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()


    n_space = int(world_size/num_nodes)


    if True:

        MPI.COMM_WORLD.Barrier()
        if rank ==0: print("############ RL+RL")
        MPI.COMM_WORLD.Barrier()
        run_simulation(nprocs_space=n_space, sweeper_class = generic_implicit_MPI, use_RL = True,  index=1, imex=False, QI = 'RL' )

        MPI.COMM_WORLD.Barrier()    
        if rank ==0: print("############ RL+0")
        MPI.COMM_WORLD.Barrier()
        run_simulation(nprocs_space=n_space, sweeper_class = generic_imex_MPI, use_RL = True,  index=0, imex=True, QI = 'RL')

        MPI.COMM_WORLD.Barrier()
        if rank ==0: print("############ MIN+0")
        MPI.COMM_WORLD.Barrier()
        run_simulation(nprocs_space=n_space, sweeper_class = generic_imex_MPI, use_RL = False,  index=2, imex=True, QI = 'MIN')
        MPI.COMM_WORLD.Barrier()
        if rank ==0: print("############ MIN+MIN")
        MPI.COMM_WORLD.Barrier()   
        run_simulation(nprocs_space=n_space, sweeper_class = generic_implicit_MPI, use_RL = False,  index=3, imex=False, QI = 'MIN')
        MPI.COMM_WORLD.Barrier()  

    if rank ==0: print("############ LU+0")
    MPI.COMM_WORLD.Barrier()    
    run_simulation(nprocs_space=world_size, sweeper_class = imex_1st_order, use_RL = False,  index=4, imex=True, QI = 'LU')
    MPI.COMM_WORLD.Barrier()   
    if rank ==0: print("############ LU+LU")
    MPI.COMM_WORLD.Barrier()    
    run_simulation(nprocs_space=world_size, sweeper_class = generic_implicit, use_RL = False, index=5, imex=False, QI = 'LU')
    MPI.COMM_WORLD.Barrier()   
    if rank ==0: plot()





if __name__ == "__main__":
    main()
