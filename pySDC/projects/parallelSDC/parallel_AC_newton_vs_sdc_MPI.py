import os
import pickle
import numpy as np
import subprocess

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.parallelSDC.ErrReductionHook import err_reduction_hook
#from pySDC.projects.parallelSDC.GeneralizedFisher_1D_FD_implicit_Jac import generalized_fisher_jac
from pySDC.projects.parallelSDC.AC_2D_FD_implicit_Jac import AC_jac
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel_MPI import linearized_implicit_fixed_parallel_MPI
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel_prec_MPI import linearized_implicit_fixed_parallel_prec_MPI
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel import linearized_implicit_fixed_parallel
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel_prec import linearized_implicit_fixed_parallel_prec

from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
import matplotlib.pyplot as plt


from pySDC.projects.parallelSDC.generic_implicit_MPI import generic_implicit_MPI
from mpi4py import MPI

def run(sweeper_list):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-8

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 20

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['nvars'] = [(256,256)] #, (128, 128)]
    problem_params['eps'] = [0.04]
    problem_params['newton_maxiter'] = 1#100
    problem_params['newton_tol'] = 1E-09
    problem_params['lin_tol'] = 1E-10
    problem_params['lin_maxiter'] = 200
    problem_params['radius'] = 0.25


    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'
    sweeper_params['fixed_time_in_jacobian'] = 0
    sweeper_params['comm'] = MPI.COMM_WORLD
        
    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = err_reduction_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = AC_jac #generalized_fisher_jac
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh

    #assert MPI.COMM_WORLD.Get_size() == sweeper_params['num_nodes']
    #print(MPI.COMM_WORLD.Get_size())
    
    # setup parameters "in time"
    t0 = 0
    Tend = 0.008 #32


    dt_list = [Tend / 2 ** i for i in range(1, 4)]

    #print(dt_list)
    results = dict()
    results['sweeper_list'] = [sweeper.__name__ for sweeper in sweeper_list]
    results['dt_list'] = dt_list



    # loop over the different sweepers and check results
    for sweeper in sweeper_list:
        description['sweeper_class'] = sweeper
        error_reduction = []
        for dt in [1e-3]:#dt_list:
            print('Working with sweeper %s and dt = %s...' % (sweeper.__name__, dt))

            level_params['dt'] = dt
            description['level_params'] = level_params

            # instantiate the controller

            controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)
            #uext = P.u_exact(Tend)
            
            # call main function to get things done...
            t1 = MPI.Wtime()
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
            t2 = MPI.Wtime()             
            print( "Elapsed time is ")
            print( t2 - t1 );
            #plt.plot(uext.values)
            #plt.savefig('oo3.png')
    
            fname = 'ref_new.npz'
            #np.savez_compressed(file=fname, uend=uend.values)
            loaded = np.load(fname)
            uref = loaded['uend']
    
            print("-----------------------------------------------------")
            print("Fehler ")
            print( np.linalg.norm(uinit.values-uend.values, np.inf))
            print(np.linalg.norm(uref-uend.values, np.inf))
            
            # filter statistics
            #filtered_stats = filter_stats(stats, type='error_pre_iteration')
            #print(filtered_stats)
            #error_pre = sort_stats(filtered_stats, sortby='iter')[0][1]

            #filtered_stats = filter_stats(stats, type='error_post_iteration')
            #error_post = sort_stats(filtered_stats, sortby='iter')[0][1]

            #error_reduction.append(error_post / error_pre)

            #print('error and reduction rate at time %s: %6.4e , %6.4e -- %6.4e' % (Tend, error_post, error_pre,error_reduction[-1]))
            
    	    # filter statistics by variant (number of iterations)
            filtered_stats = filter_stats(stats, type='niter')

            # convert filtered statistics to list of iterations count, sorted by process
            iter_counts = sort_stats(filtered_stats, sortby='time')

            # compute and print statistics
            niters = np.array([item[1] for item in iter_counts])
            print(niters)
            print(P.newton_itercount)
            print("iterationnummbers")
            for pp in controller.MS:
                print(pp.levels[0].prob.newton_itercount)
            print("-----------------------------------------------------")

        results[sweeper.__name__] = error_reduction
        print()

    file = open('data/error_reduction_data.pkl', 'wb')
    pickle.dump(results, file)
    file.close()


def plot_graphs(cwd=''):
    """
    Helper function to plot graphs of initial and final values

    Args:
        cwd (str): current working directory
    """
    plt_helper.mpl.style.use('classic')

    file = open(cwd + 'data/error_reduction_data.pkl', 'rb')
    results = pickle.load(file)

    sweeper_list = results['sweeper_list']
    dt_list = results['dt_list']

    color_list = ['red', 'blue', 'green']
    marker_list = ['o', 's', 'd']
    label_list = []
    for sweeper in sweeper_list:
        if sweeper == 'generic_implicit':
            label_list.append('SDC')
        elif sweeper == 'linearized_implicit_fixed_parallel':
            label_list.append('Simplified Newton')
        elif sweeper == 'linearized_implicit_fixed_parallel_prec':
            label_list.append('Inexact Newton')

    setups = zip(sweeper_list, color_list, marker_list, label_list)

    plt_helper.setup_mpl()

    plt_helper.newfig(textwidth=238.96, scale=0.89)

    for sweeper, color, marker, label in setups:
        plt_helper.plt.loglog(dt_list, results[sweeper], lw=1, ls='-', color=color, marker=marker,
                              markeredgecolor='k', label=label)

    plt_helper.plt.loglog(dt_list, [dt * 2 for dt in dt_list], lw=0.5, ls='--', color='k', label='linear')
    plt_helper.plt.loglog(dt_list, [dt * dt / dt_list[0] * 2 for dt in dt_list], lw=0.5, ls='-.', color='k',
                          label='quadratic')

    plt_helper.plt.xlabel('dt')
    plt_helper.plt.ylabel('error reduction')
    plt_helper.plt.grid()

    # ax.set_xticks(dt_list, dt_list)
    plt_helper.plt.xticks(dt_list, dt_list)

    plt_helper.plt.legend(loc=1, ncol=1)

    plt_helper.plt.gca().invert_xaxis()
    plt_helper.plt.xlim([dt_list[0] * 1.1, dt_list[-1] / 1.1])
    plt_helper.plt.ylim([4E-03, 1E0])

    # save plot, beautify
    fname = 'data/parallelSDC_fisher_newton'
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'
    
def main():
    """
    Main driver

    """

    # [linearized_implicit_fixed_parallel_prec, linearized_implicit_fixed_parallel]
    #run_variant(variant='sl_serial')
    #print()
    #run_variant(variant='ml_serial')
    #print()
    
    #run([generic_implicit])
    
    run([linearized_implicit_fixed_parallel_prec_MPI])
    
    #run([linearized_implicit_fixed_parallel])    

    #cmd = "mpirun -np 3 xterm -hold -e python -c \"from pySDC.projects.parallelSDC.ACnewton_vs_sdc_MPI import *; " \
    #      "run([linearized_implicit_fixed_parallel_MPI]);\"" #linearized_implicit_fixed_parallel_prec_MPI, 
    #p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    #p.wait()
    #(output, err) = p.communicate()
    #print(output)




if __name__ == "__main__":
    main()
    #plot_graphs()
