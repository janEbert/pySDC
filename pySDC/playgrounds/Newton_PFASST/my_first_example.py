import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_fullyimplicit
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.playgrounds.Newton_PFASST.allinclusive_jacmatrix_nonMPI import allinclusive_jacmatrix_nonMPI
from pySDC.playgrounds.Newton_PFASST.pfasst_newton_output import output

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def setup(dt):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = dt
    level_params['nsweeps'] = [1, 1]

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 50

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['nvars'] = [128, 64]
    problem_params['eps'] = 0.04
    problem_params['inner_maxiter'] = 1
    problem_params['inner_tol'] = 1E-09
    problem_params['radius'] = 0.25

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3, 3]
    sweeper_params['QI'] = ['LU', 'LU']

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = allencahn_fullyimplicit
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh
    description['space_transfer_params'] = space_transfer_params

    return description, controller_params


def run_newton_pfasst(dt, Tend, num_procs):

    print('THIS IS NEWTON-PFASST....')

    description, controller_params = setup(dt=dt)

    controller_params['do_coarse'] = True

    # setup parameters "in time"
    t0 = 0.0

    nsolves_all = 0
    nsolves_step = 0
    nsolves_iter = 0
    
    
    global uk , ulast      
    while t0 < Tend:


        # instantiate the controller
        controller = allinclusive_jacmatrix_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                               description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        if t0==0:
            uinit = P.u_exact(t0)
            uk = np.kron(np.ones(controller.nsteps * controller.nnodes), uinit.values)
            controller.compute_rhs(uk, t0=t0)            
        else:
            uk = np.kron(np.ones(controller.nsteps * controller.nnodes), ulast)
            controller.compute_rhs(uk, t0, u0=ulast)


        print('  Initial residual: %8.6e' % np.linalg.norm(controller.rhs, np.inf))
    

        k = 0
        while np.linalg.norm(controller.rhs, np.inf) > description['level_params']['restol'] or k == 0:
            k += 1
            
            print('#### Setup t0 %4.2f, dt %4.2f, tend %4.2f' %
                (t0, dt, t0+dt*num_procs))
            
            if t0==0:
                ek, stats = controller.run(uk=uk, t0=t0, Tend=t0+dt*num_procs)
            else:
                ek, stats = controller.run(uk=uk, t0=t0, Tend=t0+dt*num_procs, u0=ulast)
            uk -= ek
            
            if t0==0:
                controller.compute_rhs(uk, t0)
            else:
                controller.compute_rhs(uk, t0, u0=ulast)

            print('t0 %4.2f, dt %4.2f, tend %4.2f,  Outer Iteration: %i -- number of inner solves: %i -- Newton residual: %8.6e' %
                (t0, dt, t0+dt*num_procs, k, controller.inner_solve_counter, np.linalg.norm(controller.rhs, np.inf)))
        
        end =  num_procs*128*3
        start = end-128
        ulast = uk[start:end]
        
        t0 += dt*num_procs

        # compute and print statistics
        nsolves_all = controller.inner_solve_counter
        nsolves_step = nsolves_all / num_procs
        nsolves_iter = nsolves_all / k
        print('  --> Number of outer iterations: %i' % k)
        print('  --> Number of inner solves (total/per iter/per step): %i / %4.2f / %4.2f' %
            (nsolves_all, nsolves_iter, nsolves_step))

    fname = 'data/AC_reference_newton_pfasst.npz'
    loaded = np.load(fname)
    uref = loaded['uend']
   
    end = num_procs*128*3
    start = end-128
    udiff = uk[start:end]-uref
    out   = max(udiff)
    print('  --> Difference between u and u_ref: %4.10f' %out )
    print()    

    print('THIS IS PFASST-NEWTON....')

    description, controller_params = setup(dt=dt)

    # remove this line to reduce the output of PFASST
    #controller_params['hook_class'] = output

    # setup parameters "in time"
    t0 = 0.0

    #num_procs = int((Tend - t0) / description['level_params']['dt'])

    controller = allinclusive_multigrid_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by variant (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # get maximum number of iterations
    niter = max([item[1] for item in iter_counts])

    # compute and print statistics
    nsolves_all = int(np.sum([S.levels[0].prob.inner_solve_counter for S in controller.MS]))
    nsolves_step = nsolves_all / num_procs
    nsolves_iter = nsolves_all / niter
    print('  --> Number of outer iterations: %i' % niter)
    print('  --> Number of inner solves (total/per iter/per step): %i / %4.2f / %4.2f' %
          (nsolves_all, nsolves_iter, nsolves_step))

    fname = 'data/AC_reference_newton_pfasst.npz'
    loaded = np.load(fname)
    uref = loaded['uend']
    
    ### find out how to convert mesh and float this is an ugly step :(
    fname = 'data/newton_pfasst.npz'
    np.savez_compressed(file=fname, uend=uend.values)
    
    fname = 'data/newton_pfasst.npz'
    loaded = np.load(fname)
    unew = loaded['uend']
    ###
    
    udiff = unew-uref
    out   = max(udiff)
    print('  --> Difference between u and u_ref: %4.10f' %out )

    print()    
    #fname = 'data/convergence_plot_dt'.format(1E-4) +'andbiggersteps.npz'
    #np.savez_compressed(file=fname, dt=dt, udiff=udiff)




def main():


    numbers = [0.2,0.1,0.05] 
    for dt in numbers:
        Tend = 0.8 
        run_newton_pfasst(dt=dt, Tend=Tend, num_procs=4)


if __name__ == "__main__":

    main()
