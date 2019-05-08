import pySDC.helpers.plot_helper as plt_helper
import numpy as np
import pickle
import os

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.playgrounds.Newton_PFASST.allinclusive_jacmatrix_nonMPI import allinclusive_jacmatrix_nonMPI
from pySDC.playgrounds.Newton_PFASST.GeneralizedFisher_1D_FD_implicit_Jac import generalized_fisher_jac

from pySDC.helpers.stats_helper import filter_stats, sort_stats


from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher

def main():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.01

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 10

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 1.0
    problem_params['nvars'] = [127, 63]
    problem_params['lambda0'] = 1.0
    problem_params['newton_maxiter'] = 50
    problem_params['newton_tol'] = 1E-10
    problem_params['interval'] = (-5, 5)
    problem_params['radius'] = 0.25
    problem_params['eps'] = 0.2

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False
    # controller_params['hook_class'] = err_reduction_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = generalized_fisher #_jac
    description['problem_params'] = problem_params
    description['dtype_u'] = mesh
    description['dtype_f'] = mesh
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh
    description['space_transfer_params'] = space_transfer_params

    # setup parameters "in time"
    t0 = 0.0
    Tend = 0.2

    num_procs = int((Tend - t0) / level_params['dt'])

    controller_pfasst = allinclusive_multigrid_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)
    controller_pfasst.params.predict = False # True
    #for S in controller_pfasst.MS:
        #S.params.maxiter = 0

    # get initial values on finest level
    P = controller_pfasst.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uref = P.u_exact(Tend)
    
    uend, stats = controller_pfasst.run(u0=uinit, t0=t0, Tend=Tend)
    
    
    print(uend.values)
    print(np.linalg.norm(uend.values-uinit.values, np.inf))
    print(np.linalg.norm(uend.values-uref.values, np.inf))








if __name__ == "__main__":
    main()

