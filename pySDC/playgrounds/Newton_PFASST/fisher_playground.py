import numpy as np
import time

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.playgrounds.Newton_PFASST.allinclusive_jacmatrix_nonMPI import allinclusive_jacmatrix_nonMPI
from pySDC.playgrounds.Newton_PFASST.pfasst_newton_output import output

from pySDC.playgrounds.Newton_PFASST.linear_pfasst.LinearBaseTransfer import linear_base_transfer
from pySDC.playgrounds.Newton_PFASST.linear_pfasst.allinclusive_newton_nonMPI import allinclusive_newton_nonMPI
from pySDC.playgrounds.Newton_PFASST.linear_pfasst.generic_implicit_rhs import generic_implicit_rhs
#from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

#from pySDC.playgrounds.Newton_PFASST.linear_pfasst.AllenCahn_1D_FD_jac import allencahn_fullyimplicit, allencahn_fullyimplicit_jac
from pySDC.playgrounds.Newton_PFASST.GeneralizedFisher_1D_FD_implicit_Jac import generalized_fisher_jac
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_FD_implicit import generalized_fisher

from pySDC.helpers.stats_helper import filter_stats, sort_stats


def setup():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-07
    level_params['dt'] = 1E-03
    level_params['nsweeps'] = [1]

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 50
    
    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 1.0
    problem_params['nvars'] = [1023] #[127] #, 63]
    problem_params['lambda0'] = 5.0
    problem_params['newton_maxiter'] = 50
    problem_params['newton_tol'] = 1E-10
    problem_params['lin_tol'] = 1E-08    
    problem_params['lin_maxiter'] = 1000    
    problem_params['interval'] = (-200, 200)
    #problem_params['radius'] = 0.25
    #problem_params['eps'] = 0.2
    
    

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']#, 'LU']
    sweeper_params['spread'] = False

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    #space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = False


    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = generalized_fisher #allencahn_fullyimplicit
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




def run_newton_pfasst_matrixfree(num_procs, Tend=None):

    print('THIS IS MATRIX-FREE NEWTON-PFASST....')

    description, controller_params = setup()

    description['problem_class'] = generalized_fisher_jac #allencahn_fullyimplicit_jac
    description['base_transfer_class'] = linear_base_transfer
    description['sweeper_class'] = generic_implicit_rhs
    controller_params['outer_restol'] = description['level_params']['restol']
    description['step_params']['maxiter'] = description['problem_params']['newton_maxiter']
    description['level_params']['restol'] = description['problem_params']['newton_tol']

    # setup parameters "in time"
    t0 = 0.0

    # instantiate the controller
    controller = allinclusive_newton_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uref  = P.u_exact(Tend)


    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    print("anfang")
    print(np.linalg.norm(uend.values-uinit.values, np.inf))
    print("ende")
    print(np.linalg.norm(uend.values-uref.values, np.inf))
    



def run_pfasst_newton(num_procs, Tend=None):

    print('THIS IS PFASST-NEWTON....')

    description, controller_params = setup()

    # remove this line to reduce the output of PFASST
    # controller_params['hook_class'] = output

    # setup parameters "in time"
    t0 = 0.0
    controller        = allinclusive_multigrid_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)
    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uref  = P.u_exact(Tend)
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    print("gelaufen")
    print(np.linalg.norm(uend.values-uinit.values, np.inf))
    print(np.linalg.norm(uend.values-uref.values, np.inf))




def main():

    # Setup can run until 0.032 = 32 * 0.001, so the factor gives the number of time-steps.
    num_procs = 1# 4
    Tend = 1e-3 #0.032

    run_pfasst_newton(num_procs=num_procs, Tend=Tend)
    run_newton_pfasst_matrixfree(num_procs=num_procs, Tend=Tend)


    #run_newton_pfasst_matrix(Tend=Tend)

if __name__ == "__main__":

    main()
