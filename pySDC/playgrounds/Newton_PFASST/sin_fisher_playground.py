import numpy as np
import time

import sys

import scipy.sparse as sp
import scipy.sparse as spla
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv

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
from pySDC.playgrounds.Newton_PFASST.sinGeneralizedFisher_1D_FD_implicit_Jac import generalized_fisher_jac
from pySDC.implementations.problem_classes.sinGeneralizedFisher_1D_FD_implicit import generalized_fisher

from pySDC.helpers.stats_helper import filter_stats, sort_stats

dt = 1.*1e-3  #reference 1e-6
def setup(nu, lambda0, elements, inexact):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-12 #1E-10 
    level_params['dt'] = dt #0.05 #64. *1e-3 #8.*1e-3 #ref 64.*1E-06
    level_params['nsweeps'] = [1]

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 1000
    
    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = nu
    problem_params['nvars'] = elements #[1023, 511] 
    problem_params['lambda0'] = lambda0 #1.0
    problem_params['newton_maxiter'] = inexact #1000
    problem_params['newton_tol'] = 1e-14 #1e-12 #20#1E-12     
    problem_params['interval'] = (0, 2.0*np.pi)
    
    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']
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
    description['problem_class'] = generalized_fisher
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

def run_newton_pfasst_matrix(description, controller_params, Tend=None):

    #description, controller_params = setup()

    # setup parameters "in time"
    t0 = 0.0
    # WARNING: this cannot do rolling! Needs to be implemented
    num_procs = int((Tend - t0) / description['level_params']['dt'])

    # instantiate the controller
    controller = allinclusive_jacmatrix_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uk = np.kron(np.ones(controller.nsteps * controller.nnodes), uinit.values)

    controller.compute_rhs(uk, t0)
    print(np.linalg.norm(controller.rhs, np.inf))
    k = 0
    while np.linalg.norm(controller.rhs, np.inf) > 1E-08 or k == 0:
        k += 1
        ek, stats = controller.run(uk=uk, t0=t0, Tend=Tend)
        uk -= ek
        controller.compute_rhs(uk, t0)

        print(k, controller.inner_solve_counter, np.linalg.norm(controller.rhs, np.inf), np.linalg.norm(ek, np.inf))

    nsolves_all = controller.inner_solve_counter
    nsolves_step = nsolves_all / num_procs
    print(nsolves_all, nsolves_step)


def run_newton_pfasst_matrixfree(num_procs, description, controller_params, Tend=None):

    print('')
    print('##### THIS IS NEWTON-PFASST ##########################################################')
    print('')
    

    description['problem_class'] = generalized_fisher_jac 
    description['base_transfer_class'] = linear_base_transfer
    description['sweeper_class'] = generic_implicit_rhs
    
    #chance outer and inner tolerance
    controller_params['outer_restol'] = description['level_params']['restol']
    description['step_params']['maxiter'] = description['problem_params']['newton_maxiter']
    description['level_params']['restol'] = description['problem_params']['newton_tol']

    #controller_params['hook_class'] = output
    
    # setup parameters "in time"
    t0 = 0.0

    # instantiate the controller
    controller = allinclusive_newton_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    
    
    
    
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    #uref  = P.u_exact(Tend)
    
    fname = 'ref.npz'
    loaded = np.load(fname)
    uref = loaded['uend']

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)


    print("number of inner solves")

    #max = 0
    #for i in controller.MS:
#	if (i.levels[0].prob.inner_solve_counter > max):
#            max = i.levels[0].prob.inner_solve_counter
#    print(max)

    print("anfang")
    print(np.linalg.norm(uend.values-uinit.values, np.inf))
    print("ende")
    print(np.linalg.norm(uend.values-uref, np.inf))
    



def run_pfasst_newton(num_procs, description, controller_params, Tend=None):

    print('')
    print('##### THIS IS PFASST-NEWTON ##########################################################')
    print('')


    # remove this line to reduce the output of PFASST
    controller_params['hook_class'] = output

    # setup parameters "in time"
    t0 = 0.0
    controller        = allinclusive_multigrid_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)
    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    print(len(uend))
    #uref  = P.u_exact(Tend)
    
    print("number of inner solves")
    
    max = 0
    #for i in controller.MS:
#	if i.levels[0].prob.inner_solve_counter > max :
#	    max = i.levels[0].prob.inner_solve_counter
    print(max)	      	    
    
    fname = 'feinref.npz'
    loaded = np.load(fname)
    uref = loaded['uend']
    
    print("gelaufen")
    #print(np.linalg.norm(uend[-1].values-uinit.values, np.inf))
    #print(np.linalg.norm(uend[-1].values-uref, np.inf))
    #print(np.linalg.norm(uend[-1].values-uinit.values, np.inf))
    #print(np.linalg.norm(uend[-1].values-uref[-1].values, np.inf))

   
    if 1 :
    
        fname = 'feinref.npz'
        np.savez_compressed(file=fname, uend=uend)
    
    nspace = 1 #63
    nnodes=4
    nsteps=16
    jac = []
    #for S in controller.MS:
    #    L = S.levels[0]
    #    L = controller.MS[-1].levels[0]  # WARNING: this is simplified Newton to decouple space from time and nodes
    #    P = L.prob
    #    for m in range(1, nnodes + 1):
    #        jac.append(P.eval_jacobian(L.u[-1])) # WARNING: this is simplified Newton

    for time in uend:
        #A = controller.MS[0].levels[0].prob.A#[1:-1, 1:-1] 
        print("uend",uend)
        B= sp.diags(controller.MS[0].levels[0].prob.params.lambda0 - controller.MS[0].levels[0].prob.params.lambda0 *2.0*uend[-1].values, offsets=0) #sp.diags(controller.MS[0].levels[0].prob.params.lambda0 ** 2 - controller.MS[0].levels[0].prob.params.lambda0 ** 2 * (controller.MS[0].levels[0].prob.params.nu + 1) * uend[-1].values ** controller.MS[0].levels[0].prob.params.nu, offsets=0)    
        #print(A)
        #print(B)
        C=B #+A
        for nn in [1,2,3,4]:
            jac.append(C)
        
        
         
    A = spla.block_diag(jac).todense()
    Q = controller.MS[0].levels[0].sweep.coll.Qmat#[1:, 1:]
    Qd = controller.MS[0].levels[0].sweep.QI#[1:, 1:]

    E = np.zeros((nsteps, nsteps))
    np.fill_diagonal(E[1:, :], 1)

    N = np.zeros((nnodes, nnodes))
    N[:, -1] = 1

    Cp=None
    print(nsteps * nnodes * nspace)
    Cp = np.eye(nsteps * nnodes * nspace) 
    eil= np.kron(np.eye(nsteps), np.kron(Q, np.eye(nspace)))
    print(eil.shape)
    print(A.shape)    
    Cp -= dt * eil.dot(A) 
    Cp -= np.kron(E, np.kron(N, np.eye(nspace)))
    Cp = np.array(Cp)

    # N = np.eye(self.nnodes)
    # E[0,-1] = self.dt
    # Qd = Q

    P = np.eye(nsteps * nnodes * nspace) - \
             dt * np.kron(np.eye(nsteps), np.kron(Qd, np.eye(nspace))).dot(A)  - np.kron(E, np.kron(N, np.eye(nspace)))
    P = np.array(P)

    w, v = LA.eig( np.eye(nsteps * nnodes * nspace) - np.dot(inv(P), C))
    
    print(w)


def main():

    # Setup can run until 0.032 = 32 * 0.001, so the factor gives the number of time-steps.

    Tend = 16.*1e-3 #1 #1024.*1e-3
    
    
    #maxiters=1
    nu=1. 
    lambda0=1.
    #elements= [511,255] 
    elements= [3] #[63,31]     
    num_procs=8
    
    for maxiters in [16]: #1,2,4,8,16,32]:    
        description, controller_params = setup(nu=nu, lambda0=lambda0, elements=elements, inexact=maxiters)
        print("!!!!!!!!!!!!!!!!!!!! ",maxiters) 
        #run_newton_pfasst_matrix(description=description, controller_params=controller_params, Tend=Tend)            
        run_pfasst_newton(num_procs=num_procs, description=description, controller_params=controller_params, Tend=Tend)
        #run_newton_pfasst_matrixfree(num_procs=num_procs, description=description, controller_params=controller_params, Tend=Tend)    
    
    sys.exit()
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    for algo in [1]:#,0]:
        for maxiters in [1]:#1000,10,4,1]:
            for nu in [1]:#,2,3]:
                for lambda0 in [1]:#1e-8, 1e-2, 1.0, 5.0]:
                    for elements in [[1023, 511]]:#[[1023], [1023, 511]]:
                        if len(elements) > 1:
                            procs0[2]
                            #procs = [1,2,4,8]
                        else:
                            procs = [1]
                        for num_procs in procs:    


                    
                            print("algo ",algo," maxiters ", maxiters, "nu ", nu, " lambda ", lambda0, " elements ", elements, " processor ", num_procs)
                            description, controller_params = setup(nu=nu, lambda0=lambda0, elements=elements, inexact=maxiters)

                            if algo: 
                                run_pfasst_newton(num_procs=num_procs, description=description, controller_params=controller_params, Tend=Tend)
                            else:
                                run_newton_pfasst_matrixfree(num_procs=num_procs, description=description, controller_params=controller_params, Tend=Tend)

                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

if __name__ == "__main__":

    main()
