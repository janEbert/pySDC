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

from pySDC.playgrounds.Newton_PFASST.logistic_1D_FD_implicit_Jac import logistic_jac
from pySDC.implementations.problem_classes.logistic_1D_FD_implicit import logistic


#from pySDC.playgrounds.Newton_PFASST.sinGeneralizedFisher_1D_FD_implicit_Jac import generalized_fisher_jac
#from pySDC.implementations.problem_classes.sinGeneralizedFisher_1D_FD_implicit import generalized_fisher

from pySDC.helpers.stats_helper import filter_stats, sort_stats

dt =  1.0 #0.125/4. #0.0625 #0.015625 #0.03125 #0.015625 #0.03125 #0.0625
def setup(lambda0, elements, inexact):
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-13 #1e-13 #das war bei allen rechnungen
    level_params['dt'] = dt #0.05 #64. *1e-3 #8.*1e-3 #ref 64.*1E-06
    level_params['nsweeps'] = [1]

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 100
    
    #This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nvars'] = [1] #elements #[1023, 511] 
    problem_params['lambda0'] = lambda0 #1.0
    problem_params['newton_maxiter'] = inexact #1000
    #problem_params['outer_maxiter'] = 4 #1000
    problem_params['newton_tol'] = 10 #1e-10 #1e-14 #-1 #1e-12 #20#1E-12     
    problem_params['interval'] = (0, 2.0*np.pi)
    
    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3,1]
    sweeper_params['QI'] = ['LU']
    sweeper_params['spread'] = True

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    #space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['predict'] = True #False


    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = logistic #generalized_fisher
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
    while np.linalg.norm(controller.rhs, np.inf) > 1E-08 or k == 0: #bla bla
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
    

    description['problem_class'] = logistic_jac 
    description['base_transfer_class'] = linear_base_transfer
    description['sweeper_class'] = generic_implicit_rhs
    
    #chance outer and inner tolerance
    controller_params['outer_restol'] = description['level_params']['restol']
    description['step_params']['maxiter'] = description['problem_params']['newton_maxiter']
    description['level_params']['restol'] = description['problem_params']['newton_tol']

    t0 = 0.0


    controller = allinclusive_newton_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)


    
    
    
    
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)


    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)



    print("value ", uend[-1][-1].values )
    
    print("abweichung aw", np.linalg.norm(uend[-1][-1].values-uinit.values, np.inf))
    
    print("abweichung tend", np.linalg.norm(uend[-1][-1].values-P.u_exact(Tend).values))
    
   
    if 0 :
    
        fname = 'feinref.npz'
        np.savez_compressed(file=fname, uend=uend)
    
    nspace = 1 
    nnodes=3 
    nsteps=num_procs  
    jac = []

    for time in uend:
        for i in [-3,-2,-1]:      
            jac.append(controller.MS[0].levels[0].prob.params.lambda0 - controller.MS[0].levels[0].prob.params.lambda0 *2.0*time[i].values)
        
        
    A = spla.block_diag(jac).todense()

    Q = controller.MS[0].levels[0].sweep.coll.Qmat[1:, 1:]
    Qd = controller.MS[0].levels[0].sweep.QI[1:, 1:]


    E = np.zeros((nsteps, nsteps))
    np.fill_diagonal(E[1:, :], 1)

    N = np.zeros((nnodes, nnodes))
    N[:, -1] = 1


    Cp = np.eye(nsteps * nnodes * nspace) - dt * np.kron(np.eye(nsteps), np.kron(Q, np.eye(nspace))).dot(A) -np.kron(E, np.kron(N, np.eye(nspace)))
    Cp = np.array(Cp)


    P = np.eye(nsteps * nnodes * nspace) -  dt * np.kron(np.eye(nsteps), np.kron(Qd, np.eye(nspace))).dot(A)  - np.kron(E, np.kron(N, np.eye(nspace)))
    P = np.array(P)

    w, v = LA.eig( np.eye(nsteps * nnodes * nspace) - np.dot(inv(P), Cp))
    
    print(np.amax(np.linalg.norm(w)))
    

    f_nodes2 = controller.MS[0].levels[1].sweep.coll.nodes
    c_nodes2 = controller.MS[0].levels[0].sweep.coll.nodes
    
    nnodes_f2 = len(f_nodes2)
    nnodes_c2 = len(c_nodes2)
    

    tmat2 = np.zeros((nnodes_f2, nnodes_c2))


    for i2 in range(nnodes_f2):
        xi2 = f_nodes2[i2]
        for j2 in range(nnodes_c2):
            den2 = 1.0
            num2 = 1.0
            for k2 in range(nnodes_c2):
                if k2 == j2:
                    continue
                else:
                    den2 *= c_nodes2[j2] - c_nodes2[k2]
                    num2 *= xi2 - c_nodes2[k2]
            tmat2[i2, j2] = num2 / den2
            

    Cp_coarse = np.kron(tmat2, np.eye(nsteps)).dot(Cp)

    jac_p=[]
    for time in uend:
        for i in [-1]:     
            B_p= controller.MS[0].levels[0].prob.params.lambda0 - controller.MS[0].levels[0].prob.params.lambda0 *2.0*time[i].values 
            jac_p.append(B_p) 
        
          
    A_p = spla.block_diag(jac_p).todense()

    Qd_p = controller.MS[0].levels[1].sweep.QI[1:, 1:]


    N_p = np.zeros((1, 1))
    N_p[:, -1] = 1

    P_p = np.eye(nsteps * 1 * nspace) - \
             dt * np.kron(np.eye(nsteps), np.kron(Qd_p, np.eye(nspace))).dot(A_p)  \
             - np.kron(E, np.kron(N_p, np.eye(nspace)))
    P_p = np.array(P_p)

    P_p.dot(Cp_coarse)


    f_nodes = controller.MS[0].levels[0].sweep.coll.nodes
    c_nodes = controller.MS[0].levels[1].sweep.coll.nodes
    
    nnodes_f = len(f_nodes)
    nnodes_c = len(c_nodes)
    


    tmat = np.zeros((nnodes_f, nnodes_c))

    for i in range(nnodes_f):
        xi = f_nodes[i]
        for j in range(nnodes_c):
            den = 1.0
            num = 1.0
            for k in range(nnodes_c):
                if k == j:
                    continue
                else:
                    den *= c_nodes[j] - c_nodes[k]
                    num *= xi - c_nodes[k]
            tmat[i, j] = num / den



    w, v = LA.eig(               (np.eye(nsteps * nnodes * nspace) - np.kron(np.eye(nsteps) ,tmat).dot(inv(P_p).dot(Cp_coarse)) ).dot(np.eye(nsteps * nnodes * nspace) - np.dot(inv(P), Cp)))

    print(np.amax(np.linalg.norm(w)))
    





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
    #print("uinit ", uinit.values)
    #uinit.values[0]=0.88

    #uend, stats = run_with_pred(uk=uinit, u0=uinit, t0=t0, Tend=Tend)   
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    #print(len(uend))
    #uref  = P.u_exact(Tend)
    
    #print("number of inner solves")
    	      	    
    
    #fname = 'feinref.npz'
    #loaded = np.load(fname)
    #uref = loaded['uend']
    ##print(uend)
    print(uend[-1][-1].values)
    
    #print("vgl", uend[-1][-3:])
    print("abweichung aw", np.linalg.norm(uend[-1][-1].values-uinit.values, np.inf))
    
    print("abweichung tend", np.linalg.norm(uend[-1][-1].values-P.u_exact(Tend).values))
    
    #print(np.linalg.norm(uend[-1].values-uinit.values, np.inf))
    #print(np.linalg.norm(uend[-1].values-uref[-1].values, np.inf))

   
    if 1 :
    
        fname = 'feinref.npz'
        np.savez_compressed(file=fname, uend=uend)
    
    nspace = 1 #63
    nnodes=3 #4
    nsteps=num_procs  #4
    jac = []



    for time in uend:
        #for i in [1,2,3]:

        for i in [-3,-2,-1]:
            #print("############# time vgl ", i , time[-3:]) 
            #print("############# time i ", i , time[i])        
            B= controller.MS[0].levels[0].prob.params.lambda0 - controller.MS[0].levels[0].prob.params.lambda0 *2.0*time[i].values #1- 2*uend[-1].values 
            C=B 
            jac.append(C) #.append(C)
        
        
    #print("jac", jac)     
    A = spla.block_diag(jac).todense()
    #print("A", A)
    Q = controller.MS[0].levels[0].sweep.coll.Qmat[1:, 1:]
    Qd = controller.MS[0].levels[0].sweep.QI[1:, 1:]

    E = np.zeros((nsteps, nsteps))
    np.fill_diagonal(E[1:, :], 1)

    N = np.zeros((nnodes, nnodes))
    N[:, -1] = 1

    Cp=None
    #print(nsteps * nnodes * nspace)
    Cp = np.eye(nsteps * nnodes * nspace) 
    eil= np.kron(np.eye(nsteps), np.kron(Q, np.eye(nspace)))
    #print(Q.shape)
    #print(eil.shape)
    #print(A.shape)    
    Cp -= dt * eil.dot(A) 
    Cp -= np.kron(E, np.kron(N, np.eye(nspace)))
    Cp = np.array(Cp)


    P = np.eye(nsteps * nnodes * nspace) - \
             dt * np.kron(np.eye(nsteps), np.kron(Qd, np.eye(nspace))).dot(A)  - np.kron(E, np.kron(N, np.eye(nspace)))
    P = np.array(P)

    w, v = LA.eig( np.eye(nsteps * nnodes * nspace) - np.dot(inv(P), Cp))
    
    print(np.amax(np.linalg.norm(w)))
    

    f_nodes2 = controller.MS[0].levels[1].sweep.coll.nodes
    c_nodes2 = controller.MS[0].levels[0].sweep.coll.nodes
    
    nnodes_f2 = len(f_nodes2)
    nnodes_c2 = len(c_nodes2)
    

    tmat2 = np.zeros((nnodes_f2, nnodes_c2))


    for i2 in range(nnodes_f2):
        xi2 = f_nodes2[i2]
        for j2 in range(nnodes_c2):
            den2 = 1.0
            num2 = 1.0
            for k2 in range(nnodes_c2):
                if k2 == j2:
                    continue
                else:
                    den2 *= c_nodes2[j2] - c_nodes2[k2]
                    num2 *= xi2 - c_nodes2[k2]
            tmat2[i2, j2] = num2 / den2
            
    #print(np.kron(tmat2, np.eye(nsteps)))         

    #Cp_coarse_ = []    
    #for i in [2,5,8,11]:
    #    Cp_coarse_.append(Cp[i,i])
    #Cp_coarse = spla.block_diag(Cp_coarse_).todense()    

    Cp_coarse = np.kron(tmat2, np.eye(nsteps)).dot(Cp)

    jac_p=[]
    for time in uend:
        for i in [-1]:     
            B_p= controller.MS[0].levels[0].prob.params.lambda0 - controller.MS[0].levels[0].prob.params.lambda0 *2.0*time[i].values #1- 2*uend[-1].values 
            jac_p.append(B_p) 
        
          
    A_p = spla.block_diag(jac_p).todense()

    Qd_p = controller.MS[0].levels[1].sweep.QI[1:, 1:]
    #print(E)

    N_p = np.zeros((1, 1))
    N_p[:, -1] = 1

    P_p = np.eye(nsteps * 1 * nspace) - \
             dt * np.kron(np.eye(nsteps), np.kron(Qd_p, np.eye(nspace))).dot(A_p)  \
             - np.kron(E, np.kron(N_p, np.eye(nspace)))
    P_p = np.array(P_p)

    P_p.dot(Cp_coarse)


    f_nodes = controller.MS[0].levels[0].sweep.coll.nodes
    c_nodes = controller.MS[0].levels[1].sweep.coll.nodes
    
    nnodes_f = len(f_nodes)
    nnodes_c = len(c_nodes)
    
    #print(len(f_nodes), len(c_nodes))

    tmat = np.zeros((nnodes_f, nnodes_c))

    for i in range(nnodes_f):
        xi = f_nodes[i]
        for j in range(nnodes_c):
            den = 1.0
            num = 1.0
            for k in range(nnodes_c):
                if k == j:
                    continue
                else:
                    den *= c_nodes[j] - c_nodes[k]
                    num *= xi - c_nodes[k]
            tmat[i, j] = num / den

    #print(tmat)
    
    #print( np.kron(np.eye(nsteps) ,tmat).dot(inv(P_p).dot(Cp_coarse))  )

    w, v = LA.eig(               (np.eye(nsteps * nnodes * nspace) - np.kron(np.eye(nsteps) ,tmat).dot(inv(P_p).dot(Cp_coarse)) ).dot(np.eye(nsteps * nnodes * nspace) - np.dot(inv(P), Cp)))

    print(np.amax(np.linalg.norm(w)))
    
    #print(inv(P_p).dot(Cp_coarse))
    
    #print( np.kron(np.eye(nsteps) ,tmat))
    
    #print(tmat2)


def main():



    num_procs= 4 #int(Tend/dt)   
    
    Tend = num_procs * dt #0.8     
    print(num_procs)

    lambda0=1.
    elements= [1]     

    
    for maxiters in [4000]: 
        print("maxiter", maxiters)  
        description, controller_params = setup(lambda0=lambda0, elements=elements, inexact=maxiters)
        #print("!!!!!!!!!!!!!!!!!!!! ",maxiters) 
        #run_newton_pfasst_matrixfree(num_procs=num_procs, description=description, controller_params=controller_params, Tend=Tend)        
        run_pfasst_newton(num_procs=num_procs, description=description, controller_params=controller_params, Tend=Tend)
        #run_newton_pfasst_matrix(description=description, controller_params=controller_params, Tend=Tend) 
    #sys.exit()
    
if __name__ == "__main__":

    main()
