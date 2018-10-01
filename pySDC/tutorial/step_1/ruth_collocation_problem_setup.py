import numpy as np
import scipy.sparse as sp

from scipy import linalg
from numpy.linalg import inv

from mpi4py import MPI

from decimal import *
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.datatype_classes.mesh import mesh

from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d

from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_fullyimplicit

def main():
    """
    A simple test program to create and solve a collocation problem directly
    """
    tend = 1. 
    dt = 0.01  

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1 #0.0001  # diffusion coefficient
    problem_params['eps'] = 1. #0.04
    problem_params['lambda0'] =  1. #/ 0.04
    
    problem_params['nvars'] = 2048  # number of degrees of freedom
    problem_params['nsteps'] = int(tend/dt) # number of degrees of freedom

    problem_params['inner_maxiter'] = 1
    problem_params['inner_tol'] = 1E-09
    problem_params['radius'] = 0.25

    # instantiate problem
    prob = allencahn_fullyimplicit(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    # instantiate collocation class, relative to the time interval [0,1]
    coll = CollGaussRadau_Right(num_nodes=3, tleft=0, tright=1)

    # solve collocation problem
    err = solve_collocation_problem(prob=prob, coll=coll, dt=dt, tend=tend)
    #f = open('step_1_C_out.txt', 'w')
    out = 'Error of the collocation problem: %8.6e' % err
    #f.write(out + '\n')
    print(out)
    #f.close()
    #assert err <= 4E-04, "ERROR: did not get collocation error as expected, got %s" % err

def eval_f(prob, coll, u):
    f=sp.kron(sp.eye(prob.params.nsteps * coll.num_nodes),prob.A).dot(u) +1.0/prob.params.eps**2*u*(1.0-u**prob.params.nu)
    return f

def solve_collocation_problem(prob, coll, dt, tend):


    #getcontext().prec = 1028

    # shrink collocation matrix: first line and column deals with initial value, not needed here
    Q = coll.Qmat[1:, 1:]

    #setup other matrices
    E = np.zeros((prob.params.nsteps, prob.params.nsteps))
    ### E[0,prob.params.nsteps-1] = 1e-6 #controller.dt *0.1
    np.fill_diagonal(E[1:, :], 1)

    E_diag = E
    #V = sp.eye(prob.params.nsteps)
    #u, V = linalg.eig(E)

    N = np.zeros((coll.num_nodes, coll.num_nodes))
    N[:, -1] = 1
    
    #E = np.dot(V,np.dot(np.diag(u), linalg.inv(V)))
    #E_diag = np.diag(u)
    
    #print(Eneu)
    #print np.real_if_close(Eneu)



    # get initial value at t0 = 0
    u0 = prob.u_exact(t=0)
    u0_cop = prob.u_exact(t=0).values
    u0_MN = u0.values
   

      
    # fill in u0-vector as right-hand side for the collocation problem
    #u0_MN = np.kron(np.ones(coll.num_nodes), u0.values)    
    #u0_LMN = np.kron(np.ones(coll.num_nodes), u0.values)
    
    #uk = np.kron(np.ones(coll.num_nodes), u0.values)
    
    #null = u0_LMN *0
    
    for i in range(0, coll.num_nodes-1):  
        u0_MN = np.append(u0_MN, u0_cop)

    
    null = u0_MN*0
    u0_LMN = u0_MN
    uk     = u0_MN

        
    for i in range(0, prob.params.nsteps-1):
        u0_LMN = np.append(u0_LMN, null)
        uk     = np.append(uk, u0_MN)

    
    J_uk = sp.kron(sp.eye(prob.params.nsteps * coll.num_nodes),prob.A) + sp.diags(1.0 / prob.params.eps ** 2 * (1.0 - (prob.params.nu + 1) * uk ** prob.params.nu), offsets=0)    
    #u0_LMN = sp.kron( sp.kron( linalg.inv(V) , sp.eye(coll.num_nodes)) ,sp.eye(prob.params.nvars)) *u0_coll    


    # build system matrix M of collocation problem
    
    #print(prob.A.size)
    
    for i in range(u0.values.size):
        f[i] = Q.dot(prob.eval_f(u0,0).values[i])
    f_cop = f
    
    for i in range(0, prob.params.nsteps*coll.num_nodes-1):
        f = np.append(f, f_cop)
    
    f_mat = sp.diags(f, offsets=0)
    
    #sp.kron(Q, np.diag(prob.eval_f(u0,0).values)
    M    = sp.eye(prob.params.nsteps * coll.num_nodes * prob.params.nvars)   - dt *        sp.kron(sp.eye(prob.params.nsteps), sp.kron(Q, prob.A))                          - sp.kron(E_diag, sp.kron(N, sp.eye(prob.params.nvars) ))
    Jac_ = sp.eye(prob.params.nsteps * coll.num_nodes * prob.params.nvars) -   dt *        sp.kron(sp.eye(prob.params.nsteps), sp.kron(Q, sp.eye(prob.params.nvars)))*sp.kron(sp.eye(prob.params.nsteps * coll.num_nodes),prob.A) - sp.kron(E_diag, sp.kron(N, sp.eye(prob.params.nvars) ))
        
    #Jac_ = sp.eye(prob.params.nsteps * coll.num_nodes * prob.params.nvars) -   dt *        sp.kron(sp.eye(prob.params.nsteps), sp.kron(Q, prob.A)) - sp.kron(E_diag, sp.kron(N, sp.eye(prob.params.nvars) ))    
        
    #F_ = (sp.eye(prob.params.nsteps * coll.num_nodes * prob.params.nvars).dot(uk) - dt * sp.kron(sp.eye(prob.params.nsteps), sp.kron( Q, sp.eye(prob.params.nvars) ))*f - sp.kron(E_diag, sp.kron(N, sp.eye(prob.params.nvars) )).dot(uk))-u0_LMN
    F_ = (sp.eye(prob.params.nsteps * coll.num_nodes * prob.params.nvars).dot(uk) - dt * f - sp.kron(E_diag, sp.kron(N, sp.eye(prob.params.nvars) )).dot(uk))-u0_LMN
    
    for n in range(5):
    
    
    	print(np.linalg.norm(F_))
    	#print()
    	### rhs = sp.kron( sp.kron( linalg.inv(V) , sp.eye(coll.num_nodes)) ,sp.eye(prob.params.nvars)) * (Jac_.dot(uk) -F_)
    	rhs = sp.kron( sp.eye(prob.params.nsteps * coll.num_nodes) ,sp.eye(prob.params.nvars)) * (Jac_.dot(uk) -F_)    	
        uk = sp.linalg.spsolve(Jac_, rhs) #M , u0_LMN) #Jac_, Jac_.dot(uk) +F_)
        
        #uk += u_delta
        
        ### uk = sp.kron( sp.kron( V , sp.eye(coll.num_nodes)) ,sp.eye(prob.params.nvars)) * uk 
        
        f = eval_f(prob,coll, uk)
        J_uk = sp.kron(sp.eye(prob.params.nsteps * coll.num_nodes),prob.A) + sp.diags(1.0 / prob.params.eps ** 2 * (1.0 - (prob.params.nu + 1) * uk ** prob.params.nu), offsets=0) 
        F_ = (sp.eye(prob.params.nsteps * coll.num_nodes * prob.params.nvars).dot(uk) - dt * sp.kron(sp.eye(prob.params.nsteps), sp.kron( Q, sp.eye(prob.params.nvars) ))*f - sp.kron(E_diag, sp.kron(N, sp.eye(prob.params.nvars) )).dot(uk))-u0_LMN
        Jac_ = sp.eye(prob.params.nsteps * coll.num_nodes * prob.params.nvars) -   dt *        sp.kron(sp.eye(prob.params.nsteps), sp.kron(Q, sp.eye(prob.params.nvars)))*sp.kron(sp.eye(prob.params.nsteps * coll.num_nodes),prob.A) - sp.kron(E_diag, sp.kron(N, sp.eye(prob.params.nvars) ))
    
        

    #print(u0_coll)
    #print(np.ones(coll.num_nodes))
    
    # get exact solution at Tend = dt
    uend = prob.u_exact(t=tend)

    # solve collocation problem directly




    #print(u_coll)
    
    
    #print(uend.values)

    # compute error
    err = np.linalg.norm(uk[-prob.params.nvars:] - uend.values, np.inf)

    return err


if __name__ == "__main__":
    main()
