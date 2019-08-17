import os
import pickle
import numpy as np
import subprocess

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.parallelSDC.ErrReductionHook import err_reduction_hook
#from pySDC.projects.parallelSDC.AC_2D_FD_implicit_Jac import AC_jac
#from pySDC.implementations.problem_classes.GrayScott_1D_FD import grayscott_fullyimplicit
from pySDC.implementations.problem_classes.GrayScott_1D_FD_jac_neu import grayscott_fullyimplicit, grayscott_fullyimplicit_jac
#from pySDC.implementations.problem_classes.GrayScott_2D_FD import grayscott_fullyimplicit

from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel_MPI import linearized_implicit_fixed_parallel_MPI
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel_prec_MPI import linearized_implicit_fixed_parallel_prec_MPI
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel import linearized_implicit_fixed_parallel
from pySDC.projects.parallelSDC.linearized_implicit_fixed_parallel_prec import linearized_implicit_fixed_parallel_prec
from pySDC.projects.parallelSDC.div_linearized_implicit_fixed_parallel_MPI import div_linearized_implicit_fixed_parallel_MPI
from pySDC.projects.parallelSDC.div_linearized_implicit_fixed_parallel_prec_MPI import div_linearized_implicit_fixed_parallel_prec_MPI

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.parallelSDC.BaseTransfer_MPI import base_transfer_MPI
from pySDC.projects.parallelSDC.div_BaseTransfer_MPI import div_base_transfer_MPI
import matplotlib.pyplot as plt


from pySDC.projects.parallelSDC.generic_implicit_MPI import generic_implicit_MPI
from pySDC.projects.parallelSDC.div_generic_implicit_MPI import div_generic_implicit_MPI
from mpi4py import MPI

def run(sweeper_list, MPI_fake=True, controller_comm=MPI.COMM_WORLD, node_comm=None, node_list=None):


###############################################################################################################################

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-10
    level_params['dt'] = 1 #E-03
    level_params['nsweeps'] = [1]

    # This comes as read-in for the step class (this is optional!)
    step_params = dict()
    step_params['maxiter'] = 50

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nvars'] = [(2,64),(2,32)] #[(2,8)] # , (2,4)] #[(256,256), (128, 128)]  #[(256,256), (128, 128)] #[512, 256] 
    problem_params['newton_maxiter'] = 1#00
    problem_params['newton_tol'] = 1E-12#4
    problem_params['lin_tol'] = 1E-09
    problem_params['lin_maxiter'] = 100
    problem_params['radius'] = 0.25
    problem_params['comm'] = node_comm #time_comm #MPI.COMM_WORLD #node_comm
    
    problem_params['Du'] = 1.0
    problem_params['Dv'] = 0.01
    problem_params['f'] = 0.09
    problem_params['k'] = 0.086  
    problem_params['inner_maxiter'] = 1
    problem_params['inner_tol'] = 1E-12

    # This comes as read-in for the sweeper class
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']#, 'LU']
    sweeper_params['spread'] = False
    sweeper_params['fixed_time_in_jacobian'] = 0
    sweeper_params['comm'] = node_comm
    sweeper_params['node_list'] = node_list
    print("list",node_list)  
    
    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6
    space_transfer_params['periodic'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    #controller_params['predict'] = False
    controller_params['hook_class'] = err_reduction_hook

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = grayscott_fullyimplicit #_jac 
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
    t0 = 0
    Tend = 20 #01 #08 #0.01 #08 #04 #32




    # loop over the different sweepers and check results
    serial =True
    for sweeper in sweeper_list:
        description['sweeper_class'] = sweeper
        error_reduction = []
        for dt in [1]: #1e-3]:
            print('Working with sweeper %s and dt = %s...' % (sweeper.__name__, dt))

            level_params['dt'] = dt
            description['level_params'] = level_params

            # instantiate the controller and stuff
            if(sweeper.__name__=='generic_implicit'):
                if(controller_comm.Get_size()==1):
                    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
                    serial=False
                else:
                    controller = controller_MPI(controller_params=controller_params, description=description, comm=controller_comm)   
            elif (sweeper.__name__=='linearized_implicit_fixed_parallel_prec_MPI'or sweeper.__name__=="linearized_implicit_fixed_parallel_MPI" or sweeper.__name__=='linearized_implicit_fixed_parallel_prec'or sweeper.__name__=="linearized_implicit_fixed_parallel"):
                #if(node_comm is None):
                    #description['base_transfer_class'] = base_transfer                
                #controller = controller_MPI(controller_params=controller_params, description=description, comm=controller_comm)                  
                #else: 
                #    serial==False

                if (sweeper.__name__=='linearized_implicit_fixed_parallel_prec_MPI'or sweeper.__name__=="linearized_implicit_fixed_parallel_MPI"): 
                    description['base_transfer_class'] = base_transfer_MPI
                    controller = controller_MPI(controller_params=controller_params, description=description, comm=controller_comm)
                else:
                    controller = controller_MPI(controller_params=controller_params, description=description, comm=controller_comm)
                    #controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
                    #serial=False                
            elif (sweeper.__name__=='div_linearized_implicit_fixed_parallel_prec_MPI' or sweeper.__name__=='div_linearized_implicit_fixed_parallel_MPI'):    
                description['base_transfer_class'] = div_base_transfer_MPI
                controller = controller_MPI(controller_params=controller_params, description=description, comm=controller_comm)
                serial==False

            if(serial==True):
                P = controller.S.levels[0].prob            
            else:    
                P = controller.MS[0].levels[0].prob                        

            # get initial values on finest level
            uinit = P.u_exact(t0)

            plt.plot(uinit.values[0])
            
            # call main function to get things done...
            MPI.COMM_WORLD.Barrier()            
            t1 = MPI.Wtime()
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
            t2 = MPI.Wtime()          
            time =t2-t1   
            print( "My elapsed time is ", time)

            #print(uend.values)  

            plt.plot(uend.values[0])

            plt.savefig('20.png')             

            #MPI.COMM_WORLD.Barrier()     
            print("------------------------------------------------------------------------------------------------------------------------------------------")  
            #exit()     

def main():
    """
    Main driver

    """


 
    #PFASST nur Zeit Parallel 
    run([generic_implicit], controller_comm=MPI.COMM_WORLD) 
    
    #neue Versionen ABER seriell in den Knoten
    #run([linearized_implicit_fixed_parallel], controller_comm=MPI.COMM_WORLD)    
    #run([linearized_implicit_fixed_parallel_prec], controller_comm=MPI.COMM_WORLD) 
 
 
   
    #MPI default communicator
    comm = MPI.COMM_WORLD    
    
    #neue Versionen parallel in den Knoten, mit 4 Proz
    #world_rank = comm.Get_rank()
    #world_size = comm.Get_size()

    #color = int(world_rank / 4) 

    #node_comm = comm.Split(color=color)
    #node_rank = node_comm.Get_rank()

    #color = int(world_rank % 4) 

    #time_comm = comm.Split(color=color)
    #time_rank = time_comm.Get_rank()      
    
    #run([linearized_implicit_fixed_parallel_prec], controller_comm=MPI.COMM_WORLD)#time_comm, node_comm=node_comm)    
    ##run([linearized_implicit_fixed_parallel_prec_MPI], controller_comm=time_comm, node_comm=node_comm)    

    #neue Versionen parallel in den Knoten, mit 8 Proz
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    color = int(world_rank / 4) 
    node_comm = comm.Split(color=color)
    node_rank = node_comm.Get_rank()

    color = int(world_rank % 4) 
    time_comm = comm.Split(color=color)
    time_rank = time_comm.Get_rank()
    
   
    #if(node_rank==0):
    #    node_list =[0,2]       
    #else:
    #    node_list =[1,3]    
    #print(node_list)
    #run([linearized_implicit_fixed_parallel_prec], controller_comm=comm)

    #run([linearized_implicit_fixed_parallel_MPI], controller_comm=time_comm, node_comm=node_comm)  
    #run([div_linearized_implicit_fixed_parallel_prec_MPI], MPI_fake=False, controller_comm=time_comm, node_comm=node_comm)  

    print("fertig klm")
    #exit()
if __name__ == "__main__":
    main()
    #exit()
    print("in if")
print("ganz am ende")



