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

def f(x):
    return np.arctan(x);
    
def fp(x):
    return 1./(1.+x*x);

def main():

    x=10.;
    r = np.abs(f(x));
    tau1=1e-6;
    tau2=1e-6;
    print(np.abs(f(x)) > tau1*r +tau2);
    while np.abs(f(x)) > tau1*r +tau2:
        #print(f(10.));
        s= -f(x)/fp(x);
        while np.abs(f(x+s)) > np.abs(f(x)):
            s=s/2.;
	x+=s;
	print(x);







if __name__ == "__main__":
    main()

