'''
Created on May 30, 2023

@author: Miguel
'''
import os
from scripts1d.beta_scripts import run_q20_surface, run_b20_surface
from tools.helpers import importAndCompile_taurus, TBME_SUITE
from scripts1d.hamiltonian_scripts import run_computingHOhbarOmegaForD1S, \
    run_computingHOhbarOmegaForD1S_Axial

if __name__ == '__main__':
    
    ## process for the Axial program (quicker)
    
    nucleus = [
        (  8,  8),
        ( 10, 10),
        ( 12, 12),
        ( 12, 20),
    ]
    # nucleus = sorted(list(interactions.keys()))
    # run_computingHOhbarOmegaForD1S(nucleus, MZmax=1, bHO_min=1.1, bHO_max=2.0, Nsteps=3, MZmin=0)
    run_computingHOhbarOmegaForD1S_Axial(nucleus, program="HFBaxialMZ4", MZmax=4,
                                         bHO_min=1.5, bHO_max=2.5, Nsteps=30)