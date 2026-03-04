'''
Created on May 30, 2023

@author: Miguel
'''
import os
from scripts1d.hamiltonian_scripts import run_computingHOhbarOmegaForD1S, \
    run_computingHOhbarOmegaForD1S_Axial, run_computingHOhbarOmegaForArgonne
from tools.Enums import ArgonneEnum

if __name__ == '__main__':
    
    ## process for the Axial program (quicker)
    
    nucleus = [
        (  1,  1),
        # ( 10, 10),
        # ( 12, 12),
        # ( 12, 20),
    ]
    # nucleus = sorted(list(interactions.keys()))
    # run_computingHOhbarOmegaForD1S(nucleus, MZmax=1, bHO_min=1.1, bHO_max=2.0, Nsteps=3, MZmin=0)
    # run_computingHOhbarOmegaForD1S_Axial(nucleus, program="HFBaxialMZ4", MZmax=4,
    #                                      bHO_min=1.5, bHO_max=2.5, Nsteps=30)
    
    run_computingHOhbarOmegaForArgonne(nucleus, inter_av=ArgonneEnum.AV14,
                                       MZmax=7, bHO_min=1.4, bHO_max=2.6, 
                                       Nsteps=7, MZmin=0, fomenko_points=(9, 9))