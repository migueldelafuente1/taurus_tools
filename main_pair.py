'''
Created on Jan 10, 2023

@author: Miguel
'''
import os
from tools.helpers import importAndCompile_taurus, TBME_SUITE

from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus
from scripts1d.pair_scripts import run_pair_surface_D1S
from scripts1d.cranking_scripts import run_J_surface

if not (InputTaurus.PROGRAM in os.listdir()): importAndCompile_taurus()

if __name__ == '__main__':
    
    ## MZ4 lengths
    interactions = {
        # ( 6, 7): (1, 0, 1.60), ( 7, 8): (1, 0, 1.77),
        # ( 8, 8): (4, 0, 1.85), ( 8,10): (4, 0, 1.65),
        (10,10): (4, 0, 1.80), (10,12): (4, 0, 1.75),
        (12,12): (4, 0, 1.75), (12,14): (4, 0, 1.75),
        # (14,14): (4, 0, 1.65), (14,16): (4, 0, 1.80),
        # (16,16): (4, 0, 1.85), (16,18): (4, 0, 1.75),
        # (17,18): (4, 0, 1.80), (16,17): (4, 0, 1.80),
        # (17,17): (4, 0, 1.76), 
    }
    ## MZ5 lengths
    # interactions = {
    #     # ( 8, 9): (5, 0, 1.80),
    #     # (10,11): (5, 0, 1.78),
    #     # (12,13): (5, 0, 1.80),
    #     # (14,15): (5, 0, 1.75),
    #     # (16,17): (5, 0, 1.80),
    #     # (18,19): (5, 0, 1.80),
    #     # (20,19): (5, 0, 1.85),
    # }
    
    nucleus = sorted(list(interactions.keys()))    
    
    PAIR_CONSTRS = [
        InputTaurus.ConstrEnum.P_T00_J10,  InputTaurus.ConstrEnum.P_T10_J00,
        InputTaurus.ConstrEnum.P_T1m1_J00, InputTaurus.ConstrEnum.P_T1p1_J00
    ]
    
    constr_onrun = {
        InputTaurus.ConstrEnum.Jx: 0.0}
    
    run_pair_surface_D1S(nucleus, interactions, PAIR_CONSTRS, 
                         ROmega=(14,16), convergences=3,
                         seed_base=0, 
                         p_min=-0.05, p_max=2.0, N_max=41,
                         **constr_onrun
                         )
    print("I finished!")