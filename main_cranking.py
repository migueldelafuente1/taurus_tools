'''
Created on Jan 10, 2023

@author: Miguel
'''
import os
from tools.helpers import importAndCompile_taurus, TBME_SUITE

from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus
from scripts1d.cranking_scripts import run_J_surface

if not (InputTaurus.PROGRAM in os.listdir()): importAndCompile_taurus()

if __name__ == '__main__':
    
    ## MZ4 lengths
    interactions = {
        # ( 6, 6): (4, 0, 1.60), ( 6, 8): (4, 0, 1.77),
        # ( 8, 8): (4, 0, 1.85), ( 8,10): (4, 0, 1.65),
        # (10,10): (4, 0, 1.80), (10,12): (4, 0, 1.75),
        # (14,14): (4, 0, 1.65), (14,16): (4, 0, 1.80),
        # (16,16): (4, 0, 1.85), (16,18): (4, 0, 1.75),
        (18,18): (4, 0, 1.80), (18,20): (4, 0, 1.80),
        (20,20): (4, 0, 1.76), 
    }
    ## MZ5 lengths
    interactions = {
        ( 6, 6): (1, 0, 1.60), # ( 6, 7): (5, 0, 1.77),
        # # ( 8, 8): (5, 0, 1.85), ( 8, 9): (5, 0, 1.65),
        # (10,10): (5, 0, 1.80), (10,11): (5, 0, 1.75),
        # # (12,12): (5, 0, 1.80), (12,13): (5, 0, 1.75),
        # # (14,14): (5, 0, 1.65), (14,15): (5, 0, 1.80),
        # (16,16): (5, 0, 1.85), (16,17): (5, 0, 1.75),
        # (18,18): (5, 0, 1.80), (18,19): (5, 0, 1.80),
        # # (20,20): (5, 0, 1.76), 
    }
    nucleus = sorted(list(interactions.keys()))
    
    run_J_surface(nucleus, interactions, InputTaurus.ConstrEnum.Jx, 
                  seed_base=0, ROmega=(16,16), 
                  convergences=3,                  
                  j_min=-0.5, j_max=15.0, N_max=5 )
    