'''
Created on Jan 10, 2023

@author: Miguel
'''
import os
from scripts1d.beta_scripts import run_q20_surface, run_b20_surface
from tools.helpers import importAndCompile_taurus, TBME_SUITE

from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus
from scripts1d.pair_scripts import run_pair_surface
from scripts1d.cranking_scripts import run_J_surface

if not (InputTaurus.PROGRAM in os.listdir()):
    importAndCompile_taurus()

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
    nucleus = sorted(list(interactions.keys()))
    
    run_J_surface(nucleus, interactions, InputTaurus.ConstrEnum.Jx,
                  seed_base=0, j_min=0.0, j_max=25.0, N_max=50)
    