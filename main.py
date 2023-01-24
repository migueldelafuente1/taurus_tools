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
    
    # exe_ = TBME_HamiltonianManager(1.75, 1, set_com2=True)
    # exe_.setAndRun_D1Sxml()
    
    # importAndCompile_taurus()
    
    # nucleus = [(2, 4), (4, 4)]
    # interactions = {(2, 4): (2,0, 2.0), 
    #                 (4, 4): 'hamil_2'}
    # run_b20_surface(nucleus, interactions, q_min=-1., q_max=1., N_max=10)
    
    
    
    ## MZ4 lengths
    interactions = {( 6, 6): (4, 0, 2.0), ( 6, 8): (4, 0, 2.0),
                    ( 8, 8): (4, 0, 2.0), ( 8,10): (4, 0, 2.0),
                    (10,10): (4, 0, 2.0),
                    (14,14): (4, 0, 2.0), (14,16): (4, 0, 2.0),
                    (16,16): (4, 0, 2.0), (16,18): (4, 0, 2.0),
                    (18,18): (4, 0, 2.0), (18,20): (4, 0, 2.0),
                    (20,20): (4, 0, 2.0), 
                    }
    nucleus = sorted(list(interactions.keys()))
    
    run_J_surface(nucleus, interactions, InputTaurus.ConstrEnum.Jx,
                  seed_base=0, p_min=0.0, p_max=25.0, N_max=50)
    
    
    PAIR_CONSTRS = [
        InputTaurus.ConstrEnum.P_T00_J10,  InputTaurus.ConstrEnum.P_T10_J00,
        InputTaurus.ConstrEnum.P_T1m1_J00, InputTaurus.ConstrEnum.P_T1p1_J00
    ]
    # run_pair_surface(nucleus, interactions, PAIR_CONSTRS, 
    #                  seed_base=5, p_min=-0.05, p_max=2.0, N_max=41)
        