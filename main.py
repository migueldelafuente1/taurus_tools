'''
Created on Jan 10, 2023

@author: Miguel
'''
import os
from scripts1d.beta_scripts import run_q20_surface, run_b20_surface
from tools.helpers import importAndCompile_taurus, TBME_SUITE

from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus
from scripts1d.pair_scripts import run_pair_surface_D1S
from scripts1d.cranking_scripts import run_J_surface
from scripts0d.unconstrained_scripts import run_symmetry_restricted_for_hamiltonian

if not (InputTaurus.PROGRAM in os.listdir()):
    importAndCompile_taurus()

if __name__ == '__main__':
    
    # # exe_ = TBME_HamiltonianManager(1.75, 1, set_com2=True)
    # # exe_.setAndRun_D1Sxml()
    #
    # # importAndCompile_taurus()
    #
    # # nucleus = [(2, 4), (4, 4)]
    # # interactions = {(2, 4): (2,0, 2.0), 
    # #                 (4, 4): 'hamil_2'}
    # # run_b20_surface(nucleus, interactions, q_min=-1., q_max=1., N_max=10)
    interactions = {
        ( 10, 11): (2, 0, 1.80),
        ( 11, 12): (2, 0, 1.80),
        ( 11, 11): (2, 0, 1.85),
        ( 10, 10): (2, 0, 1.85),
    }
    nucleus = sorted(list(interactions.keys()))
    run_b20_surface(nucleus, interactions, 
                    seed_base=5, ROmega=(13,15), 
                    q_min=-0.5, q_max=1.0, N_max=51, convergences=20)
    # raise Exception("STOP HERE.")
    #
    # ## MZ4 lengths
    # interactions = {
    #     # ( 6, 6): (4, 0, 1.60), ( 6, 8): (4, 0, 1.77),
    #     # ( 8, 8): (4, 0, 1.85), ( 8,10): (4, 0, 1.65),
    #     # (10,10): (4, 0, 1.80), (10,12): (4, 0, 1.75),
    #     # (14,14): (4, 0, 1.65), (14,16): (4, 0, 1.80),
    #     # (16,16): (4, 0, 1.85), (16,18): (4, 0, 1.75),
    #     (18,18): (4, 0, 1.80), (18,20): (4, 0, 1.80),
    #     (20,20): (4, 0, 1.76), 
    # }
    # nucleus = sorted(list(interactions.keys()))
    #
    # run_J_surface(nucleus, interactions, InputTaurus.ConstrEnum.Jx,
    #               seed_base=0, j_min=0.0, j_max=25.0, N_max=50)
    #
    #
    # PAIR_CONSTRS = [
    #     InputTaurus.ConstrEnum.P_T00_J10,  InputTaurus.ConstrEnum.P_T10_J00,
    #     InputTaurus.ConstrEnum.P_T1m1_J00, InputTaurus.ConstrEnum.P_T1p1_J00
    # ]
    # # run_pair_surface_D1S(nucleus, interactions, PAIR_CONSTRS, 
    # #                  seed_base=5, p_min=-0.05, p_max=2.0, N_max=41)
    
    nucleus = [(2,1+n) for n in range(9)] #18, 14
    run_symmetry_restricted_for_hamiltonian(nucleus, 
                                            MZmax=1, seed_base=0, 
                                            ROmega=(15,16), convergences=20)
    
        