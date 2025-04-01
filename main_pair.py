'''
Created on Jan 10, 2023

@author: Miguel
'''
import os
from tools.helpers import importAndCompile_taurus, TBME_SUITE, printf, __log_file

from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus
from scripts1d.pair_scripts import run_pair_surface_D1S
from scripts1d.afterrun_hamil_voidstep_scripts import \
    run_b20_decomposeHamiltonian_Gogny, run_b20_decomposeHamiltonian_M3Y

from tools.base_executors import _Base1DTaurusExecutor

from tools.Enums import GognyEnum, M3YEnum

if not (InputTaurus.PROGRAM in os.listdir()): importAndCompile_taurus()

if __name__ == '__main__':
    
    ## MZ4 lengths
    M = 5
    interactions = {
        # ( 6, 7): (1, 0, 1.60), ( 7, 8): (1, 0, 1.77),
        # ( 8, 8): (4, 0, 1.85), ( 8,10): (4, 0, 1.65),
        # (10,10): (2, 0, 1.80), (10,12): (4, 0, 1.75),
        # (12,12): (4, 0, 1.75), (12,14): (4, 0, 1.75),
        
        (10,10): (M, 0, None), # (12,16): (2, 0, None), 
        (14,14): (M, 0, None), (16,16): (M, 0, None),
        (18,18): (M, 0, None), ( 8, 8): (M, 0, None), 
        
        # (10,11): (5, 0, None), (10,12): (5, 0, None),
        # (12,16):  'bench', #(5, 0, None), #
        (12,10): (M, 0, None), (12,11): (M, 0, None), 
        (12,12): (M, 0, None), (12,13): (M, 0, None), 
        (12,14): (M, 0, None), (12,15): (M, 0, None),
        (12,16): (M, 0, None), (12,17): (M, 0, None), 
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
    
    ## MZ large test
    # interactions = {
    #     (10,10): (7, 0, None), 
    #     (16,16): (7, 0, None),
    #     (24,24): (7, 0, None), 
    # }
    
    nucleus = sorted(list(interactions.keys()))    
    
    PAIR_CONSTRS = [
        InputTaurus.ConstrEnum.P_T00_J10,  
        InputTaurus.ConstrEnum.P_T10_J00,
        InputTaurus.ConstrEnum.P_T1m1_J00, 
        InputTaurus.ConstrEnum.P_T1p1_J00,
        InputTaurus.ConstrEnum.P_T00_J1m1,
        InputTaurus.ConstrEnum.P_T00_J1p1,
    ]
    
    constr_onrun = {
        InputTaurus.ConstrEnum.b10 : (0.0, 0.0),
        InputTaurus.ConstrEnum.b11 : (0.0, 0.0),
        InputTaurus.ConstrEnum.b21 : (0.0, 0.0),
        InputTaurus.ConstrEnum.b31 : (0.0, 0.0),
        InputTaurus.ConstrEnum.b41 : (0.0, 0.0),
        #InputTaurus.ConstrEnum.Jx: 0.0
        InputTaurus.ConstrEnum.P_T00_J10 : 0,  
        InputTaurus.ConstrEnum.P_T10_J00 : 0,
        InputTaurus.ConstrEnum.P_T1m1_J00: 0, 
        InputTaurus.ConstrEnum.P_T1p1_J00: 0,
        InputTaurus.ConstrEnum.P_T00_J1m1: 0,
        InputTaurus.ConstrEnum.P_T00_J1p1: 0,
    }
    fomenko_points = (7, 7)
    ROmega         = (0, 0) #(14,14) #
    if True:
        run_pair_surface_D1S(nucleus, interactions, PAIR_CONSTRS,
                             gogny_interaction=GognyEnum.B1,
                             # gogny_interaction=M3YEnum.P2,
                             ROmega=ROmega, convergences=4,
                             seed_base=4, 
                             p_min=-1.5, p_max=1.5, N_max=31, #31
                             fomenko_points=fomenko_points, parity_2_block=1,
                             sym_calc_setup=_Base1DTaurusExecutor.SymmetryOptionsEnum.NO_CORE_CALC,
                             **constr_onrun)
        # interactions[()] = (5, 0, None)
    # run_b20_decomposeHamiltonian_Gogny(interactions, GognyEnum.B1, PAIR_CONSTRS, fomenko_points, ROmega) 
    # run_b20_decomposeHamiltonian_M3Y(interactions, M3YEnum.P2, PAIR_CONSTRS, fomenko_points, ROmega)
    printf("I finished!")

