'''
Created on 25 oct 2024

@author: delafuente
'''
import os
from tools.helpers import importAndCompile_taurus, TBME_SUITE, printf, __log_file

from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus
from tools.Enums import GognyEnum
from scripts2d.pair_scripts2d import run_pair_surfaces_2d

if not (InputTaurus.PROGRAM in os.listdir()): importAndCompile_taurus()

if __name__ == '__main__':
    
    ## MZ4 lengths
    interactions = {
        (10,10): (4, 0, 1.80), (10,12): (4, 0, 1.75),
        (12,12): (4, 0, 1.75), (12,14): (4, 0, 1.75),
    }
    interactions = {
        (2, 1) : 'B1_MZ3',
    }
    
    nucleus = sorted(list(interactions.keys()))    
    
    PAIR_CONSTRS = {
        InputTaurus.ConstrEnum.P_T00_J10   : (-0.3, 0.5, 5), 
        InputTaurus.ConstrEnum.P_T1m1_J00  : (-0.5,  0.5, 5),
    }
    
    constr_onrun = {
        #InputTaurus.ConstrEnum.Jx: 0.0
    }
    
    run_pair_surfaces_2d(
        nucleus, interactions, PAIR_CONSTRS,
        gogny_interaction=GognyEnum.B1, ROmega=(0,0), convergences=5,
        seed_base=0, 
        **constr_onrun
    )
    printf("I finished!")