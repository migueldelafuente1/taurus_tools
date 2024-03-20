'''
Created on 18 mar 2024

@author: delafuente
'''
import os
from tools.Enums import GognyEnum
from scripts1d.beta_scripts import run_b20_FalseOE_Kprojections_Gogny
from tools.inputs import InputTaurus
from tools.helpers import importAndCompile_taurus

if not (InputTaurus.PROGRAM in os.listdir()): importAndCompile_taurus()

if __name__ == '__main__':
    
        
    interactions_D1S = {
        # (12,  8): (3, 0, 1.81), (12, 10): (3, 0, 1.83), 
        # (12, 12): (3, 0, 1.83), (12, 14): (3, 0, 1.79), 
        # (12, 16): (3, 0, 1.80), (12, 18): (3, 0, 1.86),
    }
    if os.getcwd().startswith('C:'):   ## TESTING
        interactions_B1 = {(2, 1): (2, 0, None), (2, 3): (2, 0, None),}
    else:
        interactions_B1 = {
            #(12,  8) : (3, 0, 1.98), 
            #(12, 10): (3, 0, 1.94), (12, 12): (3, 0, 1.92), (12, 14): (3, 0, 1.95), 
            #(12, 16): (3, 0, 1.94), 
            (12, 19): (3, 0, 1.98), (12, 21): (3, 0, 2.01),
        }
    
    nucleus = sorted(list(interactions_B1.keys()))
    run_b20_FalseOE_Kprojections_Gogny(nucleus, interactions_B1, GognyEnum.B1,
                          seed_base=3, ROmega=(0,0), #ROmega=(14,14), 
                          q_min=-0.4, q_max=0.6, N_max=50, convergences=5)
    
    raise Exception("STOP HERE.")