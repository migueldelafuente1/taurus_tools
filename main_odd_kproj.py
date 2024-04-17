'''
Created on 18 mar 2024

@author: delafuente
'''
import os
from tools.Enums import GognyEnum
from scripts1d.beta_scripts import run_b20_FalseOE_Kprojections_Gogny, \
    run_b20_FalseOE_Kmixing, run_b20_FalseOE_Block1KAndPAV
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
        interactions_B1 = {(2, 1): (3, 0, None), (2, 3): (2, 0, None),}
        interactions_B1 = {(2, 1): 'B1_MZ3', (2, 3): 'B1_MZ3',}
    else:
        interactions_B1 = {
            #(12,  8) : (3, 0, 1.98), 
            #(12, 10): (3, 0, 1.94), (12, 12): (3, 0, 1.92), (12, 14): (3, 0, 1.95), 
            #(12, 16): (3, 0, 1.94), 
            ((8, 11), (3, 0, None)),
            #(12, 19): (4, 0, 1.8), (12, 21): (4, 0, 1.8),
            #(11, 12): (4, 0, 1.8), (12, 11): (4, 0, 1.8),
        }
        # ---------------------------------------------------------------------
        ## Automation for the a range of Z,N to evaluate
        # --------------------------------------------------------------------- 
        if False:
            Z_TOPS  = 8, 20
            N_TOPS  = Z_TOPS 
            DELTA_A = 6
            CASE = 'OE' # 'OO' # 'EE' # 
            
            i = 0
            interactions_B1 = {}
            for Z in range(Z_TOPS[0], Z_TOPS[1] +1):
                for N in range(max(N_TOPS[0], Z - DELTA_A), 
                               min(N_TOPS[1], Z + DELTA_A) +1, 1):
                    if   CASE == 'OE' and (Z + N) % 2 == 0:
                        continue
                    elif CASE == 'OO' and (Z % 2 == 0 or N % 2 == 0):
                        continue
                    elif CASE == 'EE' and (Z % 2 == 1 or N % 2 == 1):
                        continue
                    i += 1
                    print("[{:2}] z{:2} n{:2} A[{}]".format(i, Z, N, Z+N))
                    interactions_B1[(Z,N)] = (4, 0, None)
                print()
    
    nucleus = sorted(list(interactions_B1.keys()))
    # run_b20_FalseOE_Kprojections_Gogny(nucleus, interactions_B1, GognyEnum.B1,
    #                       seed_base=3, ROmega=(0,0), #ROmega=(14,14), 
    #                       q_min=-0.4, q_max=0.6, N_max=50, convergences=3, 
    #                       parity_2_block=1)
    
    args = (nucleus, interactions_B1, GognyEnum.B1)
    kwargs = dict(
        seed_base=3, ROmega=(0,0), 
        q_min=-0.6, q_max=0.6, N_max=5, convergences=3, 
        parity_2_block=1,
    )
    # run_b20_FalseOE_Kmixing(*args, **kwargs)
    # raise Exception("STOP HERE.")
    
    K2block = 1
    args = (nucleus, interactions_B1, GognyEnum.B1, K2block)
    run_b20_FalseOE_Block1KAndPAV(*args, **kwargs, )
    
    raise Exception("STOP HERE.")
