'''
Created on 18 mar 2024

@author: delafuente
'''
import os
from tools.Enums import GognyEnum
from scripts1d.beta_scripts import run_b20_FalseOE_Kprojections_Gogny, \
    run_b20_FalseOE_Kmixing, run_b20_FalseOE_Block1KAndPAV
from tools.inputs import InputTaurus
from tools.helpers import importAndCompile_taurus, printf, __log_file

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
            (8, 11): (2, 0, None),
            #(12, 19): (4, 0, 1.8), (12, 21): (4, 0, 1.8),
            #(11, 12): (4, 0, 1.8), (12, 11): (4, 0, 1.8),
        }
        
        inter_ = (4, 0, None)
        #interactions_B1 = dict([((12,11+ 2*i), inter_) for i in range(0, 6)])
        #interactions_B1 = dict([((13,10+ 2*i), inter_) for i in range(0, 6)])
        #interactions_B1 = dict([((15, 8+ 2*i), inter_) for i in range(0, 6)])
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
                    printf("[{:2}] z{:2} n{:2} A[{}]".format(i, Z, N, Z+N))
                    interactions_B1[(Z,N)] = (4, 0, None)
                printf()
    
    nucleus = sorted(list(interactions_B1.keys()))
    # run_b20_FalseOE_Kprojections_Gogny(nucleus, interactions_B1, GognyEnum.B1,
    #                       seed_base=3, ROmega=(0,0), #ROmega=(14,14), 
    #                       q_min=-0.4, q_max=0.6, N_max=50, convergences=3, 
    #                       parity_2_block=1)
    
    if True: # to do the main evaluation.
        args = (nucleus, interactions_B1, GognyEnum.B1)
        kwargs = dict(
            valid_Ks = [1, 3, 5], 
            seed_base=3, ROmega=(0,0),
            q_min=-0.8, q_max=0.8, N_max=7, convergences=0,   ## 0.6, 25
            parity_2_block=1,
            fomenko_points=(7, 7),
            preconverge_blocking_sts=130,
            find_Kfor_all_sps = False
        )
        run_b20_FalseOE_Kmixing(*args, **kwargs)
        raise Exception("STOP HERE.")
    #
    # K2block = 1
    # args = (nucleus, interactions_B1, GognyEnum.B1, K2block)
    # run_b20_FalseOE_Block1KAndPAV(*args, **kwargs, )
    #
    # raise Exception("STOP HERE.")

    ## TEST with the h11/2 state
    
    interactions_B1 = {(0, 1): 'B1_h11o2', }
    nucleus = sorted(list(interactions_B1.keys()))
    
    args = (nucleus, interactions_B1, GognyEnum.B1)
    kwargs = dict(
        valid_Ks = [1, 3, 5, 7, 9, 11], 
        seed_base=3, ROmega=(0,0),
        q_min=-0.8, q_max=0.8, N_max=5, convergences=3,   ## 0.6, 25
        parity_2_block=-1,
        fomenko_points=(0, 7),
        preconverge_blocking_sts=False,
    )
    run_b20_FalseOE_Kmixing(*args, **kwargs)
    
    