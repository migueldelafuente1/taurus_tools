'''
Created on 18 mar 2024

@author: delafuente
'''
import os
from tools.Enums import GognyEnum
from scripts1d.beta_Kblocking_scripts import \
    run_b20_FalseOE_Kprojections_Gogny, run_b20_FalseOE_Block1KAndPAV, \
    run_b20_FalseOE_Kmixing, run_b20_FalseOE_Kmixing_exampleSingleJ, \
    run_b20_Block1KandPAV_exampleSingleJ
from tools.inputs import InputTaurus
from tools.helpers import importAndCompile_taurus, printf, __log_file

if not (InputTaurus.PROGRAM in os.listdir()): importAndCompile_taurus()

if __name__ == '__main__':
    
    
    if os.getcwd().startswith('C:'):   ## TESTING
        interactions_B1 = {(2, 1): (3, 0, None), (2, 3): (2, 0, None),}
        interactions_B1 = {(2, 1): 'B1_MZ3', (2, 3): 'B1_MZ3',}
    else:
        
        # ---------------------------------------------------------------------
        ## Automation for the a range of Z,N to evaluate
        # --------------------------------------------------------------------- 
        # if Falsae:
        #     Z_TOPS  = 8, 20
        #     N_TOPS  = Z_TOPS 
        #     DELTA_A = 6
        #     CASE = 'OE' # 'OO' # 'EE' # 
        #
        #     i = 0
        #     interactions_B1 = {}
        #     for Z in range(Z_TOPS[0], Z_TOPS[1] +1):
        #         for N in range(max(N_TOPS[0], Z - DELTA_A), 
        #                        min(N_TOPS[1], Z + DELTA_A) +1, 1):
        #             if   CASE == 'OE' and (Z + N) % 2 == 0:
        #                 continue
        #             elif CASE == 'OO' and (Z % 2 == 0 or N % 2 == 0):
        #                 continue
        #             elif CASE == 'EE' and (Z % 2 == 1 or N % 2 == 1):
        #                 continue
        #             i += 1
        #             printf("[{:2}] z{:2} n{:2} A[{}]".format(i, Z, N, Z+N))
        #             interactions_B1[(Z,N)] = (4, 0, None)
        #         printf()
        
        inter_ = (4, 0, None)
        #interactions_B1 = dict([(( 7, 8+ 2*i), inter_) for i in range(0, 7)])
        #interactions_B1 = dict([((11, 8+ 2*i), inter_) for i in range(0, 7)])
        #interactions_B1 = dict([((12,11+ 2*i), inter_) for i in range(0, 6)])
        #interactions_B1 = dict([((13,10+ 2*i), inter_) for i in range(0, 6)])
        #interactions_B1 = dict([((15, 8+ 2*i), inter_) for i in range(0, 6)])
    
    class __CASES:
        _0 = 'exe_surf_allK_base_noPAV' # doesnt do all-sp per K (all K)
        _1 = 'exe_surf_1K_and_PAV'      # does all-sp for K, also PAV
        _2 = 'exe_surf_Kmixing_and_PAV' # does all-sp for all K, also PAV
        _3 = 'exe_example_h11/2_singleJ'
    
    ## SELECT HERE ****
    _case = __CASES._2
    
    nucleus = sorted(list(interactions_B1.keys()))
    
    #===========================================================================
    if   _case == __CASES._0:
        run_b20_FalseOE_Kprojections_Gogny(nucleus, interactions_B1, GognyEnum.B1,
                              seed_base=3, ROmega=(0,0), #ROmega=(14,14), 
                              q_min=-0.4, q_max=0.6, N_max=50, convergences=3, 
                              parity_2_block=1)
        #=======================================================================
    elif _case == __CASES._1:   
        ## single-K execution with PAV
        K2block = 1
        args = (nucleus, interactions_B1, GognyEnum.B1, K2block)
        kwargs = dict(
            seed_base=3, ROmega=(0,0),
            q_min=-0.8, q_max=0.8, N_max=13, convergences=0,   ## 0.6, 25
            parity_2_block=1,
            fomenko_points=(7, 7),
            preconverge_blocking_sts=False,
            find_Kfor_all_sps = True
        )
        run_b20_FalseOE_Block1KAndPAV(*args, **kwargs, )
        #=======================================================================
    elif _case == __CASES._2:   
        ## KMIXING execution
        args = (nucleus, interactions_B1, GognyEnum.B1)
        kwargs = dict(
            valid_Ks = [1, 3, 5],
            seed_base=3, ROmega=(0,0),
            q_min=-0.8, q_max=0.8, N_max=13, convergences=0,   ## 0.6, 25
            parity_2_block=1,
            fomenko_points=(7, 7),
            preconverge_blocking_sts=False,
            find_Kfor_all_sps = True
        )
        run_b20_FalseOE_Kmixing(*args, **kwargs)
        #=======================================================================
    elif _case == __CASES._3:
        ## TEST with the h11/2 state
        
        interactions_B1 = {(0, 3):  'B1_h11o2', }
        nucleus = sorted(list(interactions_B1.keys()))
        
        valid_K = 11,
        args = (nucleus, interactions_B1, GognyEnum.B1, valid_K)
        kwargs = dict(
            seed_base=3, ROmega=(0,0),
            q_min=-0.6, q_max=0.6, N_max=25, convergences=3,   ## 0.6, 25
            parity_2_block=-1,
            fomenko_points=(0, 7),
        )
        
        run_b20_Block1KandPAV_exampleSingleJ(*args, **kwargs)
        
        ## NOTE: (DEPRECATED) calculation with the False-OE method.
        args = (nucleus, interactions_B1, GognyEnum.B1)
        kwargs = dict(
            valid_Ks = [1, 3, 5, 7, 9, 11], 
            seed_base=3, ROmega=(0,0),
            q_min=-0.6, q_max=0.6, N_max=25, convergences=3,   ## 0.6, 25
            parity_2_block=-1,
            fomenko_points=(0, 7),
            preconverge_blocking_sts=False,
            find_Kfor_all_sps = True
        )
            
        # run_b20_FalseOE_Kmixing_exampleSingleJ(*args, **kwargs)
        #=======================================================================
    
    printf("END OF SUITE.")