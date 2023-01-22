'''
Created on Jan 20, 2023

@author: Miguel
'''
import numpy as np
from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus

def run_computingHOhbarOmegaForD1S(nucleus, b_min=1.5, b_max=2.75, N=6):
    
    b_lengths = list(np.linspace(b_min, b_max, N, endpoint=True))
    
    InputTaurus.set_inputDDparamsFile(
        **{InputTaurus.InpDDEnum.r_dim : 12,
           InputTaurus.InpDDEnum.omega_dim : 14})
    
    input_args_start = {
        InputTaurus.ArgsEnum.com : 1,
        InputTaurus.ArgsEnum.seed: 5,
        InputTaurus.ArgsEnum.iterations: 1000,
        InputTaurus.ArgsEnum.grad_type: 1,
        InputTaurus.ArgsEnum.grad_tol : 0.001,
        InputTaurus.ArgsEnum.beta_schm: 0, ## 0= q_lm, 1 b_lm, 2 triaxial
        InputTaurus.ArgsEnum.pair_schm: 1,
    }
    
    # input_args_onrun = {
    #     InputTaurus.ArgsEnum.red_hamil: 1,
    #     InputTaurus.ArgsEnum.seed: 1,
    #     InputTaurus.ArgsEnum.iterations: 600,
    #     InputTaurus.ArgsEnum.grad_type: 1,
    #     InputTaurus.ArgsEnum.grad_tol : 0.01,
    # }
    
    for z, n in nucleus:
        
        for b in b_lengths.reverse():
            
            ## TODO: set up the Hamiltonian in the set up 
            pass
        
            ## TODO: input args_for must change seeed=1 after the right minimum
            
            ## TODO: after the export (do not zip the files) import the results and 
            ## copy here to an auxiliary file
            
            
            













if __name__ == '__main__':
    pass