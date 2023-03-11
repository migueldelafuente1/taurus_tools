'''
Created on Jan 20, 2023

@author: Miguel
'''
import os
import numpy as np
from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus
from tools.executors import ExeTaurus0D_EnergyMinimum, ExecutionException

def run_computingHOhbarOmegaForD1S(nucleus, MZmax=4, bHO_min=1.5, bHO_max=2.75, 
                                   Nsteps=6, MZmin=0):
    """ 
    Script for Taurus, get a curve b length to see where is the minimum of E hfb
    """
    b_lengths = list(np.linspace(bHO_min, bHO_max, Nsteps, endpoint=True))
    
    InputTaurus.set_inputDDparamsFile(
        **{InputTaurus.InpDDEnum.r_dim : 12,
           InputTaurus.InpDDEnum.omega_dim : 14})
    
    input_args_start = {
        InputTaurus.ArgsEnum.com : 1,
        InputTaurus.ArgsEnum.seed: 5,
        InputTaurus.ArgsEnum.iterations: 1000,
        InputTaurus.ArgsEnum.grad_type: 1,
        InputTaurus.ArgsEnum.grad_tol : 0.001,
        InputTaurus.ArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
        InputTaurus.ArgsEnum.pair_schm: 1,
    }
    
    input_args_onrun = { **input_args_start,
        InputTaurus.ArgsEnum.red_hamil: 1,
        InputTaurus.ArgsEnum.seed: 1,
        InputTaurus.ArgsEnum.iterations: 0,
    } # just get the minimum result
    
    for z, n in nucleus:
        
        summary_results = f'export_HO_TES_z{z}n{n}.txt'
        if summary_results in os.listdir():
            with open(summary_results, 'w+') as f:
                f.write('')
            
        
        for step_, b in enumerate(b_lengths.reverse()):
            
            ## set up the Hamiltonian in the set up 
            hamil_exe = TBME_HamiltonianManager(b, MZmax, MZmin=MZmin)
            hamil_fn_new = f'D1S_t0_z{z}n{n}_MZ{MZmax}_b{1000*b:4.0f}'
            hamil_exe.hamil_filename = hamil_fn_new
            hamil_exe.setAndRun_D1Sxml()
        
            ## input args_for must change seeed=1 after the right minimum
            if step_ > 0:
                input_args_start[InputTaurus.ArgsEnum.seed] = 1
                        
            ## after the export (do not zip the files) import the results and 
            ## copy here to an auxiliary file
            
            ExeTaurus0D_EnergyMinimum.EXPORT_LIST_RESULTS = f"export_{step_}.txt"
        
            try:
                exe_ = ExeTaurus0D_EnergyMinimum(z, n, hamil_fn_new)
                exe_.setInputCalculationArguments(**input_args_start)
                exe_.defineDeformationRange(0,  0, 0)
                exe_.include_header_in_results_file = False
                exe_.setUp()
                exe_.setUpExecution(**input_args_onrun)
                exe_.force_converg = True 
                exe_.run()
                exe_.gobalTearDown(zip_bufolder=False)
            except ExecutionException as e:
                print(e)
            
            line = ''
            with open(ExeTaurus0D_EnergyMinimum.EXPORT_LIST_RESULTS, 'r') as r:
                line = r.readlines()
                if len(line)>1:
                    print("[WARNING], unexpected export data file with multiple lines:\n"
                          '  \n'.join(line))
                    line = line[0]
            if line not in  ('', '\n'):
                with open(summary_results, 'a+') as f:
                    header_  = f"{len(b_lengths)-step_}: {b:5.3f}"
                    header_ += ExeTaurus0D_EnergyMinimum.HEADER_SEPARATOR
                    f.write('\n'+header_+line)
            
            ## rm the intermediate step output
            os.remove(ExeTaurus0D_EnergyMinimum.EXPORT_LIST_RESULTS)












if __name__ == '__main__':
    pass