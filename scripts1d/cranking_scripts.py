'''
Created on Jan 19, 2023

@author: Miguel
'''
import os
from datetime import datetime
import shutil

from tools.executors import ExecutionException, ExeTaurus1D_AngMomentum
from tools.inputs import InputTaurus
from scripts1d.script_helpers import getInteractionFile4D1S
from tools.data import DataTaurus

def run_J_surface(nucleus, interactions, J_i, 
                  seed_base=0, q_min= 0.0, q_max=25.0, N_max=50):
    """
    Reqire:
        Args:
        nucleus
        interactions
        J_i  from DataTaurus.ConstrEnum.J*
        deform array for linspace
    """
    assert J_i.startswith('J') and J_i in InputTaurus.ConstrEnum.members(), \
        "Script only accepts Jx, Jy, Jz"
    
    ExeTaurus1D_AngMomentum.ITERATIVE_METHOD = \
        ExeTaurus1D_AngMomentum.IterativeEnum.EVEN_STEP_SWEEPING
        
    ExeTaurus1D_AngMomentum.SAVE_DAT_FILES = [
        DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        ]
    
    for z, n in nucleus:
        interaction = getInteractionFile4D1S(interactions, z, n)
        if interaction == None or not os.path.exists(interaction+'.sho'):
            print(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        
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
        
        input_args_onrun = {
            InputTaurus.ArgsEnum.red_hamil: 1,
            InputTaurus.ArgsEnum.seed: 1,
            InputTaurus.ArgsEnum.iterations: 600,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.01,
        }
        
        ExeTaurus1D_AngMomentum.setAngularMomentumConstraint(J_i)        
        ExeTaurus1D_AngMomentum.EXPORT_LIST_RESULTS += f"_z{z}n{n}_{interaction}.txt"
        
        try:
            exe_ = ExeTaurus1D_AngMomentum(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min,  q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.run()
            exe_.gobalTearDown()
        except ExecutionException as e:
            print(e)
        
    print("End run_b20_surface: ", datetime.now().time())


