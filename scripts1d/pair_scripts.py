'''
Created on Jan 19, 2023

@author: Miguel
'''
import os
from datetime import datetime
import shutil

from tools.executors import ExeTaurus1D_PairCoupling, ExecutionException
from tools.inputs import InputTaurus
from scripts1d.script_helpers import getInteractionFile4D1S
from tools.data import DataTaurus
from tools.helpers import LINE_2


def run_pair_surface_D1S(nucleus, interactions, pair_constrs, seed_base=0, 
                         p_min=-0.05, p_max=2.0, N_max=41):
    """
    This method runs for each nucleus all pair constrains given, builds D1S m.e
    Args:
        :nucleus: <list>: (z1,n1), (z2,n2), ..
        :interactions: <dict> [Nucleus (z, n)]: (MZm_max, Mz_min, b_length)
        :pair_constrs = <list> [P_TJ**, P_TJ'**, ...]
    Optional:
        :seed_base (taurus_input seeds, pn-mix True= 0 & 4)
        :p_min
        :p_max
        :N_steps:
    """
    assert all(map(lambda x: x.startswith('P_T'), pair_constrs)), f"invalid pair constraint {pair_constrs}"
    
    ExeTaurus1D_PairCoupling.ITERATIVE_METHOD = \
        ExeTaurus1D_PairCoupling.IterativeEnum.EVEN_STEP_SWEEPING
        
    ExeTaurus1D_PairCoupling.SAVE_DAT_FILES = [
        DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        ]
    ExeTaurus1D_PairCoupling.BLOCKING_SEEDS_RANDOMIZATION = 5
    
    for z, n in nucleus:
        print(LINE_2, f" Starting Pairing Energy Surfaces for Z,N = {z},{n}",
              datetime.now().time(), "\n")
        
        interaction = getInteractionFile4D1S(interactions, z, n)
        if interaction == None or not os.path.exists(interaction+'.sho'):
            print(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        
        InputTaurus.set_inputDDparamsFile(
            **{InputTaurus.InpDDEnum.r_dim : 12,
               InputTaurus.InpDDEnum.omega_dim : 14})
        
        input_args_start = {
            InputTaurus.ArgsEnum.com : 1,
            InputTaurus.ArgsEnum.seed: seed_base,
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
            InputTaurus.ArgsEnum.grad_tol : 0.005,
        }
        
        ExeTaurus1D_PairCoupling.setPairConstraint(pair_constrs[0])
        ExeTaurus1D_PairCoupling.EXPORT_LIST_RESULTS += f"_z{z}n{n}_{interaction}_base.txt"
        
        try:
            exe_ = ExeTaurus1D_PairCoupling(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(p_min,  p_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.gobalTearDown(zip_bufolder=True, _='BASE')
        except ExecutionException as e:
            print("[PAIR_SCRIPT ERROR] :: Execution Exception rose:")
            print(e)
            print("[PAIR_SCRIPT ERROR] Could not preconverge the w.f, skipping isotope.\n")
            continue
        
        for ip, pair_constr in enumerate(pair_constrs):
            
            ExeTaurus1D_PairCoupling.setPairConstraint(pair_constr)
            ExeTaurus1D_PairCoupling.EXPORT_LIST_RESULTS += f"_z{z}n{n}_{interaction}.txt"
            
            try:
                # if ip == 0:
                #     exe_ = ExeTaurus1D_PairCoupling(z, n, interaction)
                # else:
                exe_.resetExecutorObject(keep_1stMinimum=True)
                # input_args_start = input_args_onrun
                
                # exe_.setInputCalculationArguments(**input_args_start)
                exe_.setInputCalculationArguments(input_args_onrun)
                exe_.defineDeformationRange(p_min,  p_max, N_max)
                exe_.setUp()
                exe_.setUpExecution(**input_args_onrun)
                exe_.run()
                exe_.gobalTearDown()
            except ExecutionException as e:
                print(e)
            
            print("End run_pair_surface: ", pair_constr, datetime.now().time(), "\n")
        
        print("End all run_pair_surfaces: ", datetime.now().time(), "\n\n")

