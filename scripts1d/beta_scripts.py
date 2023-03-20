'''
Created on Jan 19, 2023

@author: Miguel
'''
import os

from tools.executors import ExeTaurus1D_DeformQ20, ExecutionException,\
    ExeTaurus1D_DeformB20
from tools.inputs import InputTaurus
from tools.data import DataTaurus
from datetime import datetime
from scripts1d.script_helpers import getInteractionFile4D1S

def run_q20_surface(nucleus, interactions,
                    seed_base=0, ROmega=(10, 10),
                    q_min=-10, q_max=10, N_max=50, convergences=None):
    
    ExeTaurus1D_DeformQ20.ITERATIVE_METHOD = \
        ExeTaurus1D_DeformQ20.IterativeEnum.EVEN_STEP_SWEEPING
        
    ExeTaurus1D_DeformQ20.SAVE_DAT_FILES = DataTaurus.DatFileExportEnum.members()
    # ExeTaurus1D_DeformQ20.SAVE_DAT_FILES = [
    #     DataTaurus.DatFileExportEnum.canonicalbasis,]
    ExeTaurus1D_DeformQ20.SEEDS_RANDOMIZATION = 3
    if convergences != None:
        ExeTaurus1D_DeformQ20.SEEDS_RANDOMIZATION = convergences
        ExeTaurus1D_DeformQ20.GENERATE_RANDOM_SEEDS = True
    
    for z, n in nucleus:
        interaction = getInteractionFile4D1S(interactions, z, n)
        if interaction == None or not os.path.exists(interaction+'.sho'):
            print(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        
        InputTaurus.set_inputDDparamsFile(
            **{InputTaurus.InpDDEnum.r_dim : ROmega[0],
               InputTaurus.InpDDEnum.omega_dim : ROmega[1]})
        
        input_args_start = {
            InputTaurus.ArgsEnum.com : 1,
            InputTaurus.ArgsEnum.seed: seed_base,
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            InputTaurus.ArgsEnum.beta_schm: 0, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
        }
        
        input_args_onrun = {
            InputTaurus.ArgsEnum.red_hamil: 1,
            InputTaurus.ArgsEnum.seed: 1,
            InputTaurus.ArgsEnum.iterations: 600,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.01,
        }
        
        ExeTaurus1D_DeformQ20.EXPORT_LIST_RESULTS = f"export_TESq20_z{z}n{n}_{interaction}.txt"
        
        try:
            exe_ = ExeTaurus1D_DeformQ20(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min,  q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.force_converg = False
            exe_.run()
            exe_.gobalTearDown()
        except ExecutionException as e:
            print(e)
        
    print("End run_q20_surface: ", datetime.now().time())


def run_b20_surface(nucleus, interactions, 
                    seed_base=0, ROmega=(13, 13),
                    q_min=-2.0, q_max=2.0, N_max=41, convergences=None):
    """
    Reqire:
    Args:
        :nucleus: <list>: (z1,n1), (z2,n2), ..
        :interactions: <dict> [Nucleus (z, n)]: (MZm_max, Mz_min, b_length)
    Optional:
        :seed_base (taurus_input seeds, pn-mix True= 0 & 4)
        :ROmega: <tuple>=(R, Omega) grid of Integration (Default is 10, 10)
        :j_min
        :j_max
        :N_steps:
        :convergences: <int> number of random seeds / blocked states to get the global minimum
    """
    
    ExeTaurus1D_DeformB20.ITERATIVE_METHOD = \
        ExeTaurus1D_DeformB20.IterativeEnum.EVEN_STEP_SWEEPING
        
    ExeTaurus1D_DeformB20.SAVE_DAT_FILES = [
        DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    ExeTaurus1D_DeformB20.SEEDS_RANDOMIZATION = 3
    if convergences != None:
        ExeTaurus1D_DeformB20.SEEDS_RANDOMIZATION = convergences
        ExeTaurus1D_DeformB20.GENERATE_RANDOM_SEEDS = True
    
    for z, n in nucleus:
        interaction = getInteractionFile4D1S(interactions, z, n)
        if interaction == None or not os.path.exists(interaction+'.sho'):
            print(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        
        InputTaurus.set_inputDDparamsFile(
            **{InputTaurus.InpDDEnum.r_dim : ROmega[0],
               InputTaurus.InpDDEnum.omega_dim : ROmega[1]})
        
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
            InputTaurus.ArgsEnum.grad_tol : 0.01,
        }
        
        ExeTaurus1D_DeformB20.EXPORT_LIST_RESULTS = f"export_TESb20_z{z}n{n}_{interaction}.txt"
        
        try:
            exe_ = ExeTaurus1D_DeformB20(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min,  q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.force_converg = False
            exe_.run()
            exe_.gobalTearDown()
        except ExecutionException as e:
            print(e)
        
    print("End run_b20_surface: ", datetime.now().time())




