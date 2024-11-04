'''
Created on Oct 25, 2024

@author: Miguel
'''
import os
from datetime import datetime
import shutil

from tools.executors   import ExecutionException
from tools.executors2D import ExeTaurus2D_MultiConstrained
from tools.inputs      import InputTaurus
from scripts1d.script_helpers import getInteractionFile4D1S
from tools.data    import DataTaurus
from tools.helpers import LINE_2, prettyPrintDictionary, printf
from tools.Enums   import GognyEnum

def run_pair_surfaces_2d(nucleus, interactions, pair_constrs,
                         gogny_interaction=GognyEnum.D1S,
                         seed_base=0, ROmega=(13, 13),
                         convergences=0,
                         **constr_onrun):
    """
    This method runs for each nucleus all pair constrains given, builds D1S m.e
    Args:
        :nucleus: <list>: (z1,n1), (z2,n2), ..
        :interactions: <dict> [Nucleus (z, n)]: (MZm_max, Mz_min, b_length)
        :pair_constrs = <dict> 
                            {P_TJ**: (p_min, p_max, N_steps), P_TJ'**, ... }
    Optional:
        :seed_base (taurus_input seeds, pn-mix True= 0 & 4)
        :ROmega tuple of the integration grids
        :convergences: <int> number of random seeds / blocked states to get the global minimum
        :constr_onrun other constraints to set up the calculation.
    """
    assert all(map(lambda x: x.startswith('P_T'), pair_constrs)), f"invalid pair constraint {pair_constrs}"
    
    ## Note: modification to exclude the 3d components of the isoscalar channel
    _3d_PT0JM = {}
    if not InputTaurus.ConstrEnum.P_T00_J1m1 in pair_constrs:
        _3d_PT0JM[InputTaurus.ConstrEnum.P_T00_J1m1] = 0.0
    if not InputTaurus.ConstrEnum.P_T00_J1p1 in pair_constrs:
        _3d_PT0JM[InputTaurus.ConstrEnum.P_T00_J1p1] = 0.0
    constr_onrun = {**constr_onrun, **_3d_PT0JM}
    
    ## Normal execution.
    ExeTaurus2D_MultiConstrained.ITERATIVE_METHOD = \
        ExeTaurus2D_MultiConstrained.IterativeEnum.EVEN_STEP_SWEEPING
        
    ExeTaurus2D_MultiConstrained.SAVE_DAT_FILES = [
        DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    
    ExeTaurus2D_MultiConstrained.SEEDS_RANDOMIZATION   = convergences
    ExeTaurus2D_MultiConstrained.GENERATE_RANDOM_SEEDS = bool(convergences)
    
    for z, n in nucleus:
        printf(LINE_2, f" Starting Pairing Energy Surfaces for Z,N = {z},{n}",
              datetime.now().time(), "\n")
        
        interaction = getInteractionFile4D1S(interactions, z, n, 
                                             gogny_interaction=gogny_interaction)
        if interaction == None or not os.path.exists(interaction+'.sho'):
            printf(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        
        InputTaurus.set_inputDDparamsFile(
            **{InputTaurus.InpDDEnum.eval_dd   : ROmega != (0, 0),
               InputTaurus.InpDDEnum.r_dim     : ROmega[0],
               InputTaurus.InpDDEnum.omega_dim : ROmega[1]})
        
        input_args_start = {
            InputTaurus.ArgsEnum.com : 1,
            InputTaurus.ArgsEnum.seed: seed_base,
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            InputTaurus.ArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
            **constr_onrun
        }
        
        input_args_onrun = {
            InputTaurus.ArgsEnum.red_hamil: 1,
            InputTaurus.ArgsEnum.seed: 1,
            InputTaurus.ArgsEnum.iterations: 600,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.005,
            **constr_onrun
        }
        
        ExeTaurus2D_MultiConstrained.setExecutorConstraints(pair_constrs)
        ExeTaurus2D_MultiConstrained.EXPORT_LIST_RESULTS += f"_z{z}n{n}_{interaction}.txt"
        
        ## First unconstrained minimum
        try:
            exe_ = ExeTaurus2D_MultiConstrained(z, n, interaction)
            exe_.setInputCalculationArguments(axial_calc=True, **input_args_start)
            exe_.defineDeformationRange(pair_constrs)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.globalTearDown(zip_bufolder=True, base_calc=True)
        except ExecutionException as e:
            printf("[2D_SCRIPT ERROR] :: Execution Exception rose:")
            printf(e)
            printf("[2D_SCRIPT ERROR] Could not preconverge the w.f, skipping isotope.\n")
            continue
                        
        exe_.setInputCalculationArguments(axial_calc=True, **input_args_onrun)
        exe_.force_converg = True
        exe_.run()
        exe_.globalTearDown()
                
        printf("End all run_pair_surfaces: ", datetime.now().time(), "\n\n")
