'''
Created on Oct 25, 2024

@author: Miguel
'''
import os
from datetime import datetime
import shutil
from pathlib import Path

from tools.base_executors import SetUpStoredWFAndHamiltonian
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
                         convergences=0, valid_Ks_to_block=[],
                         fomenko_points=(1, 1), sym_calc_setup=None,
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
        :fomenko_points: (<int>, <int>)
        :valid_Ks_to_block: [list, int] Usage:
            1. for seed_base != 1: is used to stablish the valid Ks for axial 
                                    odd-nuclei calculations
            2. for seed_base == 1: imports the K from some folder, non axial 
                                    odd calculations requires to put [0,]
        :axial_calc:      <bool>
    """
    vap_ = sum(fomenko_points) > 2
    if ((fomenko_points[0]>1 or fomenko_points[1]>1) 
        and gogny_interaction != GognyEnum.B1):
        raise ExecutionException(" Projection is not defined for taurus_vap with density-dependent")
    
    assert all(map(lambda x: x.startswith('P_T'), pair_constrs)), f"invalid pair constraint {pair_constrs}"
    
    ## Note: modification to exclude the 3d components of the isoscalar channel
    _3d_PT0JM = {}
    if not InputTaurus.ConstrEnum.P_T00_J1m1 in pair_constrs:
        _3d_PT0JM[InputTaurus.ConstrEnum.P_T00_J1m1] = 0.0
    if not InputTaurus.ConstrEnum.P_T00_J1p1 in pair_constrs:
        _3d_PT0JM[InputTaurus.ConstrEnum.P_T00_J1p1] = 0.0
    constr_onrun = {**constr_onrun, **_3d_PT0JM}
    
    if sym_calc_setup: constr_onrun[sym_calc_setup] = True
    
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
    ExeTaurus2D_MultiConstrained.DO_BASE_CALCULATION   = (convergences > 0) and (seed_base != 1)
    
    for z, n in nucleus:
        printf(LINE_2, f" Starting Pairing Energy Surfaces for Z,N = {z},{n}",
              datetime.now().time(), "\n")
        
        if seed_base == 1: 
            default_K = valid_Ks_to_block[0] if valid_Ks_to_block else 0
            kwargs = {'K': default_K if 1 in (z%2, n%2) else 0, }
            interaction = SetUpStoredWFAndHamiltonian.copyWFAndHamil(z,n,**kwargs)
        else:
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
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.seed: seed_base,
            InputTaurus.ArgsEnum.iterations: 1000 if not vap_ else 7000,
            InputTaurus.ArgsEnum.grad_type: 2 if 1 in (z%2, n%2) else 1,
            InputTaurus.ArgsEnum.grad_tol : 0.002,
            InputTaurus.ArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
            **constr_onrun
        }
        
        input_args_onrun = {
            InputTaurus.ArgsEnum.red_hamil: 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.seed: 1,
            InputTaurus.ArgsEnum.iterations: 2500 if not vap_ else 7000,
            InputTaurus.ArgsEnum.grad_type: 2 if 1 in (z%2, n%2) else 1,
            InputTaurus.ArgsEnum.grad_tol : 0.002,
            **constr_onrun
        }
        
        ExeTaurus2D_MultiConstrained.setExecutorConstraints(pair_constrs)
        ExeTaurus2D_MultiConstrained.updateTotalKForOddNuclei(valid_Ks_to_block)
        ExeTaurus2D_MultiConstrained.EXPORT_LIST_RESULTS += f"_z{z}n{n}_{interaction}.txt"
        
        _constr_keys  = [x.replace('_','') for x in pair_constrs]
        ## First unconstrained minimum
        try:
            exe_ = ExeTaurus2D_MultiConstrained(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(pair_constrs)
            exe_.setUp(*_constr_keys)
            exe_.setUpExecution(**input_args_onrun)
            exe_.globalTearDown(zip_bufolder=True, base_calc=True)
        except ExecutionException as e:
            printf("[2D_SCRIPT ERROR] :: Execution Exception rose:")
            printf(e)
            printf("[2D_SCRIPT ERROR] Could not preconverge the w.f, skipping isotope.\n")
            continue
                        
        exe_.setInputCalculationArguments(**input_args_onrun)
        exe_.force_converg = True
        exe_.run()
        exe_.globalTearDown()
                
        printf("End all run_pair_surfaces: ", datetime.now().time(), "\n\n")
