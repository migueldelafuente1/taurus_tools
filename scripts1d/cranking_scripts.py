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
from tools.helpers import printf

def run_J_surface(nucleus, interactions, J_i, 
                  seed_base=0, ROmega=(13, 13),
                  j_min= 0.0, j_max=25.0, N_max=50, convergences=0,
                  sym_calc_setup=None,
                  **constr_onrun):
    """
    Reqire:
    Args:
        :nucleus: <list>: (z1,n1), (z2,n2), ..
        :interactions: <dict> [Nucleus (z, n)]: (MZm_max, Mz_min, b_length)
        :J_i:  from DataTaurus.ConstrEnum.J*
    Optional:
        :seed_base (taurus_input seeds, pn-mix True= 0 & 4)
        :ROmega: <tuple>=(R, Omega) grid of Integration (Default is 10, 10)
        :j_min
        :j_max
        :N_steps: 
        :convergences: <int> number of random seeds / blocked states to get the global minimum
        :constr_onrun other constraints to set up the calculation.
        :sym_calc_setup=None: Symmetry for the calculation to use, i.e. certain
                            symmetry restrictions for the constraints for non axial calculuations:
                            _Base1DTaurusExecutor.SymmetryOptionsEnum.NO_CORE_CALC
    """
    assert J_i.startswith('J') and J_i in InputTaurus.ConstrEnum.members(), \
        "Script only accepts Jx, Jy, Jz"
    
    ExeTaurus1D_AngMomentum.ITERATIVE_METHOD = \
        ExeTaurus1D_AngMomentum.IterativeEnum.EVEN_STEP_SWEEPING
        
    ExeTaurus1D_AngMomentum.SAVE_DAT_FILES = [
        DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,        
        DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    
    ExeTaurus1D_AngMomentum.SEEDS_RANDOMIZATION   = convergences
    ExeTaurus1D_AngMomentum.GENERATE_RANDOM_SEEDS = bool(convergences)
    
    if sym_calc_setup: constr_onrun[sym_calc_setup] = True
    
    for z, n in nucleus:
        interaction = getInteractionFile4D1S(interactions, z, n)
        if interaction == None or not os.path.exists(interaction+'.sho'):
            printf(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        
        InputTaurus.set_inputDDparamsFile(
            **{InputTaurus.InpDDEnum.eval_dd    : ROmega != (0, 0),
               InputTaurus.InpDDEnum.r_dim      : ROmega[0],
               InputTaurus.InpDDEnum.omega_dim  : ROmega[1]})
        
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
        
        ExeTaurus1D_AngMomentum.setAngularMomentumConstraint(J_i)        
        ExeTaurus1D_AngMomentum.EXPORT_LIST_RESULTS += f"_z{z}n{n}_{interaction}.txt"
        
        try:
            exe_ = ExeTaurus1D_AngMomentum(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(j_min, j_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.force_converg = True
            exe_.run()
            exe_.globalTearDown()
        except ExecutionException as e:
            printf(e)
        
    printf("End run_J_surface: ", datetime.now().time())


