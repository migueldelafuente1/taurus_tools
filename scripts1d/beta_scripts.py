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
from tools.hamiltonianMaker import  TBME_HamiltonianManager
from tools.Enums import GognyEnum
from tools.helpers import printf

def run_q20_surface(nucleus, interactions,
                    seed_base=0, ROmega=(10, 10),
                    q_min=-10, q_max=10, N_max=50, convergences=None,
                    fomenko_points=(1, 1)):
    
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
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            InputTaurus.ArgsEnum.beta_schm: 0, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
        }
        
        input_args_onrun = {
            InputTaurus.ArgsEnum.red_hamil: 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
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
            exe_.globalTearDown()
        except ExecutionException as e:
            printf("[SCRIPT ERROR]:", e)
        
    printf("End run_q20_surface: ", datetime.now().time())


def run_b20_surface(nucleus, interactions,
                    seed_base=0, ROmega=(10, 10),
                    q_min=-1., q_max=1., N_max=50, convergences=None,
                    fomenko_points=(1, 1)):
    
    ExeTaurus1D_DeformB20.ITERATIVE_METHOD = \
        ExeTaurus1D_DeformB20.IterativeEnum.EVEN_STEP_SWEEPING
        
    # ExeTaurus1D_DeformB20.SAVE_DAT_FILES = DataTaurus.DatFileExportEnum.members()
    ExeTaurus1D_DeformQ20.SAVE_DAT_FILES = [
        DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
    ]
    ExeTaurus1D_DeformB20.SEEDS_RANDOMIZATION = 3
    if convergences != None:
        ExeTaurus1D_DeformB20.SEEDS_RANDOMIZATION = convergences
        ExeTaurus1D_DeformB20.GENERATE_RANDOM_SEEDS = True
    
    for z, n in nucleus:
        interaction = getInteractionFile4D1S(interactions, z, n)
        if interaction == None or not os.path.exists(interaction+'.sho'):
            printf(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        
        InputTaurus.set_inputDDparamsFile(
            **{InputTaurus.InpDDEnum.eval_dd   : ROmega != (0, 0),
               InputTaurus.InpDDEnum.r_dim     : ROmega[0],
               InputTaurus.InpDDEnum.omega_dim : ROmega[1]})
        
        axial_calc = seed_base in (2, 3, 9)
        
        input_args_start = {
            InputTaurus.ArgsEnum.com : 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.seed: seed_base,
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            InputTaurus.ArgsEnum.beta_schm: 1,    ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
            'axial_calc' : axial_calc,
        }
        
        input_args_onrun = {
            InputTaurus.ArgsEnum.seed: 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.iterations: 600,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            'axial_calc' : axial_calc,
        }
        
        ExeTaurus1D_DeformB20.EXPORT_LIST_RESULTS = f"export_TESq20_z{z}n{n}_{interaction}.txt"
        
        try:
            exe_ = ExeTaurus1D_DeformB20(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min,  q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.force_converg = False
            exe_.run()
            exe_.globalTearDown()
        except ExecutionException as e:
            printf("[SCRIPT ERROR]:", e)
        
    printf("End run_q20_surface: ", datetime.now().time())


def run_b20_Gogny_surface(nucleus, interactions, gogny_interaction,
                          seed_base=0, ROmega=(13, 13),
                          q_min=-2.0, q_max=2.0, N_max=41, convergences=0,
                          fomenko_points=(1, 1),
                          sym_calc_setup=None,
                          **constr_onrun):
    """
    Reqire:
    Args:
        :nucleus: <list>: (z1,n1), (z2,n2), ..
        :interactions: <dict> [Nucleus (z, n)]: (MZm_max, Mz_min, b_length)
        :gogny_interaction: str from GognyEnum
    Optional:
        :seed_base (taurus_input seeds, pn-mix True= 0 & 4)
        :ROmega: <tuple>=(R, Omega) grid of Integration (Default is 10, 10)
        :j_min
        :j_max
        :N_steps:
        :convergences: <int> number of random seeds / blocked states to get the global minimum
        :fomenko_points: (M protons, M neutron), default is HFB
        :parity_2_block: parity of the states to block
        
        :sym_calc_setup=None: Symmetry for the calculation to use, i.e. certain
                            symmetry restrictions for the constraints for non axial calculuations:
                            _Base1DTaurusExecutor.SymmetryOptionsEnum.NO_CORE_CALC
        :constr_onrun other constraints to set up the calculation.
    """
    if ((fomenko_points[0]>1 or fomenko_points[1]>1) 
        and gogny_interaction != GognyEnum.B1):
        raise ExecutionException(" Projection is not defined for taurus_vap with density-dependent")
    
    ExeTaurus1D_DeformB20.ITERATIVE_METHOD = \
        ExeTaurus1D_DeformB20.IterativeEnum.EVEN_STEP_SWEEPING
        
    ExeTaurus1D_DeformB20.SAVE_DAT_FILES = [
        DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        # DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    
    ExeTaurus1D_DeformB20.SEEDS_RANDOMIZATION   = convergences
    ExeTaurus1D_DeformB20.GENERATE_RANDOM_SEEDS = bool(convergences)
    
    if sym_calc_setup: constr_onrun[sym_calc_setup] = True
    
    for z, n in nucleus:
        interaction = getInteractionFile4D1S(interactions, z, n, 
                                             gogny_interaction=gogny_interaction)
        if interaction == None or not os.path.exists(interaction+'.sho'):
            printf(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        
        InputTaurus.set_inputDDparamsFile(
            **{InputTaurus.InpDDEnum.eval_dd   : ROmega != (0, 0),
               InputTaurus.InpDDEnum.r_dim     : ROmega[0],
               InputTaurus.InpDDEnum.omega_dim : ROmega[1]})
        
        axial_calc = seed_base in (2, 3, 9)
        
        input_args_start = {
            InputTaurus.ArgsEnum.com : 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.seed: seed_base,
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            InputTaurus.ArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
            'axial_calc' : axial_calc,
            **constr_onrun
        }
        
        input_args_onrun = {
            InputTaurus.ArgsEnum.red_hamil: 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.seed: 1,
            InputTaurus.ArgsEnum.iterations: 600,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.01,
            'axial_calc' : axial_calc,
            **constr_onrun
        }
        
        ExeTaurus1D_DeformB20.EXPORT_LIST_RESULTS = f"export_TESb20_z{z}n{n}_{interaction}.txt"
        
        try:
            exe_ = ExeTaurus1D_DeformB20(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min, q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.force_converg = False
            exe_.run()
            exe_.globalTearDown()
        except ExecutionException as e:
            printf("[SCRIPT ERROR]:", e)
        
    printf("End run_b20_surface: ", datetime.now().time())


def run_b20_composedInteraction(nucleus, interactions, interaction_runnable,
                                seed_base=0,
                                q_min=-2.0, q_max=2.0, N_max=41, convergences=0,
                                fomenko_points=(1, 1)):
    """
    Reqire:
    Args:
        :nucleus: <list>: (z1,n1), (z2,n2), ..
        :interactions: <dict> [Nucleus (z, n)]: (MZm_max, Mz_min, b_length)
        :interaction_runnable: <list [(XMLsetter, kwargs for it)] 
            (see examples at tools/HamiltonianMaker.py in __main__ section)
    Optional:
        :seed_base (taurus_input seeds, pn-mix True= 0 & 4)
        :j_min
        :j_max
        :N_steps:
        :convergences: <int> number of random seeds / blocked states to get the global minimum
        :fomenko_points: (M protons, M neutron), default is HFB
        :parity_2_block: parity of the states to block
    """
    
    
    ExeTaurus1D_DeformB20.ITERATIVE_METHOD = \
        ExeTaurus1D_DeformB20.IterativeEnum.EVEN_STEP_SWEEPING
        
    ExeTaurus1D_DeformB20.SAVE_DAT_FILES = [
        # DataTaurus.DatFileExportEnum.canonicalbasis,
        # DataTaurus.DatFileExportEnum.eigenbasis_h,
        # DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    
    ExeTaurus1D_DeformB20.SEEDS_RANDOMIZATION   = convergences
    ExeTaurus1D_DeformB20.GENERATE_RANDOM_SEEDS = bool(convergences)
    
    #===========================================================================
    # INTERACTION DEFINITION  (Examples in HAMILTONIAN-MANAGER)
    # interaction_runable = []
    # kwargs = {CentralMEParameters.potential: PotentialForms.Gaussian,
    #           CentralMEParameters.constant : -100.,
    #           CentralMEParameters.mu_length: 1.4,}
    # interaction_runable.append((TBMEXML_Setter.set_central_force, kwargs ))
    #
    # kwargs = {CentralMEParameters.constant: 50.3,
    #           CentralMEParameters.n_power:  2,}
    # interaction_runable.append((TBMEXML_Setter.set_quadrupole_force, kwargs ))
    
    #===========================================================================    
    for z, n in nucleus:

        exe_ = TBME_HamiltonianManager(*interactions[(z, n)], set_com2=True)
        exe_.setAndRun_ComposeInteractions(interaction_runnable)
        interaction = exe_.hamil_filename
        
        if interaction == None or not os.path.exists(interaction+'.sho'):
            printf(f"Interaction not found for (z,n)=({z},{n}), Continue.")
            continue
        axial_calc = seed_base in (2, 3, 9)
        
        input_args_start = {
            InputTaurus.ArgsEnum.com : 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.seed: seed_base,
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            InputTaurus.ArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            'axial_calc' : axial_calc,
        }
        
        input_args_onrun = {
            InputTaurus.ArgsEnum.red_hamil: 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.seed: 1,
            InputTaurus.ArgsEnum.iterations: 600,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.01,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            'axial_calc' : axial_calc,
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
            exe_.globalTearDown()
        except ExecutionException as e:
            printf("[SCRIPT ERROR]:", e)
        
    printf("End run_b20_surface: ", datetime.now().time())

