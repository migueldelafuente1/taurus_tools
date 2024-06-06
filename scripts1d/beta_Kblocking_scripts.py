'''
Created on 3 jun 2024

@author: delafuente

'''
import os
from tools.executors import ExecutionException
from tools.inputs import InputTaurus, InputTaurusPAV
from tools.data import DataTaurus
from datetime import datetime
from scripts1d.script_helpers import getInteractionFile4D1S
from tools.Enums import GognyEnum
from tools.exec_blocking_Kprojections import \
    ExeTaurus1D_B20_OEblocking_Ksurfaces, ExeTaurus1D_B20_KMixing_OEblocking, \
    ExeTaurus1D_B20_Ksurface_Base
from tools.helpers import importAndCompile_taurus, printf



def run_b20_FalseOE_Kprojections_Gogny(nucleus, interactions, gogny_interaction,
                          seed_base=0, ROmega=(13, 13),
                          q_min=-2.0, q_max=2.0, N_max=41, convergences=0,
                          fomenko_points=(1, 1), 
                          parity_2_block= 1, ):
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
    """
    if ((fomenko_points[0]>1 or fomenko_points[1]>1) 
        and gogny_interaction != GognyEnum.B1):
        raise ExecutionException(" Projection is not defined for taurus_vap with density-dependent")
    
    ExeTaurus1D_B20_OEblocking_Ksurfaces.IGNORE_SEED_BLOCKING  = True
    ExeTaurus1D_B20_OEblocking_Ksurfaces.BLOCK_ALSO_NEGATIVE_K = False
    
    ExeTaurus1D_B20_OEblocking_Ksurfaces.ITERATIVE_METHOD = \
        ExeTaurus1D_B20_OEblocking_Ksurfaces.IterativeEnum.EVEN_STEP_STD
        
    ExeTaurus1D_B20_OEblocking_Ksurfaces.SAVE_DAT_FILES = [
        # DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        # DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    ExeTaurus1D_B20_OEblocking_Ksurfaces.SEEDS_RANDOMIZATION   = convergences
    ExeTaurus1D_B20_OEblocking_Ksurfaces.GENERATE_RANDOM_SEEDS = bool(convergences)
    ExeTaurus1D_B20_OEblocking_Ksurfaces.PARITY_TO_BLOCK       = parity_2_block
    
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
            InputTaurus.ArgsEnum.z_Mphi : 0,
            InputTaurus.ArgsEnum.n_Mphi : 0,
            InputTaurus.ArgsEnum.seed: seed_base,
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.grad_tol : 0.001,
            InputTaurus.ArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
        }        
        input_args_onrun = {
            InputTaurus.ArgsEnum.red_hamil: 1,
            InputTaurus.ArgsEnum.z_Mphi : fomenko_points[0],
            InputTaurus.ArgsEnum.n_Mphi : fomenko_points[1],
            InputTaurus.ArgsEnum.seed: 1,
            InputTaurus.ArgsEnum.iterations: 1000,
            InputTaurus.ArgsEnum.grad_type: 1,
            InputTaurus.ArgsEnum.eta_grad : 0.015,
            InputTaurus.ArgsEnum.mu_grad  : 0.02, # 0.5
            InputTaurus.ArgsEnum.grad_tol : 0.01,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
        }        
        ExeTaurus1D_B20_OEblocking_Ksurfaces.EXPORT_LIST_RESULTS = \
            f"export_TESb20_z{z}n{n}_{interaction}.txt"
        try:
            exe_ = ExeTaurus1D_B20_OEblocking_Ksurfaces(z, n, interaction)
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

def run_b20_FalseOE_Block1KAndPAV(nucleus, interactions, gogny_interaction, K,
                                  seed_base=0, ROmega=(13, 13),
                                  q_min=-2.0, q_max=2.0, N_max=41, 
                                  convergences=0, fomenko_points=(1, 1), 
                                  parity_2_block= 1, 
                                  preconverge_blocking_sts=False,
                                  find_Kfor_all_sps=True):
    
    """
        This script evaluate the projection after the blocking from a previous
        false- odd-even b20 TES.
    Reqires:
    Args:
        :nucleus: <list>: (z1,n1), (z2,n2), ..
        :interactions: <dict> [Nucleus (z, n)]: (MZm_max, Mz_min, b_length)
        :gogny_interaction: str from GognyEnum
        :K <int> 2*K, only value that will be blocked.
    Optional:
        :seed_base (taurus_input seeds, pn-mix True= 0 & 4)
        :ROmega: <tuple>=(R, Omega) grid of Integration (Default is 10, 10)
        :j_min
        :j_max
        :N_steps:
        
        :convergences: <int> number of random seeds / blocked states to get the global minimum
        :fomenko_points: (M protons, M neutron), default is HFB
        :parity_2_block: parity of the states to block
        :preconverge_blocking_sts <int> = 0  :fully converge, > 0 -> number of steps
        :find_Kfor_all_sps =True, evaluate all valid sps (recomended but slower)
    """
    if ((fomenko_points[0]>1 or fomenko_points[1]>1) 
        and gogny_interaction != GognyEnum.B1):
        raise ExecutionException(" Projection is not defined for taurus_vap with density-dependent")
    
    ExeTaurus1D_B20_KMixing_OEblocking.IGNORE_SEED_BLOCKING  = True
    ExeTaurus1D_B20_KMixing_OEblocking.BLOCK_ALSO_NEGATIVE_K = False
    ExeTaurus1D_B20_KMixing_OEblocking.RUN_PROJECTION        = True 
    ExeTaurus1D_B20_KMixing_OEblocking.FIND_K_FOR_ALL_SPS    = find_Kfor_all_sps
        
    ExeTaurus1D_B20_KMixing_OEblocking.ITERATIVE_METHOD = \
        ExeTaurus1D_B20_KMixing_OEblocking.IterativeEnum.EVEN_STEP_STD
        
    ExeTaurus1D_B20_KMixing_OEblocking.SAVE_DAT_FILES = [
        # DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        # DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    ExeTaurus1D_B20_KMixing_OEblocking.SEEDS_RANDOMIZATION   = convergences
    ExeTaurus1D_B20_KMixing_OEblocking.GENERATE_RANDOM_SEEDS = bool(convergences)
    ExeTaurus1D_B20_KMixing_OEblocking.DO_BASE_CALCULATION   = convergences >= 0
    ExeTaurus1D_B20_KMixing_OEblocking.PARITY_TO_BLOCK       = parity_2_block
    
    ExeTaurus1D_B20_KMixing_OEblocking.FULLY_CONVERGE_BLOCKING_ITER_MODE  = \
        preconverge_blocking_sts in (0, False)
    ExeTaurus1D_B20_KMixing_OEblocking.PRECONVERNGECE_BLOCKING_ITERATIONS = \
        preconverge_blocking_sts
    
    if ExeTaurus1D_B20_KMixing_OEblocking.RUN_PROJECTION: 
        ## Import the programs if they do not exist
        importAndCompile_taurus(pav= not os.path.exists(InputTaurusPAV.PROGRAM))
    
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
        
        ## NOTE: fomenko_points are not used for false-oe 1st minimization
        ##       we store them to be used in the run process to fix them after
        ##       the false minimization is done, apply afterwards for blocking
        IArgsEnum = InputTaurus.ArgsEnum
        vap_args = {IArgsEnum.z_Mphi : 0, IArgsEnum.n_Mphi : 0,}
        if ((z+n) % 2 == 0 and n % 2 == 0):
            ## in case of even-even one can do the VAP
            vap_args = {IArgsEnum.z_Mphi : fomenko_points[0],
                        IArgsEnum.n_Mphi : fomenko_points[1],}
        
        input_args_start = {** vap_args,
            IArgsEnum.com : 1,
            IArgsEnum.seed: seed_base,
            IArgsEnum.iterations: 1000,
            IArgsEnum.grad_type: 1,
            IArgsEnum.grad_tol : 0.0001,
            IArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            IArgsEnum.pair_schm: 1,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
        }
        input_args_onrun = {**vap_args, 
            IArgsEnum.red_hamil: 1,
            IArgsEnum.seed: 1,
            IArgsEnum.iterations: 1000,
            IArgsEnum.grad_type: 1,
            IArgsEnum.eta_grad : 0.015,
            IArgsEnum.mu_grad  : 0.02, # 0.5
            IArgsEnum.grad_tol : 0.0001,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
            'valid_Ks'   : [K, ] 
        }
        input_args_projection = {
            InputTaurusPAV.ArgsEnum.red_hamil : 1,
            InputTaurusPAV.ArgsEnum.alpha : 10,
            InputTaurusPAV.ArgsEnum.beta  : 20,
            InputTaurusPAV.ArgsEnum.gamma : 10,
            InputTaurusPAV.ArgsEnum.empty_states : 0,
            InputTaurusPAV.ArgsEnum.disable_simplifications_P : 0,
            InputTaurusPAV.ArgsEnum.j_min : K, 
            InputTaurusPAV.ArgsEnum.cutoff_overlap : 1.0e-10,
            # PN-PAV and J bound arguments set by the program, P-PAV = no
        }
        ExeTaurus1D_B20_KMixing_OEblocking.EXPORT_LIST_RESULTS = \
            f"export_TESb20_z{z}n{n}_{interaction}.txt"
        try:
            exe_ = ExeTaurus1D_B20_KMixing_OEblocking(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min, q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.setUpProjection(**input_args_projection)
            exe_.force_converg = False
            exe_.run(fomenko_points)
            exe_.globalTearDown()
        except ExecutionException as e:
            printf("[SCRIPT ERROR]:", e)
        
    printf("End run_b20_surface k-mixing: ", datetime.now().time())
    
def run_b20_FalseOE_Kmixing(nucleus, interactions, gogny_interaction,
                            valid_Ks = [], 
                            seed_base=0, ROmega=(13, 13),
                            q_min=-2.0, q_max=2.0, N_max=41, convergences=0,
                            fomenko_points=(1, 1), 
                            parity_2_block= 1, preconverge_blocking_sts=False,
                            find_Kfor_all_sps= True,
                            ):
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
        :preconverge_blocking_states <int> = 0  :fully converge, > 0 -> number of steps
        :find_Kfor_all_sps =True, evaluate all valid sps (recomended but slower)
    """
    if ((fomenko_points[0]>1 or fomenko_points[1]>1) 
        and gogny_interaction != GognyEnum.B1):
        raise ExecutionException(" Projection is not defined for taurus_vap with density-dependent")
    
    ExeTaurus1D_B20_KMixing_OEblocking.IGNORE_SEED_BLOCKING  = True
    ExeTaurus1D_B20_KMixing_OEblocking.BLOCK_ALSO_NEGATIVE_K = False
    ExeTaurus1D_B20_KMixing_OEblocking.RUN_PROJECTION        = True 
    ExeTaurus1D_B20_KMixing_OEblocking.FIND_K_FOR_ALL_SPS    = find_Kfor_all_sps
    
    ExeTaurus1D_B20_KMixing_OEblocking.ITERATIVE_METHOD = \
        ExeTaurus1D_B20_KMixing_OEblocking.IterativeEnum.EVEN_STEP_STD
        
    ExeTaurus1D_B20_KMixing_OEblocking.SAVE_DAT_FILES = [
        # DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        # DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    ExeTaurus1D_B20_KMixing_OEblocking.SEEDS_RANDOMIZATION   = convergences
    ExeTaurus1D_B20_KMixing_OEblocking.GENERATE_RANDOM_SEEDS = bool(convergences)
    ExeTaurus1D_B20_KMixing_OEblocking.DO_BASE_CALCULATION   = convergences >= 0
    ExeTaurus1D_B20_KMixing_OEblocking.PARITY_TO_BLOCK       = parity_2_block
    
    ExeTaurus1D_B20_KMixing_OEblocking.FULLY_CONVERGE_BLOCKING_ITER_MODE  = \
        preconverge_blocking_sts in (0, False)
    ExeTaurus1D_B20_KMixing_OEblocking.PRECONVERNGECE_BLOCKING_ITERATIONS = \
        preconverge_blocking_sts
    
    if ExeTaurus1D_B20_KMixing_OEblocking.RUN_PROJECTION: 
        ## Import the programs if they do not exist
        importAndCompile_taurus(pav= not os.path.exists(InputTaurusPAV.PROGRAM))
    
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
        
        ## Note: For false o-e projection, fomenko_points are given in run()
        ##          see Note in script run_b20_FalseOE_Block1KAndPAV()
        IArgsEnum = InputTaurus.ArgsEnum
        vap_args = {IArgsEnum.z_Mphi : 0, IArgsEnum.n_Mphi : 0,}
        if ((z+n) % 2 == 0 and n % 2 == 0 and not 1 in (z, n)):
            ## in case of even-even one can do the VAP
            vap_args = {IArgsEnum.z_Mphi : fomenko_points[0],
                        IArgsEnum.n_Mphi : fomenko_points[1],}
        
        input_args_start = {**vap_args,
            IArgsEnum.com : 1,
            IArgsEnum.seed: seed_base,
            IArgsEnum.iterations: 1500,
            IArgsEnum.grad_type: 1,
            IArgsEnum.grad_tol : 0.0001,
            IArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            IArgsEnum.pair_schm: 1,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
        }
        input_args_onrun = {**vap_args, 
            IArgsEnum.red_hamil: 1,
            IArgsEnum.seed: 1,
            IArgsEnum.iterations: 1500,
            IArgsEnum.grad_type: 1,
            IArgsEnum.eta_grad : 0.015,
            IArgsEnum.mu_grad  : 0.02, # 0.5
            IArgsEnum.grad_tol : 0.0001,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
            'valid_Ks'   : valid_Ks,
        }
        input_args_projection = {
            InputTaurusPAV.ArgsEnum.red_hamil : 1,
            InputTaurusPAV.ArgsEnum.alpha : 10,
            InputTaurusPAV.ArgsEnum.beta  : 20,
            InputTaurusPAV.ArgsEnum.gamma : 10,
            InputTaurusPAV.ArgsEnum.empty_states : 0,
            InputTaurusPAV.ArgsEnum.disable_simplifications_P : 0,
            InputTaurusPAV.ArgsEnum.cutoff_overlap : 1.0e-10,
            # PN-PAV and J bound arguments set by the program, P-PAV = no
        }
        ExeTaurus1D_B20_KMixing_OEblocking.EXPORT_LIST_RESULTS = \
            f"export_TESb20_z{z}n{n}_{interaction}.txt"
        try:
            exe_ = ExeTaurus1D_B20_KMixing_OEblocking(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min, q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.setUpProjection(**input_args_projection)
            exe_.force_converg = False
            exe_.run(fomenko_points)
            exe_.globalTearDown()
        except ExecutionException as e:
            printf("[SCRIPT ERROR]:", e)
        
    printf("End run_b20_surface k-mixing: ", datetime.now().time())
    
def run_b20_FalseOE_Kmixing_exampleSingleJ(nucleus, interactions, gogny_interaction,
                            valid_Ks = [], 
                            seed_base=0, ROmega=(13, 13),
                            q_min=-2.0, q_max=2.0, N_max=41, convergences=0,
                            fomenko_points=(1, 1), 
                            parity_2_block= 1, preconverge_blocking_sts=False,
                            find_Kfor_all_sps= True,
                            ):
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
        :preconverge_blocking_states <int> = 0  :fully converge, > 0 -> number of steps
        :find_Kfor_all_sps =True, evaluate all valid sps (recomended but slower)
    """
    if ((fomenko_points[0]>1 or fomenko_points[1]>1) 
        and gogny_interaction != GognyEnum.B1):
        raise ExecutionException(" Projection is not defined for taurus_vap with density-dependent")
    
    ExeTaurus1D_B20_KMixing_OEblocking.IGNORE_SEED_BLOCKING  = False
    ExeTaurus1D_B20_KMixing_OEblocking.BLOCK_ALSO_NEGATIVE_K = False
    ExeTaurus1D_B20_KMixing_OEblocking.RUN_PROJECTION        = True 
    ExeTaurus1D_B20_KMixing_OEblocking.FIND_K_FOR_ALL_SPS    = find_Kfor_all_sps
    
    ExeTaurus1D_B20_KMixing_OEblocking.ITERATIVE_METHOD = \
        ExeTaurus1D_B20_KMixing_OEblocking.IterativeEnum.EVEN_STEP_STD
        
    ExeTaurus1D_B20_KMixing_OEblocking.SAVE_DAT_FILES = [
        # DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        # DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    ExeTaurus1D_B20_KMixing_OEblocking.SEEDS_RANDOMIZATION   = convergences
    ExeTaurus1D_B20_KMixing_OEblocking.GENERATE_RANDOM_SEEDS = bool(convergences)
    ExeTaurus1D_B20_KMixing_OEblocking.DO_BASE_CALCULATION   = convergences >= 0
    ExeTaurus1D_B20_KMixing_OEblocking.PARITY_TO_BLOCK       = parity_2_block
    
    ExeTaurus1D_B20_KMixing_OEblocking.FULLY_CONVERGE_BLOCKING_ITER_MODE  = \
        preconverge_blocking_sts in (0, False)
    ExeTaurus1D_B20_KMixing_OEblocking.PRECONVERNGECE_BLOCKING_ITERATIONS = \
        preconverge_blocking_sts
    
    if ExeTaurus1D_B20_KMixing_OEblocking.RUN_PROJECTION: 
        ## Import the programs if they do not exist
        importAndCompile_taurus(pav= not os.path.exists(InputTaurusPAV.PROGRAM))
    
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
        
        ## Note: For false o-e projection, fomenko_points are given in run()
        ##          see Note in script run_b20_FalseOE_Block1KAndPAV()
        IArgsEnum = InputTaurus.ArgsEnum
        vap_args = {IArgsEnum.z_Mphi : 0, IArgsEnum.n_Mphi : 0,}
        if ((z+n) % 2 == 0 and n % 2 == 0 and not 1 in (z, n)):
            ## in case of even-even one can do the VAP
            vap_args = {IArgsEnum.z_Mphi : fomenko_points[0],
                        IArgsEnum.n_Mphi : fomenko_points[1],}
        
        input_args_start = {**vap_args,
            IArgsEnum.com : 1,
            IArgsEnum.seed: seed_base,
            IArgsEnum.iterations: 1500,
            IArgsEnum.grad_type: 1,
            IArgsEnum.grad_tol : 0.0001,
            IArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            IArgsEnum.pair_schm: 1,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
        }
        input_args_onrun = {**vap_args, 
            IArgsEnum.red_hamil: 1,
            IArgsEnum.seed: 1,
            IArgsEnum.iterations: 1500,
            IArgsEnum.grad_type: 1,
            IArgsEnum.eta_grad : 0.015,
            IArgsEnum.mu_grad  : 0.02, # 0.5
            IArgsEnum.grad_tol : 0.0001,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
            'valid_Ks'   : valid_Ks,
        }
        input_args_projection = {
            InputTaurusPAV.ArgsEnum.red_hamil : 1,
            InputTaurusPAV.ArgsEnum.alpha : 13,
            InputTaurusPAV.ArgsEnum.beta  : 20,
            InputTaurusPAV.ArgsEnum.gamma : 13,
            InputTaurusPAV.ArgsEnum.j_max : 21,
            InputTaurusPAV.ArgsEnum.j_min : 1,
            InputTaurusPAV.ArgsEnum.empty_states : 0,
            InputTaurusPAV.ArgsEnum.disable_simplifications_P : 0,
            InputTaurusPAV.ArgsEnum.cutoff_overlap : 1.0e-10,
            # PN-PAV and J bound arguments set by the program, P-PAV = no
        }
        ExeTaurus1D_B20_KMixing_OEblocking.EXPORT_LIST_RESULTS = \
            f"export_TESb20_z{z}n{n}_{interaction}.txt"
        try:
            exe_ = ExeTaurus1D_B20_KMixing_OEblocking(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min, q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.setUpProjection(**input_args_projection)
            exe_.force_converg = False
            exe_.run(fomenko_points)
            exe_.globalTearDown()
        except ExecutionException as e:
            printf("[SCRIPT ERROR]:", e)
        
    printf("End run_b20_surface k-mixing: ", datetime.now().time())


def run_b20_Block1KandPAV_exampleSingleJ(
                            nucleus, interactions, gogny_interaction, valid_Ks, 
                            seed_base=0, ROmega=(13, 13),
                            q_min=-2.0, q_max=2.0, N_max=41, convergences=0,
                            fomenko_points=(1, 1), parity_2_block= 1,
                            ):
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
    """
    if ((fomenko_points[0]>1 or fomenko_points[1]>1) 
        and gogny_interaction != GognyEnum.B1):
        raise ExecutionException(" Projection is not defined for taurus_vap with density-dependent")
    
    ExeTaurus1D_B20_Ksurface_Base.RUN_PROJECTION        = True 
    
    ExeTaurus1D_B20_Ksurface_Base.ITERATIVE_METHOD = \
        ExeTaurus1D_B20_Ksurface_Base.IterativeEnum.EVEN_STEP_STD
        
    ExeTaurus1D_B20_Ksurface_Base.SAVE_DAT_FILES = [
        # DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        # DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    ExeTaurus1D_B20_Ksurface_Base.DO_BASE_CALCULATION   = False    # *** IMPORTANT
    ExeTaurus1D_B20_Ksurface_Base.GENERATE_RANDOM_SEEDS = False    # *** IMPORTANT
    ExeTaurus1D_B20_Ksurface_Base.SEEDS_RANDOMIZATION   = convergences
    if convergences > 0:
        printf(" [WARNING] given CONVERGENCES>0 to scripts, that will do BASE calculations (not recomended)\n"*3)
                
        ExeTaurus1D_B20_Ksurface_Base.GENERATE_RANDOM_SEEDS = bool(convergences)
        ExeTaurus1D_B20_Ksurface_Base.DO_BASE_CALCULATION   = convergences >= 0
    ExeTaurus1D_B20_Ksurface_Base.PARITY_TO_BLOCK       = parity_2_block
    
    ExeTaurus1D_B20_Ksurface_Base.FULLY_CONVERGE_BLOCKING_ITER_MODE  = True
    ExeTaurus1D_B20_Ksurface_Base.PRECONVERNGECE_BLOCKING_ITERATIONS = 0
    
    ## Most of the previous class-execution attributes cannot be set.
    
    if ExeTaurus1D_B20_Ksurface_Base.RUN_PROJECTION: 
        ## Import the programs if they do not exist
        importAndCompile_taurus(pav= not os.path.exists(InputTaurusPAV.PROGRAM))
    
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
        
        ## Note: For false o-e projection, fomenko_points are given in run()
        ##          see Note in script run_b20_FalseOE_Block1KAndPAV()
        IArgsEnum = InputTaurus.ArgsEnum
        
        input_args_start = {
            IArgsEnum.z_Mphi : fomenko_points[0], 
            IArgsEnum.n_Mphi : fomenko_points[1],
            IArgsEnum.com : 1,
            IArgsEnum.seed: seed_base,
            IArgsEnum.iterations: 2000,
            IArgsEnum.grad_type: 1,
            IArgsEnum.grad_tol : 0.0005,
            IArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            IArgsEnum.pair_schm: 1,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
        }
        input_args_onrun = {
            IArgsEnum.z_Mphi : fomenko_points[0],
            IArgsEnum.n_Mphi : fomenko_points[1],
            IArgsEnum.red_hamil: 1,
            IArgsEnum.seed: 1,
            IArgsEnum.iterations: 2000,
            IArgsEnum.grad_type: 1,
            IArgsEnum.eta_grad : 0.015,
            IArgsEnum.mu_grad  : 0.02, # 0.5
            IArgsEnum.grad_tol : 0.0005,
            InputTaurus.ConstrEnum.b22 : (0.00, 0.00),
            #InputTaurus.ConstrEnum.b40 : (0.00, 0.00),
            'axial_calc' : axial_calc,
            'valid_Ks'   : valid_Ks,
        }
        input_args_projection = {
            InputTaurusPAV.ArgsEnum.red_hamil : 1,
            InputTaurusPAV.ArgsEnum.alpha : 20,
            InputTaurusPAV.ArgsEnum.beta  : 20,
            InputTaurusPAV.ArgsEnum.gamma : 20,
            InputTaurusPAV.ArgsEnum.j_max : 19,
            InputTaurusPAV.ArgsEnum.j_min : 1,
            InputTaurusPAV.ArgsEnum.empty_states : 0,
            InputTaurusPAV.ArgsEnum.disable_simplifications_P : 0,
            InputTaurusPAV.ArgsEnum.cutoff_overlap : 1.0e-10,
            # PN-PAV and J bound arguments set by the program, P-PAV = no
        }
        ExeTaurus1D_B20_Ksurface_Base.EXPORT_LIST_RESULTS = \
            f"export_TESb20_z{z}n{n}_{interaction}.txt"
        try:
            exe_ = ExeTaurus1D_B20_Ksurface_Base(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(q_min, q_max, N_max)
            exe_.setUp()
            exe_.setUpExecution(**input_args_onrun)
            exe_.setUpProjection(**input_args_projection)
            exe_.force_converg = False
            exe_.run(fomenko_points)
            exe_.globalTearDown()
        except ExecutionException as e:
            printf("[SCRIPT ERROR]:", e)
        
    printf("End run_b20_surface k-mixing: ", datetime.now().time())
