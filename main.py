'''
Created on Jan 10, 2023

@author: Miguel
'''
import os, shutil
from scripts1d.beta_scripts import run_q20_surface, run_b20_Gogny_surface,\
    run_b20_composedInteraction, run_b20_surface
from tools.helpers import importAndCompile_taurus, TBME_SUITE

from tools.hamiltonianMaker import TBME_HamiltonianManager, TBMEXML_Setter
from tools.inputs import InputTaurus
from scripts1d.pair_scripts import run_pair_surface_D1S
from scripts1d.cranking_scripts import run_J_surface
from scripts0d.unconstrained_scripts import run_symmetry_restricted_for_hamiltonian
from tools.Enums import GognyEnum, CentralMEParameters, PotentialForms,\
    OutputFileTypes

if not (InputTaurus.PROGRAM in os.listdir()):
    importAndCompile_taurus()

if __name__ == '__main__':
    
    # # exe_ = TBME_HamiltonianManager(1.75, 1, set_com2=True)
    # # exe_.setAndRun_Gogny_xml(TBME_HamiltonianManager.GognyEnum.D1S)
    #
    # importAndCompile_taurus()
    
    
    """
    This executable, with some modifications in run_b20 to do an axial calculation
    is to copy and read hamiltonians from a folder and perform, without DD components
    the minimization.
    """
    # for trf in range(75, 131, 5):
    #
    #     MAIN_HAMIL_FOLDER = 'FOLDER_GDD/'
    #     curr_hamil = f'hamil_gdd_{trf:03}'
    #     print(" Execute for Hamil:", curr_hamil)
    #     if (curr_hamil+'.sho' in os.listdir(MAIN_HAMIL_FOLDER)):
    #         for extension in OutputFileTypes.members():
    #             file_ = MAIN_HAMIL_FOLDER+curr_hamil+extension
    #             if os.path.exists(file_): 
    #                 shutil.copy(file_, curr_hamil+extension)
    #     else:
    #         print(" [ERROR] could not find it:", MAIN_HAMIL_FOLDER+curr_hamil)
    #         continue
    #
    #     nucleus = [(12, 12), ]
    #     interactions = {(12, 12): curr_hamil, }
    #
    #     run_b20_surface(nucleus, interactions, q_min=-.4, q_max=.5, N_max=45,
    #                     seed_base=3, ROmega= (0, 0), convergences=5,
    #                     fomenko_points=(9, 9))
        
    """
    This script perform gogny surfaces by also obtaining the non-density dependent
    part of Gogny interactions by the key-word argument:
        gogny_interaction = GognyEnum.
    """
    interactions = {
        # ( 12, 10): (3, 0, 1.94),
        ( 12, 12): (3, 0, 1.83),
        # ( 12, 14): (3, 0, 1.95),
    }
    nucleus = sorted(list(interactions.keys()))
    run_b20_Gogny_surface(nucleus, interactions, GognyEnum.D1S,
                          seed_base=3, ROmega=(14,12), 
                          q_min=-0.4, q_max=0.6, N_max=51, convergences=5)
    # raise Exception("STOP HERE.")
    
    
    """
    Build up interactions and perform the b20 interaction.
    """
    # interactions = {
    #     ( 12, 10): (3, 0, 1.94),
    #     ( 12, 12): (3, 0, 1.92),
    #     ( 12, 14): (3, 0, 1.95),
    # }
    # nucleus = sorted(list(interactions.keys()))
    #
    # interaction_combination = []
    # kwargs = {CentralMEParameters.potential: PotentialForms.Gaussian,
    #           CentralMEParameters.constant : -100.,
    #           CentralMEParameters.mu_length: 1.4,}
    # interaction_combination.append((TBMEXML_Setter.set_central_force, kwargs ))
    #
    # kwargs = {CentralMEParameters.constant: 50.3,
    #           CentralMEParameters.n_power:  2,}
    # interaction_combination.append((TBMEXML_Setter.set_quadrupole_force, kwargs ))
    #
    # run_b20_composedInteraction(nucleus, interactions, interaction_combination,
    #                             seed_base=3,
    #                             q_min=-0.4, q_max=0.6, N_max=51, convergences=5)
    # raise Exception("STOP HERE.")
    
    #
    # ## MZ4 lengths
    # interactions = {
    #     # ( 6, 6): (4, 0, 1.60), ( 6, 8): (4, 0, 1.77),
    #     # ( 8, 8): (4, 0, 1.85), ( 8,10): (4, 0, 1.65),
    #     # (10,10): (4, 0, 1.80), (10,12): (4, 0, 1.75),
    #     # (14,14): (4, 0, 1.65), (14,16): (4, 0, 1.80),
    #     # (16,16): (4, 0, 1.85), (16,18): (4, 0, 1.75),
    #     (18,18): (4, 0, 1.80), (18,20): (4, 0, 1.80),
    #     (20,20): (4, 0, 1.76), 
    # }
    # nucleus = sorted(list(interactions.keys()))
    #
    # run_J_surface(nucleus, interactions, InputTaurus.ConstrEnum.Jx,
    #               seed_base=0, j_min=0.0, j_max=25.0, N_max=50)
    #
    #
    # PAIR_CONSTRS = [
    #     InputTaurus.ConstrEnum.P_T00_J10,  InputTaurus.ConstrEnum.P_T10_J00,
    #     InputTaurus.ConstrEnum.P_T1m1_J00, InputTaurus.ConstrEnum.P_T1p1_J00
    # ]
    # # run_pair_surface_D1S(nucleus, interactions, PAIR_CONSTRS, 
    # #                  seed_base=5, p_min=-0.05, p_max=2.0, N_max=41)
    
    # nucleus = [(2,1+n) for n in range(9)] #18, 14
    # run_symmetry_restricted_for_hamiltonian(nucleus, 
    #                                         MZmax=1, seed_base=0, 
    #                                         ROmega=(15,16), convergences=20)
    
        