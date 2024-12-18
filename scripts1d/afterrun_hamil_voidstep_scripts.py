'''
Created on 11 dic 2024

@author: delafuente
'''
from tools.helpers import importAndCompile_taurus, printf
import os
from tools.hamiltonianMaker import TBMEXML_Setter
from tools.Enums import PotentialForms, CentralMEParameters,\
    BrinkBoekerParameters, GognyEnum, ForceFromFileParameters
from tools.inputs import InputTaurus, InputTaurusPAV
from tools.afterrun_hamiltonians import ExeTaurus1D_AfterRun_HamilDecomposition 

def run_b20_decomposeHamiltonian_GognyB1(nuclei, constraints=['',], 
                                         fomenko_points=(1, 1),):
    """
    Evaluate the different contributions from the Gogny interaction. Evaluate 
    as no-core Hamiltonian Generator.
    
    nuclei: {(z,n): (Mzmin, Mzmax, b length), ...}
    """
        
    importAndCompile_taurus(use_dens_taurus=False, #pav=True,
                            force_compilation=not os.path.exists('taurus_pav.exe'))
    input_args ={
        InputTaurus.ArgsEnum.red_hamil : 0,
        InputTaurus.ArgsEnum.z_Mphi    : fomenko_points[0],
        InputTaurus.ArgsEnum.n_Mphi    : fomenko_points[1],
        InputTaurus.ArgsEnum.com       : 1,
        InputTaurus.ArgsEnum.grad_type : 0,
        InputTaurus.ArgsEnum.iterations: 0,
        InputTaurus.ArgsEnum.eta_grad  : 1.0e-50,
        InputTaurus.ArgsEnum.mu_grad   : 0.0,
        InputTaurus.ArgsEnum.pair_schm : 1,    ## check out from the previous calc
        InputTaurus.ArgsEnum.seed      : 1,
    }
    input_args_projection = {
        InputTaurusPAV.ArgsEnum.red_hamil : 0,
        InputTaurusPAV.ArgsEnum.com   : 1,
        InputTaurusPAV.ArgsEnum.z_Mphi: fomenko_points[0],
        InputTaurusPAV.ArgsEnum.n_Mphi: fomenko_points[1],
        InputTaurusPAV.ArgsEnum.disable_simplifications_NZA: 1,
        # InputTaurusPAV.ArgsEnum.alpha : 0,
        # InputTaurusPAV.ArgsEnum.beta  : 0,
        # InputTaurusPAV.ArgsEnum.gamma : 0,
        InputTaurusPAV.ArgsEnum.disable_simplifications_JMK: 1,
        InputTaurusPAV.ArgsEnum.disable_simplifications_P : 1,
        InputTaurusPAV.ArgsEnum.empty_states : 0,
        InputTaurusPAV.ArgsEnum.cutoff_overlap : 1.0e-10,
        # PN-PAV and J bound arguments set by the program, P-PAV = no
    }
    
    interactions = {}
    interactions['gauss1Wigner'] = (
        TBMEXML_Setter.set_central_force,  
        {
            CentralMEParameters.potential: PotentialForms.Gaussian,
            BrinkBoekerParameters.Wigner :   595.55,
            BrinkBoekerParameters.Bartlett :   0.0,
            BrinkBoekerParameters.Heisenberg : 0.0,
            BrinkBoekerParameters.Majorana :   0.0,
            CentralMEParameters.mu_length:     0.7,
        }
    )
    interactions['gauss1Majorana'] = (
        TBMEXML_Setter.set_central_force,  
        {
            CentralMEParameters.potential: PotentialForms.Gaussian,
            BrinkBoekerParameters.Wigner :       0.0,
            BrinkBoekerParameters.Bartlett :     0.0,
            BrinkBoekerParameters.Heisenberg:    0.0,
            BrinkBoekerParameters.Majorana :  -206.05,
            CentralMEParameters.mu_length:       0.7,
        }
    )
    interactions['gauss2Wigner'] = (
        TBMEXML_Setter.set_central_force,  
        {
            CentralMEParameters.potential: PotentialForms.Gaussian,
            BrinkBoekerParameters.Wigner :   -72.21,
            BrinkBoekerParameters.Bartlett :   0.0,
            BrinkBoekerParameters.Heisenberg : 0.0,
            BrinkBoekerParameters.Majorana :   0.0,
            CentralMEParameters.mu_length:     1.4,
        }
    )
    interactions['gauss2Majorana'] = (
        TBMEXML_Setter.set_central_force,  
        {
            CentralMEParameters.potential: PotentialForms.Gaussian,
            BrinkBoekerParameters.Wigner :       0.0,
            BrinkBoekerParameters.Bartlett :     0.0,
            BrinkBoekerParameters.Heisenberg :   0.0,
            BrinkBoekerParameters.Majorana :   -68.39,
            CentralMEParameters.mu_length:       1.4,
        }
    )
    interactions['spinOrbit'] = (
        TBMEXML_Setter.set_spinorbitSR_force,  
        {
            CentralMEParameters.potential: PotentialForms.Power,
            CentralMEParameters.constant :     115.0,
            CentralMEParameters.mu_length:       1.0,
            CentralMEParameters.n_power:           0,
        }
    )
    interactions['coulomb'] = (TBMEXML_Setter.set_coulomb_force,  {})
    interactions['void'] = (
        TBMEXML_Setter.set_file_force, {'filename' : 'savedHamilsBeq1/void.2b', }
    )
    
    MAIN_FLD = ''
    printf(" Evaluating Energies from different contributions for B1 interaction.")
    for zn, inter in nuclei.items():
        ExeTaurus1D_AfterRun_HamilDecomposition.HAMIL_NAME = GognyEnum.B1
        
        exe_ = ExeTaurus1D_AfterRun_HamilDecomposition(*zn, inter, MAIN_FLD)
        #exe_.setUpPAVparameters(**input_args_projection)
        exe_.setUpVAPparameters(**input_args)
        exe_.setUpHamiltonians(interactions, combine_all=True)
        
        for constr in constraints:
            cstr = constr.replace('_', '')
            if cstr != '': cstr = '_' + cstr
            bu_folder = 'BU_folder{}_{}_z{}n{}'.format(cstr, exe_.interaction, *zn)
            
            printf(f"    *** Evaluating hamiltonians on [{bu_folder}] solutions.")
            ok_ = exe_.processData(bu_folder, observable=constr)
            if ok_: exe_.run()
            else: printf("    *** Problem found in folder. SKIPPING.")
        
        printf(" Finished Energy decomposition z,n=", zn,"!")



