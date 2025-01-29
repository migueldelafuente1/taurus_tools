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
from copy import deepcopy

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

def run_b20_decomposeHamiltonian_GognyD1S(nuclei, constraints=['',], 
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
    
    _gaussians = {
        'Wigner':     {1: -1720.3,  2:   103.639},
        'Bartlett':   {1:  1300,    2:  -163.483},
        'Heisenberg': {1: -1813.53, 2:   162.812},
        'Majorana':   {1:  1397.6,  2:  -223.934},
    }
    
    params0 = dict([(p, 0.0) for p in BrinkBoekerParameters.members()])
    params0[CentralMEParameters.potential] = PotentialForms.Gaussian
    mu_lengths = {1: 0.7, 2: 1.2}
    for term in BrinkBoekerParameters.members():
        if term == BrinkBoekerParameters.mu_length: continue
        for part in (1, 2):
            
            parameters = deepcopy(params0)
            parameters[term] = _gaussians[term][part]
            parameters[BrinkBoekerParameters.mu_length] = mu_lengths[part]
            
            interactions[f'gauss{part}{term}'] = (
                TBMEXML_Setter.set_central_force, parameters
            )
    
    interactions['spinOrbit'] = (
        TBMEXML_Setter.set_spinorbitSR_force,  
        {
            CentralMEParameters.potential: PotentialForms.Power,
            CentralMEParameters.constant :     130.0,
            CentralMEParameters.mu_length:       1.0,
            CentralMEParameters.n_power:           0,
        }
    )
    interactions['coulomb'] = (TBMEXML_Setter.set_coulomb_force,  {})
    interactions['void'] = (
        TBMEXML_Setter.set_file_force, {'filename' : 'savedHamilsBeq1/void.2b', }
    )
    
    MAIN_FLD = ''
    printf(" Evaluating Energies from different contributions for D1S interaction.")
    for zn, inter in nuclei.items():
        ExeTaurus1D_AfterRun_HamilDecomposition.HAMIL_NAME = GognyEnum.D1S
        
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

def run_b20_decomposeHamiltonian_M3Y(nuclei, constraints=['',], 
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
    COMPONENTS = {
        'central': {
            'Wigner': {1: 1501.7500000, 2: -1299.0000000, 3: 3.4877500, },
            'Heisenberg': {1: -4411.7500000, 2: 531.0000000, 3: 6.9752500, },
            'Bartlett': {1: 3438.2500000, 2: -1224.0000000, 3: -6.9752500, },
            'Majorana': {1: 5551.7500000, 2: -2274.0000000, 3: -13.9507500, },
            'mu_length': {1: 0.2500000, 2: 0.4000000, 3: 1.4140000, },
        },
        'spinOrbit': {
            'Wigner': {1: -3149.1000000, 2: -436.0500000, },
            'Heisenberg': {1: -1441.8000000, 2: 132.7500000, },
            'Bartlett': {1: -3149.1000000, 2: -436.0500000, },
            'Majorana': {1: -1441.8000000, 2: 132.7500000, },
            'mu_length': {1: 0.2500000, 2: 0.4000000, },
        },
        'tensor': {
            'Wigner': {1: -25.5600000, 2: -0.4590000, },
            'Heisenberg': {1: -40.2000000, 2: -1.3950000, },
            'Bartlett': {1: -25.5600000, 2: -0.4590000, },
            'Majorana': {1: -40.2000000, 2: -1.3950000, },
            'mu_length': {1: 0.4000000, 2: 0.7000000, },
        }
    }
    
    for term, dict_ in COMPONENTS.items():
        params0 = dict([(p, 0.0) for p in BrinkBoekerParameters.members()])
        params0[CentralMEParameters.potential] = PotentialForms.Gaussian
        mu_lengths = dict_[BrinkBoekerParameters.mu_length]
        for term in BrinkBoekerParameters.members():
            if term == BrinkBoekerParameters.mu_length: continue
            for part in (1, 2, 3):
                if term != 'central' and part == 3: continue
                
                parameters = deepcopy(params0)
                parameters[term] = dict_[term][part]
                parameters[BrinkBoekerParameters.mu_length] = mu_lengths[part]
                
                if term == 'central':
                    interactions[f'centralYuk{part}{term}'] = (
                        TBMEXML_Setter.set_central_force, parameters )
                elif term == 'spinOrbit':
                    interactions[f'tenYuk{part}{term}'] = (
                        TBMEXML_Setter.set_spinorbitFR_force, parameters )
                elif term == 'tensor':
                    interactions[f'tenYuk{part}{term}'] = (
                        TBMEXML_Setter.set_tensor_force, parameters )
        
    interactions['coulomb'] = (TBMEXML_Setter.set_coulomb_force,  {})
    interactions['void'] = (
        TBMEXML_Setter.set_file_force, {'filename' : 'savedHamilsBeq1/void.2b', }
    )
    
    MAIN_FLD = ''
    printf(" Evaluating Energies from different contributions for D1S interaction.")
    for zn, inter in nuclei.items():
        ExeTaurus1D_AfterRun_HamilDecomposition.HAMIL_NAME = GognyEnum.D1S
        
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
