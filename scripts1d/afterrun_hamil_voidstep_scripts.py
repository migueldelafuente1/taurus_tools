'''
Created on 11 dic 2024

@author: delafuente
'''
from tools.helpers import importAndCompile_taurus, printf
import os
from tools.hamiltonianMaker import TBMEXML_Setter
from tools.Enums import PotentialForms, CentralMEParameters,\
    BrinkBoekerParameters, GognyEnum, ForceFromFileParameters, M3YEnum
from tools.inputs import InputTaurus, InputTaurusPAV
from tools.afterrun_hamiltonians import ExeTaurus1D_AfterRun_HamilDecomposition 
from copy import deepcopy

def run_b20_decomposeHamiltonian_Gogny(nuclei, parametrization, constraints=['',], 
                                          fomenko_points=(1, 1), ROmega=(0,0)):
    """
    Evaluate the different contributions from the Gogny interaction. Evaluate 
    as no-core Hamiltonian Generator.
    
    nuclei: {(z,n): (Mzmin, Mzmax, b length), ...}
    """
    use_dens_taurus = parametrization != GognyEnum.B1
    importAndCompile_taurus(use_dens_taurus=use_dens_taurus, #pav=True,
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
    
    if parametrization == GognyEnum.D1S:
        _gaussians = {
            'Wigner':     {1: -1720.3,  2:   103.639},
            'Bartlett':   {1:  1300,    2:  -163.483},
            'Heisenberg': {1: -1813.53, 2:   162.812},
            'Majorana':   {1:  1397.6,  2:  -223.934},
        }
        mu_lengths = {1: 0.7, 2: 1.2}
        W_ls = 130.0
        input_DD = {
            InputTaurus.InpDDEnum.alpha_param: 0.333333,
            InputTaurus.InpDDEnum.x0_param: 1.0,
            InputTaurus.InpDDEnum.t3_param: 1390.6,
            InputTaurus.InpDDEnum.r_dim: ROmega[0],
            InputTaurus.InpDDEnum.omega_dim: ROmega[1],
        }
    elif parametrization == GognyEnum.B1:
        _gaussians = {
            'Wigner':     {1: 595.55,   2: -72.21},
            # 'Bartlett':   {1: 0,        2:   0},
            # 'Heisenberg': {1: 0,        2:   0},
            'Majorana':   {1: -206.05,  2: -68.39},
        }
        mu_lengths = {1: 0.7, 2: 1.4}
        W_ls = 115.0
        input_DD = {}
    else:
        raise Exception(f" Not implemented [{parametrization}]")
    
    params0 = dict([(p, 0.0) for p in BrinkBoekerParameters.members()])
    params0[CentralMEParameters.potential] = PotentialForms.Gaussian
    for term in BrinkBoekerParameters.members():
        if term == BrinkBoekerParameters.mu_length: continue
        if not term in _gaussians: continue # term is not present
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
            CentralMEParameters.constant : W_ls,
            CentralMEParameters.mu_length: 1.0,
            CentralMEParameters.n_power:   0,
        }
    )
    interactions['coulomb'] = (TBMEXML_Setter.set_coulomb_force,  {})
    interactions['void'] = (
        TBMEXML_Setter.set_file_force, {'filename' : 'savedHamilsBeq1/void.2b', }
    )
    
    MAIN_FLD = ''
    printf(" Evaluating Energies from different contributions for D1S interaction.")
    for zn, inter in nuclei.items():
        ExeTaurus1D_AfterRun_HamilDecomposition.HAMIL_NAME = parametrization
        
        exe_ = ExeTaurus1D_AfterRun_HamilDecomposition(*zn, inter, MAIN_FLD)
        #exe_.setUpPAVparameters(**input_args_projection)
        exe_.setUpVAPparameters(**input_args)
        exe_.setUpHamiltonians(interactions, combine_all=True)
        if input_DD: exe_.setUpDDTerm(input_DD)
        
        for constr in constraints:
            cstr = constr.replace('_', '')
            if cstr != '': cstr = '_' + cstr
            bu_folder = 'BU_folder{}_{}_z{}n{}'.format(cstr, exe_.interaction, *zn)
            
            printf(f"    *** Evaluating hamiltonians on [{bu_folder}] solutions.")
            ok_ = exe_.processData(bu_folder, observable=constr)
            if ok_: exe_.run()
            else: printf("    *** Problem found in folder. SKIPPING.")
        
        printf(" Finished Energy decomposition z,n=", zn,"!")

def run_b20_decomposeHamiltonian_M3Y(nuclei, parametrization, constraints=['',],
                                     fomenko_points=(1, 1), ROmega=(0,0)):
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
    input_DD, input_DD_2 = {}, {}
    if parametrization == M3YEnum.P0:
        COMPONENTS = {
            "central": {
                'Wigner': {1: 8840.0000, 2: -2275.0000, 3:  3.4878, },
                'Heisenberg': {1: -2565.5000, 2: 453.0000, 3:  6.9752, },
                'Bartlett': {1: 3816.0000, 2: -972.0000, 3: -6.9753, },
                'Majorana': {1: 3876.5000, 2: -1800.0000, 3: -13.9507, },
                'mu_length': {1:  0.2500, 2:  0.4000, 3:  1.4140, },
            },
            "spinOrb": {
                'Wigner': {1: -1749.5000, 2: -242.2500, },
                'Heisenberg': {1: -801.0000, 2: 73.7500, },
                'Bartlett': {1: -1749.5000, 2: -242.2500, },
                'Majorana': {1: -801.0000, 2: 73.7500, },
                'mu_length': {1:  0.2500, 2:  0.4000, },
            },
            "tensor": {
                'Wigner': {1: -213.0000, 2: -3.8250, },
                'Heisenberg': {1: -335.0000, 2: -11.6250, },
                'Bartlett': {1: -213.0000, 2: -3.8250, },
                'Majorana': {1: -335.0000, 2: -11.6250, },
                'mu_length': {1:  0.4000, 2:  0.7000, },
        }}
    elif parametrization == M3YEnum.P2:
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
        }}
        input_DD = InputTaurus.getDDParametersByInteraction(M3YEnum.P2)
    elif parametrization == M3YEnum.P6:
        COMPONENTS = {
            "central": {
                'Wigner': {1: 7741.2500, 2: -2079.0000, 3:  3.4878, },
                'Heisenberg': {1: -3868.2500, 2: 475.0000, 3:  6.9752, },
                'Bartlett': {1: 2722.2500, 2: -1012.0000, 3: -6.9753, },
                'Majorana': {1: 1878.7500, 2: -1978.0000, 3: -13.9507, },
                'mu_length': {1:  0.2500, 2:  0.4000, 3:  1.4140, },
            },
            "spinOrb": {
                'Wigner': {1: -3848.9000, 2: -532.9500, },
                'Heisenberg': {1: -1762.2000, 2: 162.2500, },
                'Bartlett': {1: -3848.9000, 2: -532.9500, },
                'Majorana': {1: -1762.2000, 2: 162.2500, },
                'mu_length': {1:  0.2500, 2:  0.4000, },
            },
            "tensor": {
                'Wigner': {1: -213.0000, 2: -3.8250, },
                'Heisenberg': {1: -335.0000, 2: -11.6250, },
                'Bartlett': {1: -213.0000, 2: -3.8250, },
                'Majorana': {1: -335.0000, 2: -11.6250, },
                'mu_length': {1:  0.4000, 2:  0.7000, },
        }}
        input_DD = {
            InputTaurus.InpDDEnum.eval_dd : 1, InputTaurus.InpDDEnum.eval_rea: 1,
            InputTaurus.InpDDEnum.x0_param: 1.0,   
            InputTaurus.InpDDEnum.alpha_param : 0.333333,  
            InputTaurus.InpDDEnum.t3_param: 482.5, 
            InputTaurus.InpDDEnum.more_options: {31: 1.0,    32: 1.0}
        }
        input_DD_2 = {
            InputTaurus.InpDDEnum.eval_dd : 1, InputTaurus.InpDDEnum.eval_rea: 1,
            InputTaurus.InpDDEnum.x0_param: -1.0,   
            InputTaurus.InpDDEnum.alpha_param : 1.0,  
            InputTaurus.InpDDEnum.t3_param: 96.0, 
            InputTaurus.InpDDEnum.more_options: {31: -1.0,    32: 1.0}
        }
    
    interactions = {}
    for termY, dict_ in COMPONENTS.items():
        params0 = dict([(p, 0.0) for p in BrinkBoekerParameters.members()])
        params0[CentralMEParameters.potential] = PotentialForms.Gaussian
        mu_lengths = dict_[BrinkBoekerParameters.mu_length]
        for term in BrinkBoekerParameters.members():
            if term == BrinkBoekerParameters.mu_length: continue
            if not term in COMPONENTS[termY]: continue # term is not present
            
            for part in (1, 2, 3):
                if termY != 'central' and part == 3: continue
                
                parameters = deepcopy(params0)
                parameters[term] = dict_[term][part]
                parameters[BrinkBoekerParameters.mu_length] = mu_lengths[part]
                
                if termY == 'central':
                    interactions[f'centralYuk{part}{term}'] = (
                        TBMEXML_Setter.set_central_force, parameters )
                elif termY == 'spinOrbit':
                    interactions[f'LS_Yuk{part}{term}'] = (
                        TBMEXML_Setter.set_spinorbitFR_force, parameters )
                elif termY == 'tensor':
                    interactions[f'tenYuk{part}{term}'] = (
                        TBMEXML_Setter.set_tensor_force, parameters )
        
    interactions['coulomb'] = (TBMEXML_Setter.set_coulomb_force,  {})
    interactions['void'] = (
        TBMEXML_Setter.set_file_force, {'filename' : 'savedHamilsBeq1/void.2b', }
    )
    
    MAIN_FLD = ''
    printf(" Evaluating Energies from different contributions for D1S interaction.")
    for zn, inter in nuclei.items():
        ExeTaurus1D_AfterRun_HamilDecomposition.HAMIL_NAME = parametrization
        
        exe_ = ExeTaurus1D_AfterRun_HamilDecomposition(*zn, inter, MAIN_FLD)
        #exe_.setUpPAVparameters(**input_args_projection)
        exe_.setUpVAPparameters(**input_args)
        exe_.setUpHamiltonians(interactions, combine_all=True)
        if input_DD:   exe_.setUpDDTerm(input_DD)
        if input_DD_2: exe_.setUpDDTerm(input_DD_2)
        
        for constr in constraints:
            cstr = constr.replace('_', '')
            if cstr != '': cstr = '_' + cstr
            bu_folder = 'BU_folder{}_{}_z{}n{}'.format(cstr, exe_.interaction, *zn)
            
            printf(f"    *** Evaluating hamiltonians on [{bu_folder}] solutions.")
            ok_ = exe_.processData(bu_folder, observable=constr)
            if ok_: exe_.run()
            else: printf("    *** Problem found in folder. SKIPPING.")
        
        printf(" Finished Energy decomposition z,n=", zn,"!")
