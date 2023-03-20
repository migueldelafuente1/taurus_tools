'''
Created on Mar 17, 2023

@author: Miguel
'''
from tools.executors import ExeTaurus0D_EnergyMinimum, ExecutionException
from tools.data import DataTaurus
from scripts1d.script_helpers import getInteractionFile4D1S
from tools.inputs import InputTaurus
import os


def run_symmetry_restricted_for_hamiltonian(nucleus, MZmax=6,
                                            seed_base=0, ROmega=(15,15),
                                            convergences=None):
    """
    Script with no constrains, only evaluating the minimum for a certain 
    hamiltonian, repetitions are necessary depending on the seed 
    symmetries.
    
    The base results will be saved in a results file indexed from the try
    Args:
        :nucleus: <list>: (z1,n1), (z2,n2), ..
           (changed) :interactions: <dict> [Nucleus (z, n)]: (MZ_max, Mz_min, b_length)
        :MZmax: int (the interaction dict uses b_lenght empirical)
    Optional:
        :seed_base (taurus_input seeds, pn-mix True= 0 & 4)
        :ROmega: <tuple>=(R, Omega) grid of Integration (Default is 10, 10)
        :convergences: <int> number of random seeds / blocked states to get the global minimum  
    """
    
    ExeTaurus0D_EnergyMinimum.SAVE_DAT_FILES = [
        DataTaurus.DatFileExportEnum.canonicalbasis,
        DataTaurus.DatFileExportEnum.eigenbasis_h,
        DataTaurus.DatFileExportEnum.occupation_numbers,
        ]
    
    if convergences != None:
        assert type(convergences) == int, "it should be an integer"
        ExeTaurus0D_EnergyMinimum.SEEDS_RANDOMIZATION = convergences
        ExeTaurus0D_EnergyMinimum.GENERATE_RANDOM_SEEDS = True
    
    interactions = [((z,n), (MZmax, 0, 1.005*((z+n)**(1/6)))) for z,n in nucleus]
    interactions = dict(interactions)
    
    for z, n in nucleus:
        ## TODO: define the hamiltonian b=empirical formula
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
            InputTaurus.ArgsEnum.grad_tol : 0.003,
            InputTaurus.ArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
            InputTaurus.ArgsEnum.pair_schm: 1,
        }
        ## process to save the results _ index is just the index
        try:
            if convergences != None:
                # for even-odd run blocked states until left without valence space
                if not (z%2 or n%2): ## its even-even
                    # for even-even nuclei run only  one case
                    ExeTaurus0D_EnergyMinimum.SEEDS_RANDOMIZATION = 3
                else:
                    ExeTaurus0D_EnergyMinimum.SEEDS_RANDOMIZATION = convergences
            
            exe_ = ExeTaurus0D_EnergyMinimum(z, n, interaction)
            exe_.setInputCalculationArguments(**input_args_start)
            exe_.defineDeformationRange(0.0, 0.0, 0)
            exe_.setUp()
            exe_.setUpExecution()
            # exe_.force_converg = True
            # exe_.run()
            exe_.gobalTearDown()
        except ExecutionException as e:
            print(e)
        
        ## Maybe to use a single evaluation of a 1d script. Then read the 
        # zip BASE for the seed results.
        
        
    
    pass


        