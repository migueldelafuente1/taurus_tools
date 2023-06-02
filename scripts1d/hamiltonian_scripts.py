'''
Created on Jan 20, 2023

@author: Miguel
'''
import os
import numpy as np
from tools.hamiltonianMaker import TBME_HamiltonianManager
from tools.inputs import InputTaurus, InputAxial
from tools.executors import ExeTaurus0D_EnergyMinimum, ExeAxial1D_DeformB20, \
    ExecutionException, ExeAxial0D_EnergyMinimum
from tools.data import DataTaurus, DataAxial
from tools.plotter_1d import Plotter1D_Axial, Plotter1D_Taurus
from tools.helpers import prettyPrintDictionary

def run_computingHOhbarOmegaForD1S(nucleus, MZmax=4, bHO_min=1.5, bHO_max=2.75, 
                                   Nsteps=6, MZmin=0):
    """ 
    Script for Taurus, get a curve b length to see where is the minimum of E hfb
    """
    b_lengths = list(np.linspace(bHO_min, bHO_max, Nsteps, endpoint=True))
    b_lengths.reverse()
    
    InputTaurus.set_inputDDparamsFile(
        **{InputTaurus.InpDDEnum.r_dim : 12,
           InputTaurus.InpDDEnum.omega_dim : 14})
    
    input_args_start = {
        InputTaurus.ArgsEnum.com : 1,
        InputTaurus.ArgsEnum.seed: 5,
        InputTaurus.ArgsEnum.iterations: 1000,
        InputTaurus.ArgsEnum.grad_type: 1,
        InputTaurus.ArgsEnum.grad_tol : 0.001,
        InputTaurus.ArgsEnum.beta_schm: 1, ## 0= q_lm, 1 b_lm, 2 triaxial
        InputTaurus.ArgsEnum.pair_schm: 1,
    }
    
    input_args_onrun = { **input_args_start,
        InputTaurus.ArgsEnum.red_hamil: 1,
        InputTaurus.ArgsEnum.seed: 1,
        InputTaurus.ArgsEnum.iterations: 0,
    } # just get the minimum result
    ExeAxial0D_EnergyMinimum.PRINT_CALCULATION_PARAMETERS = False
    optimal_lengths = {}
    
    for z, n in nucleus:
        b_Ehfb_min = (0, 999999)
        summary_results = f'export_HO_TES_z{z}n{n}.txt'
        if summary_results in os.listdir():
            head_ = ", ".join([ExeAxial0D_EnergyMinimum.DTYPE.__name__,
                               'ho_b_length'])
            with open(summary_results, 'w+') as f:
                f.write(head_+'\n')
                
        for step_, b_len in enumerate(b_lengths):
            
            ## set up the Hamiltonian in the set up 
            hamil_exe = TBME_HamiltonianManager(b_len, MZmax, MZmin=MZmin)
            hamil_fn_new = f'D1S_t0_z{z}n{n}_MZ{MZmax}_b{1000*b_len:4.0f}'
            hamil_exe.hamil_filename = hamil_fn_new
            hamil_exe.setAndRun_D1Sxml()
        
            ## input args_for must change seeed=1 after the right minimum
            if step_ > 0:
                input_args_start[InputTaurus.ArgsEnum.seed] = 1
                        
            ## after the export (do not zip the files) import the results and 
            ## copy here to an auxiliary file
            
            line = ''        
            try:
                exe_ = ExeTaurus0D_EnergyMinimum(z, n, hamil_fn_new)
                exe_.setInputCalculationArguments(**input_args_start)
                exe_.defineDeformationRange(0,  0, 0)
                exe_.include_header_in_results_file = False
                exe_.setUp()
                exe_.setUpExecution(**input_args_onrun)
                exe_.force_converg = True 
                exe_.run()
                exe_.gobalTearDown(zip_bufolder=False)
                
                line = exe_._1stSeedMinima.getAttributesDictLike
            except ExecutionException as e:
                print(e)
            
            
            if line not in  ('', '\n'):
                # register the Optimal length
                if exe_._1stSeedMinima.E_HFB < b_Ehfb_min[1]:
                    b_Ehfb_min = (b_len, exe_._1stSeedMinima.E_HFB)
                
                with open(summary_results, 'a+') as f:
                    header_  = f"{len(b_lengths)-step_}: {b_len:5.3f}"
                    header_ += ExeTaurus0D_EnergyMinimum.HEADER_SEPARATOR
                    f.write(header_+line+'\n')
        
        optimal_lengths[(z, n)] = b_Ehfb_min
        ## plot in a file,
        Plotter1D_Taurus.FOLDER_PATH = ''
        plot = Plotter1D_Taurus(summary_results, attr2plot='E_HFB') # no testeado
        plot.defaultPlot(attr2plot='E_HFB')
    
    print("\n\n[DONE] Minimization completed, the optimal HO lenghts are:")
    prettyPrintDictionary(optimal_lengths)
                


def run_computingHOhbarOmegaForD1S_Axial(nucleus, program='HFBaxial',
                                         MZmax=4, bHO_min=1.5, bHO_max=2.75, 
                                         Nsteps=6):
    """ 
    Script for HFBAxial, get a curve b length to see where is the minimum of E hfb
        NOTE: MzMin is always 0
        HFBAxial cannot be imported from github
    """
    assert program in os.listdir(), \
         f"[ERROR] program[{program}] not found in cwd[{os.getcwd()}]"\
          ", compile and place it there"
    
    b_lengths = list(np.linspace(bHO_min, bHO_max, Nsteps, endpoint=True))
    b_lengths.reverse()
    
    input_args_start = {
        InputAxial.ArgsEnum.com : 2,
        InputAxial.ArgsEnum.seed: 2,
        InputAxial.ArgsEnum.iterations: 1000,
    }
    
    input_args_onrun = { **input_args_start,
        InputAxial.ArgsEnum.seed: 0,
        InputAxial.ArgsEnum.iterations: 1000,
    } # just get the minimum result    
    
    if InputAxial.PROGRAM != program:
        InputAxial.PROGRAM = program
    ExeAxial0D_EnergyMinimum.PRINT_CALCULATION_PARAMETERS = False
    optimal_lengths = {}
    
    for z, n in nucleus:
        b_Ehfb_min = (0, 999999)
        if z % 2 == 1 or n % 2 == 1:
            print(f"[WARNING] isotope :z{z}, n{n} is not even-even, no blocking applied")
            
        summary_results = f'export_HO_TES_axial_z{z}n{n}.txt'
        if summary_results in os.listdir():
            head_ = ", ".join([ExeAxial0D_EnergyMinimum.DTYPE.__name__,
                               'ho_b_length'])
            with open(summary_results, 'w+') as f:
                f.write(head_+'\n')
        
        for step_, b_len in enumerate(b_lengths):           
            input_args_start[InputAxial.ArgsEnum.b_len] = b_len
            ## input args_for must change seeed=1 after the right minimum
            if step_ > 0:
                input_args_start[InputAxial.ArgsEnum.seed] = 1
                        
            ## after the export (do not zip the files) import the results and 
            ## copy here to an auxiliary file
            
            line = ''
            try:
                exe_ = ExeAxial0D_EnergyMinimum(z, n, 0, MZmax)
                exe_.setInputCalculationArguments(**input_args_start)
                exe_.defineDeformationRange(0,  0, 0)
                exe_.include_header_in_results_file = False
                exe_.setUp(reset_folder= step_==0 )
                exe_.setUpExecution(**input_args_onrun)
                exe_.force_converg = True 
                exe_.run()
                exe_.gobalTearDown(zip_bufolder=False)
                                
                line = exe_._1stSeedMinima.getAttributesDictLike
            except ExecutionException as e:
                print(e)
            
            if line not in  ('', '\n'):
                # register the Optimal length
                if exe_._1stSeedMinima.E_HFB < b_Ehfb_min[1]:
                    b_Ehfb_min = (b_len, exe_._1stSeedMinima.E_HFB)
                
                with open(summary_results, 'a+') as f:
                    header_  = f"{len(b_lengths)-step_}: {b_len:5.3f}"
                    header_ += ExeTaurus0D_EnergyMinimum.HEADER_SEPARATOR
                    f.write(header_+line+'\n')
            
        optimal_lengths[(z, n)] = b_Ehfb_min
        ## plot in a file,
        Plotter1D_Axial.FOLDER_PATH = ''
        plot = Plotter1D_Axial(summary_results, attr2plot='E_HFB')
        
        plot.defaultPlot(attr2plot='E_HFB')
        plot.EXPORT_PDF_AND_MERGE = True
        plot.setExportFigureFilename(f"hoOptim_z{z}n{n}.pdf")
        
    print("\n\n[DONE] Minimization completed, the optimal HO lenghts are:")
    prettyPrintDictionary(optimal_lengths)

if __name__ == '__main__':
    pass