'''
Created on Jan 10, 2023

@author: Miguel

Module for script setting

'''
import os
import subprocess
import inspect
import shutil
import numpy as np
from numpy.random import randint
from copy import deepcopy, copy

from tools.inputs import InputTaurus, InputAxial
from tools.data import DataTaurus, DataAxial
from tools.helpers import LINE_2, LINE_1, prettyPrintDictionary, \
    zipBUresults, readAntoine
from tools.Enums import Enum, OutputFileTypes


class ExecutionException(BaseException):
    pass


class _Base1DTaurusExecutor(object):
    
    class IterativeEnum(Enum):
        SINGLE_EVALUATION  = "SINGLE_EVALUATION"   # do not iterate (constrained or not)
        EVEN_STEP_STD      = "EVEN_STEP_STD"       # complete the ranges once in order
        EVEN_STEP_SWEEPING = "EVEN_STEP_SWEEPING"  # do the ranges and come back
        VARIABLE_STEP      = "VARIABLE_STEP"       # do the ranges once with non-fixed step
        
    PRINT_STEP_RESULT = True
    ITERATIVE_METHOD  = None
    SAVE_DAT_FILES = []  # list for saving the auxillary files from taurus
    EXPORT_LIST_RESULTS = 'export_resultTaurus.txt'
    HEADER_SEPARATOR = ' ## '
    
    CONSTRAINT    : str = None # InputTaurus Variable to compute
    CONSTRAINT_DT : str = None # DataTaurus (key) Variable to compute
    
    DTYPE = DataTaurus  # DataType for the outputs to manage
    ITYPE = InputTaurus # Input type for the input management
    
    BLOCKING_SEEDS_RANDOMIZATION = 5 # Number of random blocking sp state for odd calculation
    
    # @classmethod
    # def __new__(cls, *args, **kwargs):
    #     """
    #     Reset the class attributes
    #     """
    #     pass
    
    def setUp(self, *args, **kwargs):
        raise ExecutionException("Abstract method, implement parameters for the execution")
    
    def __init__(self, z, n, interaction, *args, **kwargs):
        
        ## interactions and nucleus
        self.z : int = z
        self.n : int = n
        self.interaction : str = interaction
        
        self.inputObj  : self.ITYPE  = None
        self._DDparams : dict = None
        self._1stSeedMinima : self.DTYPE  = None
        self._current_result: self.DTYPE  = None ## TODO: might be useless, remove in that case
        
        self.activeDDterm = True
        self._output_filename = self.DTYPE.DEFAULT_OUTPUT_FILENAME
        
        self.deform_oblate   : list = []
        self.deform_prolate  : list = []
        self._deformations_map  : list = [[], []] #oblate, prolate
        self._curr_deform_index : int  = None
        self._deform_lim_max = None
        self._deform_lim_min = None
        self._deform_base    = None
        self._iter    = 0
        self._N_steps = 0
        self._preconvergence_steps = 0
        self._base_wf_filename = None
        
        self.include_header_in_results_file = True # set the export result to have the deformation header (i: Q[i]) ## data
        self.save_final_wf  = False # To set for the final wf to be copied as initial
        self.force_converg  = False # requirement of convergence for wf copy
        
        self._results : list = [[], []] # oblate, prolate
        if kwargs:
            for key_, value in kwargs.items():
                setattr(self, key_, value)
        
        self._checkExecutorSettings()
    
    def resetExecutorObject(self, keep_1stMinimum=False):
        """
        Clean deformations and results from a previous calculation,
        only instanced attributes generated during the run (inputObject, params, ... keept)
        could keep the first minimum data
        """
                
        self._current_result: self.DTYPE  = None ## TODO: might be useless, remove in that case
                
        self.deform_oblate   : list = []
        self.deform_prolate  : list = []
        self._deformations_map  : list = [[], []] #oblate, prolate
        self._curr_deform_index : int  = None
        
        self._iter    = 0
        self._N_steps = 0
        self._preconvergence_steps = 0
        
        if not keep_1stMinimum:
            
            self._1stSeedMinima : self.DTYPE  = None
            self._base_wf_filename = None
            
            self._deform_lim_max = None
            self._deform_lim_min = None
            self._deform_base    = None
            
    @property
    def numberParityOfIsotope(self):
        return (self.z % 2, self.n % 2)
    
    def _checkExecutorSettings(self):
        """
        Tests for the method to verify the calculation.
        """
        if ((self.CONSTRAINT != None) and 
            (self.CONSTRAINT not in self.ITYPE.ConstrEnum.members()) ):
            raise ExecutionException("Main constraint for the calculation is invalid"
                f", given [{self.CONSTRAINT}] but must be in Input***.ConstrEnum" )
        
        if (self.ITERATIVE_METHOD != self.IterativeEnum.SINGLE_EVALUATION):
            if (self.CONSTRAINT == None): 
                raise ExecutionException("All Iterative procedures must have a defined"
                                         " constraint, None set.")
            if (self.CONSTRAINT_DT == None):
                raise ExecutionException("Must be a defined observable for the result to",
                                         " evaluate the deformations. None set.")
        
    def setInputCalculationArguments(self, core_calc=False, axial_calc=False,
                                     **input_kwargs):
        """
        Set arguments or constraints to be STATIC in the calculation, 
        must be arguments from:
        InputTaurus.ArgsEnum:
            Global     : com, red_hamil, seed, interm_wf ...
            Gradient   : grad_type, iterations, grad_tol, eta_grad, mu_grad
            Def schemes: beta_schm, pair_schm
        InputTaurus.ConstrEnum
        
        Set up for special modes boolean (mutually exclusive):
            :axial_calculation: calls for pn-separated, b_l(m!=0) = 0 set up 
                                (no core calculation)
            :core_calculation: core calculation with same parity states require
                                odd l-multipole constraints to be 0.
        EXCEPTIONS:
        !! Executor constraint present in input_kwargs will raise an exception
        !! Values given will be checked by the inputSetter when instanced.
        """
        
        assert not(core_calc and axial_calc), ExecutionException(
            "No-core Axial and valence set ups are mutually exclusive")
        if self.CONSTRAINT in input_kwargs.keys():
            raise ExecutionException("The executor constraint must not be set static, remove it.")
        
        self.inputObj = self.ITYPE(self.z, self.n, self.interaction)
        
        if core_calc:
            self.inputObj.setUpValenceSpaceCalculation()
        elif axial_calc:
            self.inputObj.setUpNoCoreAxialCalculation()
        
        _check = [(hasattr(self.ITYPE.ArgsEnum,k) or 
                   hasattr(self.ITYPE.ConstrEnum,k)) for k in input_kwargs.keys()]
        if not all(_check):
            raise ExecutionException("One of the arguments is invalid for "
                f"{self.ITYPE.__name__}:\n {input_kwargs}")
        
        self.inputObj.setParameters(**input_kwargs)
        self._DDparams = self.inputObj._DD_PARAMS 
        # NOTE:  by assign the _DD (class) dictionary in input, changes in the
        # attribute _DDparams is transfered to the input.
    
    def setUpExecution(self, *args, **kwargs):
        """
        If the process require an starting point for other parameters to be
        defined. I.e:
            * Calculations that start from a minimum, and from it extracting the
              independent variable range and order.
            * 
        """
        raise ExecutionException("Abstract method, implement me!")
        
    def defineDeformationRange(self, min_, max_, N_steps):
        """
        Set the arrays for the deformations (prolate, oblate)
        deformation boundaries are required, only SINGLE_EVALUATION ignores them
        """
        if self.ITERATIVE_METHOD == self.IterativeEnum.SINGLE_EVALUATION:
            frame = inspect.currentframe()
            print(" [NOTE] Single evaluation process do not require this process [{}], continue"
                  .format(frame.f_code.co_name))
            return
        
        assert min_ < max_, "defining range for 1D where min >= max"
        N_steps = N_steps + 1 if (N_steps % 2 == 0) else N_steps
        
        self._deform_lim_max = float(max_)
        self._deform_lim_min = float(min_) # all constraints are float (parse)
        self._N_steps = N_steps
        if self.ITERATIVE_METHOD == self.IterativeEnum.VARIABLE_STEP:
            ## The results and deformations are unfixed, just between the boundaries
            return
        
        array_ = list(np.linspace(min_, max_, num=N_steps, endpoint=True))
        self.deform_oblate  = []
        self.deform_prolate = array_
        
        self._deformations_map[0] = []
        self._deformations_map[1] = list(enumerate(array_))
        #
    
    def _setDeformationFromMinimum(self, p_min, p_max, N_max):
        """ 
        TODO: Update and fix
        Complete the deformations to the left(oblate) and right(prol)
        for the pairing minimum. dq defined by N_max, final total length= N_max+1
        """
        if not (p_max and p_min) or N_max == 0:
            ## consider as ITERATIVE_METHOD SINGLE EVALUATION (save the deform)
            q0 = getattr(self._1stSeedMinima, self.CONSTRAINT_DT, None)
            self._deform_base = q0
            self.deform_prolate = [q0, ]
            self._deformations_map[1] = [(0, q0)]
            self._results[1].append(self._1stSeedMinima)
            return
        
        # p_min = max(p_min, 0.0)
        deform_oblate, deform_prolate = [], []
        dq = round((p_max - p_min) / N_max,  3)
        q0 = getattr(self._1stSeedMinima, self.CONSTRAINT_DT, None)
        if   q0 < p_min: # p0 outside to the left (add the difference to the left)
            p_min = q0 + (q0 - p_min)
        elif q0 > p_max: # p0 outside to the right (add the diff. to the right)
            p_max = q0 + (q0 - p_max)
        
        if  q0 == None: 
            print("[WARNING] _set_pair could not get the main Constraint [",
                  self.CONSTRAINT_DT, "]. Default setting it to 0.00")
            q0 = 0.00
        deform_prolate.append(q0)
        q = q0
        while (q >  p_min):
            q = q - dq
            if (q > p_min):
                deform_oblate.append(q)
            else:
                deform_oblate.append(p_min)
        q = q0
        while (q < p_max):
            q = q + dq
            if  (q < p_max):
                deform_prolate.append(q)
            else:
                deform_prolate.append(p_max)
                
        self._results[1].append(self._1stSeedMinima)
        self._deform_base   = q0
        ## Variable step fill the 
        if self.ITERATIVE_METHOD != self.IterativeEnum.VARIABLE_STEP:
            self.deform_oblate  = deform_oblate
            self.deform_prolate = deform_prolate
            
            self._deformations_map[0] = []
            for k, val in enumerate(deform_oblate):
                self._deformations_map[0].append( (-k - 1, val) )
            self._deformations_map[1] = list(enumerate(deform_prolate))
        
        
    
    def run(self):
        self._checkExecutorSettings()
        
        if self.ITERATIVE_METHOD == self.IterativeEnum.VARIABLE_STEP:
            self._runVariableStep()
        else:
            print(" ** oblate:")
            for k, val in self._deformations_map[0]: # oblate
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                self._results[0].append(self._executeProgram())
            if self.ITERATIVE_METHOD == self.IterativeEnum.EVEN_STEP_SWEEPING:
                self._run_backPropagationSweeping(oblate_part=True) 
            
            print(" ** prolate:")
            for k, val in self._deformations_map[1][1:]: # prolate
                ## exclude the first state since it is the original seed
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                self._results[1].append(self._executeProgram())
            
            if self.ITERATIVE_METHOD == self.IterativeEnum.EVEN_STEP_SWEEPING:
                self._run_backPropagationSweeping(oblate_part=False)
    
    def _run_backPropagationSweeping(self, oblate_part=None):
        """ 
        Method to reevaluate the execution limits backwards, 
        over-writable method to change the acceptance criteria (impose lower energy)
        """
        if oblate_part == None: 
            raise ExecutionException("Specify the oblate or prolate part")
           
        if oblate_part:
            print(" ** oblate (back):")
            for k, val in reversed(self._deformations_map[0]): # oblate
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                res  : self.DTYPE = self._executeProgram() # reexecuting
                
                indx_ = -k-1
                res0 : self.DTYPE = self._results[0][indx_]
                if self._backPropagationAcceptanceCriteria(res, res0):
                    self._results[0][indx_] =  res
        else:
            print(" ** prolate (back):")
            for k, val in reversed(self._deformations_map[1][1:]): # prolate
                ## exclude the first state since it is the original seed
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                res  : self.DTYPE = self._executeProgram()
                # indx_ = k - 1
                res0 : self.DTYPE = self._results[1][k]
                if self._backPropagationAcceptanceCriteria(res, res0):
                    self._results[1][k] =  res
    
    def _backPropagationAcceptanceCriteria(self, result, prev_result):
        """
        Order of acceptance:
            1. Properly_finished=True before None/False,
            2. lower energy in case of same properly_fin status
        
        Tested possible outputs (None results already tested)
               PROPER_FIN
                PREV CURR  E_prev  E_fin then:  ACCEPTED_STATUS
            0 0 True True   -30.0 -30.0      :  True
            0 1 True True   -30.0 -20.0      :  False
            0 2 True False  -30.0 -30.0      :  False
            0 3 True False  -30.0 -20.0      :  False
                PREV CURR  E_prev  E_fin then:  ACCEPTED_STATUS
            1 0 True True   -20.0 -30.0      :  True
            1 1 True True   -20.0 -20.0      :  True
            1 2 True False  -20.0 -30.0      :  False
            1 3 True False  -20.0 -20.0      :  False
                PREV CURR  E_prev  E_fin then:  ACCEPTED_STATUS
            2 0 False True  -30.0 -30.0      :  True
            2 1 False True  -30.0 -20.0      :  True
            2 2 False False -30.0 -30.0      :  True
            2 3 False False -30.0 -20.0      :  False
                PREV CURR  E_prev  E_fin then:  ACCEPTED_STATUS
            3 0 False True  -20.0 -30.0      :  True
            3 1 False True  -20.0 -20.0      :  True
            3 2 False False -20.0 -30.0      :  True
            3 3 False False -20.0 -20.0      :  True
        
        """
        if not isinstance(result,  self.DTYPE): 
            return False    # cannot accept a new broken result
        if isinstance(prev_result, self.DTYPE): 
            if   result.properly_finished and prev_result.properly_finished:
                return result.E_HFB <= prev_result.E_HFB
            elif prev_result.properly_finished and not result.properly_finished:
                return False 
            elif not prev_result.properly_finished and result.properly_finished:
                return True 
            else: # both wrong, so get the minimum
                return result.E_HFB <= prev_result.E_HFB
        else:
            return True
                
    def _energyDiffRejectionCriteria(self, curr_energ,  old_energ, old_e_diff, 
                                     tol_factor=2.0):
        """
        Criteria for Variable Step to accept a variation in energy, avoiding 
        the TES leaps, keeping the solution as far as possible in the same path. 
        """
        new_e_diff = curr_energ - old_energ
        # change in direction of the derivative, reject if difference is > 25%
        if new_e_diff * old_e_diff < 0: 
            return abs(new_e_diff) > 2.0 * abs(old_e_diff)
        # reject if new difference is tol_factor greater than the last one.
        return abs(new_e_diff) > tol_factor * abs(old_e_diff)
    
    def _runVariableStep(self):
        """
        Variable step in the oblate and prolate part.
        Fucntion inherit from legacy:
            TODO: Not Tested!!
        """
        if self._1stSeedMinima == None or self._base_wf_filename == None:
            raise ExecutionException("Running runVarialeStep without a first "
                                     "converged result, stop. \n Suggestion:"
                                     "Call setUpExecution")
        
        self.force_converg = True
        
        N_MAX   = 10 * self._N_steps
        dq_base = abs(self._deform_lim_max - self._deform_lim_min)/self._N_steps 
        ener_base     = float(self._1stSeedMinima.E_HFB)
        dqDivisionMax = 3
        
        self.inputObj.seed = 1 
        b20_base = self._deform_base
        print(f" ** Variable Step Running start point {self.CONSTRAINT}={b20_base:7.3f}")
        
        for prolate, b_lim in enumerate((self._deform_lim_min, 
                                         self._deform_lim_max)):
            shutil.copy(self._base_wf_filename, 'initial_wf.bin')
            
            b20_i  = b20_base
            energ  = ener_base
            curr_energ = 10.0   #  first oversized value
            e_diff     = 10.0   #  first oversized value
            self._iter = 0
            div = 0
            
            _whileCond = b20_i < b_lim if (prolate==1) else b20_i > b_lim
            while _whileCond and (self._iter < N_MAX):
                b20 = b20_i - ((-1)**(prolate)*(dq_base / (2**div)))
                
                self.inputObj.setConstraints(**{self.CONSTRAINT: b20})
                
                ## TODO: legacy_, here if the division were at the limit,
                ## the script call to a convergence loop which reduced the eta/mu 
                ## parameters: see _legacy.exe_q20pes_taurus._convergence_loop
                # if div > 3:
                #     res = _convergence_loop(kwargs, output_filename, b20,
                #                             save_final_wf=False, setDD=False)
                # else:
                res = self._executeProgram(base_execution=False)
                
                ## Case 1: the program broke and the result is NULL
                if res.broken_execution:
                    self._iter += 1
                    if div < dqDivisionMax:
                        # reject(increase division)
                        div += 1
                        print("  * reducing b20 increment(1): [{}] Ei{:9.2f} - Eim1{:9.2f} ={:8.5f} > {:8.5f}"
                              .format(div, curr_energ, energ, curr_energ-energ, e_diff))
                        continue
                    else:
                        # accept and continue (DONT copy final function)
                        # increase the step for valid or deformation precision overflow 
                        div = max(0, div - 1) ## smoothly recover the dq
                        e_diff = curr_energ - energ
                        energ = curr_energ
                        b20_i = b20
                        print("  * Failed but continue: DIV{} DIFF{:10.4f} ENER{:10.4f} B{:5.3f}"
                              .format(div, e_diff, energ, b20_i))
                        continue # cannot evaluate next Step or save results
                
                ## Case 2: the program did'nt broke and the result has values
                # take the E_HFB energy and compare the previous (acceptance criteria)
                curr_energ = float(res.E_HFB)
                self._iter += 1
                if ((div < dqDivisionMax)
                    and (self._energyDiffRejectionCriteria(curr_energ, energ, e_diff, 
                                                           tol_factor=2.5)
                         or (not res.properly_finished))):
                    # reject(increase division)
                    div += 1
                    print("  * reducing b20 increment(2)[i{}]: [{}] Ei{:9.2f} - Eim1{:9.2f} ={:8.5f} >  ({:8.5f}, {:8.5f})"
                          .format(self._iter,div, curr_energ, energ, curr_energ-energ, e_diff,
                                  3.0*e_diff, 2.0*e_diff))
                    continue
                else:
                    print("  * [OK] step accepted DIV:{} CE{:10.4} C.DIFF:{:10.4}"
                          .format(div, curr_energ, curr_energ - energ))
                    # accept and continue (copy final function)
                    _e = subprocess.call('cp final_wf.bin initial_wf.bin', shell=True)
                    # increase the step for valid or deformation precision overflow 
                    div =  max(0, div - 1) ## smoothly recover the dq
                    e_diff = curr_energ - energ
                    energ = curr_energ
                    b20_i = b20
                    print("  * [OK] WF directly copied  [i{}]: DIV:{} DIFF{:10.4f} ENER{:10.4f} B{:5.3f}"
                          .format(self._iter, div, e_diff, energ, b20_i))
                
                if prolate == 0: #grow in order [-.5, -.4, .., 0, ..., +.5]
                    self._results[0].insert(0, res) 
                    self._deformations_map[0].append((-len(self._results[0]), b20_i))
                else:
                    self._results[1].append(res)
                    self._deformations_map[1].append((len(self._results[1]), b20_i))
                
                self.include_header_in_results_file=True
                self.exportResults()
    
    def _auxWindows_executeProgram(self, output_fn):
        """ 
        Dummy method to test the scripts1d in Windows
        """
        # program = """ """
        # exec(program)
        file2copy = "aux_output_Z10N10_00_00.txy"
        file2copy = "aux_output_Z10N6_23.txt"
        # file2copy = 'aux_output_Z10N6_broken.txt'
        file2copy = "res_z18n18_dbase.OUT"
        
        txt = ''
        with open(file2copy, 'r') as f:
            txt = f.read()
        
        with open(output_fn, 'w+') as f:
            f.write(txt)
        
        ## wf intermediate
        with open('final_wf.bin', 'w+') as f:
            f.write("")
        if self.inputObj.interm_wf == 1:
            with open('intermediate_wf.bin', 'w+') as f:
                f.write("")
        ## simulates the other .dat prompt
        for file_ in  ("canonicalbasis", "eigenbasis_h", 
                       "eigenbasis_H11", "occupation_numbers"):
            txt = ""
            with open(f"{file_}_original.dat", 'r') as f:
                txt = f.read()
            with open(f"{file_}.dat", 'w+') as f:
                f.write(txt) 
    
    def saveFinalWFprocedure(self, base_execution=False):
        """ 
        Method to save wf and other objects from _executeProgram outputs
        Args
            :result: dataTaurus result of the calculation
            :base_execution: is to know if the execution were the starter minimum
            localization or if it is in the run process
        """
        if self._current_result.broken_execution: return
        
        if base_execution:
            self._base_wf_filename = 'base_initial_wf.bin'
            shutil.copy('final_wf.bin', self._base_wf_filename)
            tail = f"z{self.z}n{self.n}_dbase"
        else:
            tail = f"z{self.z}n{self.n}_d{self._curr_deform_index}"
        
        ## copy the wf to the initial wf always except non 
        if (not self.force_converg) or self._current_result.properly_finished:
            shutil.copy('final_wf.bin', 'initial_wf.bin')
        
        shutil.move(self._output_filename, 
                    f"{self.DTYPE.BU_folder}/res_{tail}.OUT")
        shutil.copy('final_wf.bin', 
                    f"{self.DTYPE.BU_folder}/seed_{tail}.bin")
        
        if self.SAVE_DAT_FILES:
            dat_files = filter(lambda x: x.endswith('.dat'), os.listdir())
            dat_files = filter(lambda x: x[:-4] in self.SAVE_DAT_FILES, dat_files)
            for _df in dat_files:
                _df2 = _df.replace('.dat', f'_{tail}.dat')
                shutil.copy(_df, f"{self.DTYPE.BU_folder}/{_df2}")
                if not os.getcwd().startswith('C:'):
                    shutil.move(_df, f"{self.DTYPE.BU_folder}/{_df2}")
        
        if os.getcwd().startswith('C:'): # Testing on windows
            f_delete = filter(lambda x: x.endswith('.dat') and not 'original' in x, 
                              os.listdir() )
            for f in f_delete:
                os.remove(f)
        else:
            _e = subprocess.call('rm *.dat *.gut', shell=True)
    
    
    def _executeProgram(self, base_execution=False):
        """
        TODO: Main execution method, prints.
        Input object and its dd-parameters must be already setted at this point
            :base_execution: indicates the minimization to be for the 1 seed
            (configure options to save the wave function)
        """
        assert self.inputObj != None, ExecutionException(
            "Trying to launch an execution without defined Input object.")
        
        res = None
        
        if self.activeDDterm and isinstance(self.inputObj, InputTaurus):
            with open(self.inputObj.INPUT_DD_FILENAME, 'w+') as f:
                f.write(self.inputObj.get_inputDDparamsFile())
        
        try:
            with open(self.inputObj.input_filename, 'w+') as f2:
                f2.write(self.inputObj.getText4file())
            
            _inp_fn = self.inputObj.input_filename
            _out_fn = self._output_filename
            
            if os.getcwd().startswith('C:'): ## Testing purpose 
                self._auxWindows_executeProgram(_out_fn)
            else:
                order_ =f'./{self.inputObj.PROGRAM} < {_inp_fn} > {_out_fn}'
                _e = subprocess.call(order_, 
                                     shell=True,
                                     timeout=43200) # 12 h timeout
            
            res = self.DTYPE(self.z, self.n, _out_fn)
            self._current_result = deepcopy(res)
            
            self.saveFinalWFprocedure(base_execution) ## TODO: here??
            self.executionTearDown   (res, base_execution)
            
        except Exception as e:
            raise e
            raise Exception("TODO: manage exceptions in taurus execution")
        
        return res
    
    def printTaurusResult(self, result : DataTaurus, print_head=False, 
                          *params2print):
        """
        Standard step information
        """
        HEAD = "  z  n  (st)        E_HFB        Kin     Pair       b2"
        if print_head:
            print('\n'+HEAD+LINE_2)
            return
        
        status_fin = 'X' if not result.properly_finished  else '.'
        _iter_str = "[{}/{}: {}']".format(result.iter_max, self.inputObj.iterations, 
                                          result.iter_time_seconds//60)
        
        txt  =" {:2} {:2}    {}      {:9.3f}  {:8.3f}  {:7.3f}   {:+6.3f} "
        txt = txt.format(result.z, result.n, status_fin, 
                         result.E_HFB, result.kin, result.pair, 
                         result.b20_isoscalar)
        print(txt, _iter_str)
        
    
    @property
    def calculationParameters(self):
        """
        TODO: Print properties of the calculation to know while running, 
            such as the input object, folders, set up properties, attributes ...
        """
        print(LINE_2)
        print(f" ** Executor 1D [{self.__class__.__name__}] Parameters:")
        print(LINE_1)
        priv_attr = ('_1stSeedMinima', '_DDparams', '_deform_base', 
                     '_N_steps', '_iter', '_output_filename')
        priv_attr = dict([(k,getattr(self, k, None)) for k in priv_attr])
        priv_attr = {'PRIVATE_ATTR:': priv_attr}
        pub_attr = dict(list(filter(
            lambda x: not (x[0].startswith('_') or isinstance(x[1], type)),
            self.__dict__.items() )))
        prettyPrintDictionary(pub_attr)
        prettyPrintDictionary(priv_attr)
        print(LINE_2)
    
    def executionTearDown(self, result : DataTaurus, base_execution, *args, **kwargs):
        """
        Proceedings to do after the execution of a single step.
            copying the wf and output to a folder, clean auxiliary files,
        """
        if self.PRINT_STEP_RESULT:
            self.printTaurusResult(result)
        
        if self.force_converg and not result.properly_finished:
            return
        
        if base_execution:
            self._1stSeedMinima = result
        else:
            ## save the intermediate Export File
            self.exportResults()
        
        
    def exportResults(self, output_filename=None):
        """
        writes a text file with the results dict-like (before the program ends)
        Order: oblate[-N, -N-1, ..., -1] > prolate [0, 1, ..., N']
        """
        res2print = []
        for part_ in (0, 1):
            for indx, res in enumerate(self._results[part_]):
                key_, dval = self._deformations_map[part_][indx]
                line = []
                if self.include_header_in_results_file:
                    line.append(f"{key_:5}: {dval:+6.3f}")
                line.append(res.getAttributesDictLike)
                line = self.HEADER_SEPARATOR.join(line)
                
                if part_ == 1: ## prolate_ (add to the final)
                    res2print.append(line)
                else:
                    res2print.insert(0, line)
        
        txt_ = '\n'.join(res2print) 
        
        if output_filename == None:
            output_filename = self.EXPORT_LIST_RESULTS
        
        with open(output_filename, 'w+') as f:
            f.write(txt_)
        
    
    def gobalTearDown(self, *args, **kwargs):
        """
        Proceedings for the execution to do. i.e:
            zipping files, launch tests on the results, plotting things
        """
        raise ExecutionException("Abstract Method, implement me!")


class _Base1DAxialExecutor(_Base1DTaurusExecutor):
    
    EXPORT_LIST_RESULTS = 'export_resultAxial.txt'
    
    DTYPE = DataAxial  # DataType for the outputs to manage
    ITYPE = InputAxial # Input type for the input management

#===============================================================================
#
#    EXECUTOR DEFINITIONS: MULTIPOLE DEFORMATIONS 
#
#===============================================================================

class ExeTaurus1D_DeformQ20(_Base1DTaurusExecutor):
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.EVEN_STEP_SWEEPING
    
    CONSTRAINT    = InputTaurus.ConstrEnum.b20
    CONSTRAINT_DT = DataTaurus.getDataVariable(InputTaurus.ConstrEnum.b20,
                                               beta_schm = 0)
    
    EXPORT_LIST_RESULTS = 'export_TESq20'
        
    def setUp(self):
        """
        set up: 
            * back up folder for results
            * dumping filename
            * save the hamiltonian files in BU folder for recovery
        """
        
        self._DDparams = self.inputObj._DD_PARAMS
        self.DTYPE.BU_folder = f'BU_folder_{self.interaction}_z{self.z}n{self.n}'
        self.DTYPE.setUpFolderBackUp()
        
        for ext_ in OutputFileTypes.members():
            if os.path.exists(self.interaction+ext_):
                shutil.copy(self.interaction+ext_,  self.DTYPE.BU_folder)
        
    
    def setUpExecution(self, *args, **kwargs):
        """
        base solution pre-convergence.
        *   to change after the execution, put by InputTaurus.*Enum new values 
            as keyword arguments.
        """
        
        self.calculationParameters
        
        res = None
        self._preconvergence_steps = 0
        self.printTaurusResult(None, print_head=True)
        
        if 1 in self.numberParityOfIsotope:
            ## procedure to evaluate odd-even nuclei
            self._oddNumberParitySeedConvergence()
        else:
            ## even-even general case
            while not self._preconvergenceAccepted(res):
                res = self._executeProgram(base_execution=True)
        
        ## negative values might be obtained        
        self._setDeformationFromMinimum(self._deform_lim_min, 
                                        self._deform_lim_max, self._N_steps)
        
        _new_input_args = dict(filter(lambda x: x[0] in self.ITYPE.ArgsEnum.members(), 
                                      kwargs.items() ))
        _new_input_cons = dict(filter(lambda x: x[0] in self.ITYPE.ConstrEnum.members(), 
                                      kwargs.items() ))
        self.inputObj.setParameters(**_new_input_args, **_new_input_cons)
        
    
    def _oddNumberParitySeedConvergence(self):
        """
        TODO: procedure to select the sp state to block with the lower energy
        """
        ## get a sp_space for the state to block 
        odd_p, odd_n = self.numberParityOfIsotope
        # the hamiltonian is already copied in CWD for execution
        sh_states, l_ge_10 = [], True
        with open(self.interaction+OutputFileTypes.sho, 'r') as f:
            data = f.readlines()
            hmty = data[1].strip().split()
            if int(hmty[0]) ==  1:
                sh_states = hmty[2:] # Antoine_ v.s. hamiltonians 
                l_ge_10 = False
            else:
                line = data[2].strip().split()
                sh_states = line[1:]
        sh_states = [int(st) for st in sh_states]
        
        ## construct sp_dim for index randomization (sh_state, deg(j))
        sp_states = map(lambda x: (int(x), readAntoine(x, l_ge_10)[2] + 1), sh_states)
        sp_states = dict(list(sp_states))
        sp_dim    = sum(list(sp_states.values()))
        
        ## randomization of the blocked state and repeat the convergence
        ## several times to get the lower energy
        blocked_states  = []
        blocked_sh_states     = {}
        blocked_seeds_inputs  = {}
        blocked_seeds_results = {}
        blocked_energies      = {}
        bk_min, bk_E_min      = 0, 1.0e+69
        print("  ** Blocking minimization process (random sp-st 2 block). MAX iter=", 
              self.BLOCKING_SEEDS_RANDOMIZATION, " #-par:", self.numberParityOfIsotope, LINE_2)
        for rand_step in range(self.BLOCKING_SEEDS_RANDOMIZATION):
            bk_sp_p, bk_sp_n = 0, 0
            bk_sh_p, bk_sh_n = 0, 0
            bk_sp, bk_sh = None, None
            if odd_p:
                bk_sh_p = randint(0, len(sh_states))
                cdim = sum([sp_states[sh_states[k]] for k in range(bk_sh_p)])
                bk_sp_p = cdim + randint(1, sp_states[sh_states[bk_sh_p]] +1)
                bk_sp, bk_sh = bk_sp_p, sh_states[bk_sh_p]
            if odd_n:
                bk_sh_n = randint(0, len(sp_states))
                cdim = sum([sp_states[sh_states[k]] for k in range(bk_sh_n)])
                cdim += sp_dim
                bk_sp_n = cdim + randint(1, sp_states[sh_states[bk_sh_n]] +1)
                bk_sp = (bk_sp, bk_sp_n) if bk_sp else bk_sp_n
                bk_sh = (bk_sh, sh_states[bk_sh_n]) if bk_sh else sh_states[bk_sh_n]
            
            if bk_sp in blocked_states:
                print(rand_step, f"  * Blocked state [{bk_sp}] is already calculated [SKIP]")
                continue
            self.inputObj.qp_block = bk_sp if type(bk_sp)==int else [*bk_sp]
            
            blocked_states.append(bk_sp)
            blocked_sh_states[bk_sp] = bk_sh
            
            blocked_seeds_inputs [bk_sp] = deepcopy(self.inputObj)
            blocked_seeds_results[bk_sp] = None
            blocked_energies     [bk_sp] = 4.20e+69
            
            self._preconvergence_steps = 0
            self._1stSeedMinima = None
            
            res = None
            while not self._preconvergenceAccepted(res):
                res = self._executeProgram(base_execution=True)
            
            blocked_seeds_results[bk_sp] = deepcopy(res)
            blocked_energies     [bk_sp] = res.E_HFB
            
            ## actualize the minimum result
            if res.E_HFB < bk_E_min:
                bk_min, bk_E_min = bk_sp, res.E_HFB
        
            print(rand_step, f"  * Blocked state [{bk_sp}] done, Ehfb={res.E_HFB:6.3f}")
        
        print("\n  ** Blocking minimization process [FINISHED], Results:")
        print(f"  [  sp-state]  [    shells    ]   [ E HFB ]  sp/sh_dim={sp_dim}, {len(sp_states)}")
        for bk_st in blocked_states:
            print(f"  {str(bk_st):>12}  {str(blocked_sh_states[bk_st]):>16}   "
                  f"{blocked_energies[bk_st]:>9.4f}")
        print("  ** importing the state(s)", bk_min, "with energy ", bk_E_min)
        print(LINE_2)
        
        ## after the convegence, remove the blocked states and copy the 
        # copy the lowest energy solution and output.
        self.inputObj.qp_block = 0
        self._1stSeedMinima = blocked_seeds_results[bk_min]
        
    
    def _preconvergenceAccepted(self, result: DataTaurus):
        """
        define the steps to accept the result.
            Modification: it there is a _1sSeedMinima and _preconvergence_steps=0
            skip (to run another constraint from a previous minimum)
        """
        
        if self._preconvergence_steps == 0 and self._1stSeedMinima != None:
            self.inputObj.seed = 1
            shutil.copy(self._base_wf_filename, 'initial_wf.bin')
            return True 
        
        self._preconvergence_steps += 1
        MAX_REPETITIONS = 4
        str_rep = f"[{self._preconvergence_steps} / {MAX_REPETITIONS}]"
        
        if self._preconvergence_steps > MAX_REPETITIONS:
            ## iteration for preconvergence stops
            raise ExecutionException(f" !! {str_rep} Could not converge to the "
                                     "initial wf., execution process stops.")
        
        if result == None or result.broken_execution:
            print(f" ** {str_rep} Seed convergence procedure:")
            return False
        else:
            ## there is no critical problem for the result, might be garbage or
            ## not enough iterations, the result is copied and changed the input
            ## to import it (if result is crazy the process will stop in if #1)
            self.inputObj.seed = 1
            shutil.copy('final_wf.bin', 'initial_wf.bin')
            if result.properly_finished:
                print(f" ** {str_rep} Seed convergence procedure [DONE]:")
            else:
                print(f" ** {str_rep} Seed convergence procedure [FAIL]: repeating ")
            return result.properly_finished
        
        return False # if not valid 2 repeat
    
    def saveFinalWFprocedure(self, base_execution=False):
        _Base1DTaurusExecutor.saveFinalWFprocedure(self, base_execution)
    
    def run(self):
        ## TODO: might require aditional changes
        _Base1DTaurusExecutor.run(self)
    
    def gobalTearDown(self, zip_bufolder=True, *args, **kwargs):
        """
        Proceedings for the execution to do. i.e:
            zipping files, launch tests on the results, plotting things
        """
        if zip_bufolder:
            args = []
            if self.CONSTRAINT != None:
                args.append(self.CONSTRAINT)
                
            zipBUresults(DataTaurus.BU_folder, self.z, self.n, self.interaction,
                         *args)
    


class ExeAxial1D_DeformQ20(ExeTaurus1D_DeformQ20):
    
    CONSTRAINT    = InputAxial.ConstrEnum.b20
    CONSTRAINT_DT = DataAxial .getDataVariable(InputAxial.ConstrEnum.b20,
                                               beta_schm = 0)
    EXPORT_LIST_RESULTS = 'exportAx_TESq20'
    
    EXPORT_LIST_RESULTS = 'export_resultAxial.txt'
    
    DTYPE = DataAxial  # DataType for the outputs to manage
    ITYPE = InputAxial # Input type for the input management



class ExeTaurus1D_DeformB20(ExeTaurus1D_DeformQ20):
    
    CONSTRAINT    = InputTaurus.ConstrEnum.b20
    CONSTRAINT_DT = DataTaurus.getDataVariable(InputTaurus.ConstrEnum.b20,
                                               beta_schm = 1)
    EXPORT_LIST_RESULTS = 'export_TESb20'
    
class ExeAxial1D_DeformB20(ExeTaurus1D_DeformB20):
    
    CONSTRAINT    = InputAxial.ConstrEnum.b20
    CONSTRAINT_DT = DataAxial .getDataVariable(InputAxial.ConstrEnum.b20,
                                               beta_schm = 1)
    
    EXPORT_LIST_RESULTS = 'export_resultAxial.txt'
    
    DTYPE = DataAxial  # DataType for the outputs to manage
    ITYPE = InputAxial # Input type for the input management

#===============================================================================
#
#    EXECUTOR DEFINITIONS: ANGULAR MOMENTUM DEFORMATIONS 
#
#===============================================================================

class ExeTaurus1D_AngMomentum(ExeTaurus1D_DeformB20):
    
    """
    Evaluate <J_i> energy surfaces, Require setting the constraint via 
    classmethod setPairConstraint
    
    (since it contains all same steps than ExeTaurus1D_DeformB20, we inherit it)
    """
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.EVEN_STEP_SWEEPING
    
    CONSTRAINT    = None
    CONSTRAINT_DT = None
    
    EXPORT_LIST_RESULTS = 'export_TES_J'
    
    @classmethod
    def setAngularMomentumConstraint(cls, j_constr):
        """
        launch before the program to set the pairing constraint 
        (unset to prompt an exception to avoid default constraint set up)
        """
        assert j_constr in InputTaurus.ConstrEnum.members(), \
            ExecutionException("Constraint must be in DataTaurus.ConstrEnum")
        assert j_constr in (InputTaurus.ConstrEnum.Jx, 
                            InputTaurus.ConstrEnum.Jy, 
                            InputTaurus.ConstrEnum.Jz), \
            ExecutionException("must give Jx, Jy or Jz as constraint")
        
        cls.CONSTRAINT    = j_constr
        cls.CONSTRAINT_DT = DataTaurus.getDataVariable(j_constr, beta_schm=0)
        
        cls.EXPORT_LIST_RESULTS = f'export_TES_{j_constr}'
        DataTaurus.BU_folder = f'export_TES_{j_constr}'
    

#===============================================================================
#
#    EXECUTOR DEFINITIONS: PAIR-COUPLING DEFORMATIONS 
#
#===============================================================================

class ExeTaurus1D_PairCoupling(ExeTaurus1D_DeformB20):
    
    """
    Evaluate Pair-coupling energy surfaces, Require setting the constraint via 
    classmethod setPairConstraint
    
    (since it contains all same steps than ExeTaurus1D_DeformB20, we inherit it)
    """
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.EVEN_STEP_SWEEPING
    
    CONSTRAINT    = None
    CONSTRAINT_DT = None
    
    EXPORT_LIST_RESULTS = 'export_TESpair'
    
    @classmethod
    def setPairConstraint(cls, pair_constr):
        """
        launch before the program to set the pairing constraint 
        (unset to prompt an exception to avoid default constraint set up)
        """
        assert pair_constr in InputTaurus.ConstrEnum.members(), \
            ExecutionException("Pair constraint must be in DataTaurus.ConstrEnum")
        assert pair_constr.startswith('P_T'), \
            ExecutionException("must give a pair-coupling constraint")
        
        cls.CONSTRAINT = pair_constr
        cls.CONSTRAINT_DT = DataTaurus.getDataVariable(pair_constr, beta_schm=0)
        
        cls.EXPORT_LIST_RESULTS = f'export_TES_{pair_constr}'
        DataTaurus.BU_folder = f'export_TES_{pair_constr}'
    

#===============================================================================
#
#    EXECUTOR DEFINITIONS: GENERATE HAMILTONIAN 
#
#===============================================================================

class ExeTaurus0D_EnergyMinimum(ExeTaurus1D_DeformB20):
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.SINGLE_EVALUATION
    
    CONSTRAINT = None
    ## default value to see, 
    CONSTRAINT_DT = DataTaurus.getDataVariable(InputTaurus.ConstrEnum.b20,
                                               beta_schm = 1)
    
    EXPORT_LIST_RESULTS = 'export_HOminimums'
    
    """ Finish, just get _1sMinimum and export to a file"""    
    
    
    
