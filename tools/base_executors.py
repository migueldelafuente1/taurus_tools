'''
Created on 5 abr 2024

@author: delafuente
'''
import os
import subprocess
import inspect
import shutil
import numpy as np
from copy import deepcopy, copy
from random import random

from tools.inputs import InputTaurus, InputAxial, InputTaurusPAV, InputTaurusMIX
from tools.data import DataTaurus, DataAxial, DataTaurusPAV, DataTaurusMIX
from tools.helpers import LINE_2, LINE_1, prettyPrintDictionary, \
    OUTPUT_HEADER_SEPARATOR, LEBEDEV_GRID_POINTS, readAntoine, printf
from tools.Enums import Enum, OutputFileTypes
from scripts1d.script_helpers import parseTimeVerboseCommandOutputFile



class ExecutionException(BaseException):
    pass


class _Base1DTaurusExecutor(object):
    
    class IterativeEnum(Enum):
        SINGLE_EVALUATION  = "SINGLE_EVALUATION"   # do not iterate (constrained or not)
        EVEN_STEP_STD      = "EVEN_STEP_STD"       # complete the ranges once in order
        EVEN_STEP_SWEEPING = "EVEN_STEP_SWEEPING"  # do the ranges and come back
        VARIABLE_STEP      = "VARIABLE_STEP"       # do the ranges once with non-fixed step
        
    PRINT_STEP_RESULT = True
    PRINT_CALCULATION_PARAMETERS = True
    ITERATIVE_METHOD  = None
    MAX_ITER_WAITING_TIME = 43200   # 12 h timeout 
    SAVE_DAT_FILES        = []      # list for saving the auxillary files from taurus
    EXPORT_LIST_RESULTS   = 'export_resultTaurus.txt'
    HEADER_SEPARATOR = OUTPUT_HEADER_SEPARATOR
    
    TRACK_TIME_AND_RAM = False
    TRACK_TIME_FILE = '_time_program.log'
    
    CONSTRAINT    : str = None # InputTaurus Variable to compute
    CONSTRAINT_DT : str = None # DataTaurus (key) Variable to compute
    
    DTYPE = DataTaurus  # DataType for the outputs to manage
    ITYPE = InputTaurus # Input type for the input management
    
    DO_BASE_CALCULATION   = True  # Do at least 1 minimization (indep of SEEDS_RAND) 
    SEEDS_RANDOMIZATION   = 5     # Number of random seeds for even-even calculation / 
                                  # ALSO: Number of blocking sp state for odd calculation
    GENERATE_RANDOM_SEEDS = False # If set to false, it will not re-fix the deform grid
    IGNORE_SEED_BLOCKING  = False # Set True to do false odd-even nuclear surfaces
    RUN_PROJECTION        = False # Set True to project PAV the MF - results
    
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
        self.inputObj_PAV   : InputTaurusPAV  = None
        self._1stSeedMinima : self.DTYPE  = None
        self._current_result: self.DTYPE  = None
        self._curr_PAV_result : DataTaurusPAV = None
        
        self.activeDDterm = True
        self.axialSymetryRequired = False ## set up to reject non-axial results
        self.sphericalSymmetryRequired = False
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
        self._base_seed_type = None
        
        self._final_bin_list_data = [{}, {}] # List of the names for the saved files to export
        
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
        
        self._results = [[], []]
        
        if not keep_1stMinimum:
            
            self._1stSeedMinima : self.DTYPE  = None
            self._base_wf_filename = None
            
            self._deform_lim_max = None
            self._deform_lim_min = None
            self._deform_base    = None
            
    @property
    def numberParity(self):
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
        if self._base_seed_type == None:
            printf("[WARNING Executor] Seed was not defined, it will be copied (s=1).")
        
    def setInputCalculationArguments(self, core_calc=False, axial_calc=False,
                                     spherical_calc=False,
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
        
        ## read the properties of the basis for the interaction
        self._getStatesAndDimensionsOfHamiltonian()
        
        assert not(core_calc and axial_calc), ExecutionException(
            "No-core Axial and valence set ups are mutually exclusive")
        if self.CONSTRAINT in input_kwargs.keys():
            raise ExecutionException("The executor constraint must not be set static, remove it.")
        
        self.inputObj = self.ITYPE(self.z, self.n, self.interaction)
        
        if core_calc:
            self.inputObj.setUpValenceSpaceCalculation()
        elif axial_calc or spherical_calc:
            self.inputObj.setUpNoCoreAxialCalculation()
            self.axialSymetryRequired = True
            if spherical_calc:
                self.sphericalSymmetryRequired = True
                input_kwargs[InputTaurus.ArgsEnum.seed] = 2
            
            ## check if there is only one parity-shell states to supress odd-Q constraints
            parities = [readAntoine(i, l_ge_10=True)[1] for i in self._sh_states]
            parities = set([(-1)**i for i in parities])
            if len(parities) == 1:
                for q_const in self.inputObj.ConstrEnum.members():
                    if q_const.startswith('b1') or q_const.startswith('b3'):
                        setattr(self.inputObj, q_const, None)                
        
        _check = [(hasattr(self.ITYPE.ArgsEnum,k) or 
                   hasattr(self.ITYPE.ConstrEnum,k)) for k in input_kwargs.keys()]
        if not all(_check):
            raise ExecutionException("One of the arguments is invalid for "
                f"{self.ITYPE.__name__}:\n {input_kwargs}")
        
        self.inputObj.setParameters(**input_kwargs)
        if isinstance(self.ITYPE, InputTaurus):
            self._DDparams = self.inputObj._DD_PARAMS
        self._base_seed_type = self.inputObj.seed
        # NOTE:  by assign the _DD (class) dictionary in input, changes in the
        # attribute _DDparams is transfered to the input.
    
    def _getStatesAndDimensionsOfHamiltonian(self):
        """
        Read the hamiltonian and get the sp states/shell for the calculation
        """
        ## the hamiltonian is already copied in CWD for execution
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
        
        self._sh_states = sh_states
        self._sp_states = sp_states
        self._sp_dim    = sp_dim
        
        sp_2j = dict(map(lambda x: (int(x), readAntoine(x, l_ge_10)[2]), sh_states))
        self._sp_2jmax = max(sp_2j.values())
        self._sp_2jmin = min(sp_2j.values())
        
        sp_n = dict(map(lambda x: (int(x), readAntoine(x, l_ge_10)[0]), sh_states))
        self._sp_n_max = max(sp_n.values())
        self._sp_n_min = min(sp_n.values())
    
    def setUpExecution(self, *args, **kwargs):
        """
        If the process require an starting point for other parameters to be
        defined. I.e:
            * Calculations that start from a minimum, and from it extracting the
              independent variable range and order.
            * 
        """
        raise ExecutionException("Abstract method, implement me!")
    
    def setUpProjection(self, **params):
        """
        Defines the parameters for the projection of the nucleus.
        The z, n, interaction, com, Fomenko-points, and Jvals from the program
        """
        raise ExecutionException("Abstract method, implement me!")
    
    def runProjection(self, **params):
        """
        After a VAP-HFB result, project using taurus_pav.exe (GITHUB/project-taurus)
        """
        raise ExecutionException("Abstract method, implement me!")
    
    def _auxWindows_executeProgram_PAV(self, output_fn):
        """ 
        Dummy method to test the scripts1d - PAV in Windows
        """
        FLD_TEST_ = 'data_resources/testing_files/'
        file2copy = FLD_TEST_+'TEMP_res_PAV_z2n1_odd.txt'
        # file2copy = FLD_TEST_+'TEMP_res_PAV_z8n9_1result.txt'
        # file2copy = FLD_TEST_+'TEMP_res_PAV_z12n19_nan_norm_components.txt'
        
        txt = ''
        with open(file2copy, 'r') as f:
            txt = f.read()        
        with open(output_fn, 'w+') as f:
            f.write(txt)
    
    def projectionExecutionTearDown(self):
        """
        Process to save result after calling runProjection()
        """
        raise ExecutionException("Abstract method, implement me!")
    
    def defineDeformationRange(self, min_, max_, N_steps):
        """
        Set the arrays for the deformations (prolate, oblate)
        deformation boundaries are required, only SINGLE_EVALUATION ignores them
        """
        if self.ITERATIVE_METHOD == self.IterativeEnum.SINGLE_EVALUATION:
            frame = inspect.currentframe()
            printf(" [NOTE] Single evaluation process do not require this process [{}], continue"
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
        Complete the deformations to the left(oblate) and right(prol)
        for the pairing minimum. dq defined by N_max, final total length= N_max+1
        """
        
        if not (p_max!=None and p_min!=None) or N_max == 0:
            ## consider as ITERATIVE_METHOD SINGLE EVALUATION (save the deform)
            q0 = getattr(self._1stSeedMinima, self.CONSTRAINT_DT, 0.0)
            self._deform_base = q0
            self.deform_prolate = [q0, ]
            self._deformations_map[1] = [(0, q0)]
            self._results[1].append(self._1stSeedMinima)
            return
        
        # p_min = max(p_min, 0.0)
        deform_oblate, deform_prolate = [], []
        dq = round((p_max - p_min) / N_max,  3)
        q0 = getattr(self._1stSeedMinima, self.CONSTRAINT_DT, 0.0)
        if   q0 < p_min: # p0 outside to the left (add the difference to the left)
            p_min = q0 + (q0 - p_min)
        elif q0 > p_max: # p0 outside to the right (add the diff. to the right)
            p_max = q0 + (q0 - p_max)
        
        if  q0 == None: 
            printf("[WARNING] _set_pair could not get the main Constraint [",
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
        
    def _acceptanceCriteria(self, result):
        """
        Additional criteria to require certain properties after running.
        """
        if self.sphericalSymmetryRequired:
            return result.isAxial(and_spherical=True)
        if self.axialSymetryRequired:
            return result.isAxial()
        return True
    
    def _runUntilConvergence(self, MAX_STEPS=3):
        """
        Option to converge a result 
        """
        res : self.DTYPE = None
        res = self._executeProgram()
        
        if not self.force_converg or MAX_STEPS==0:
            return  res
        
        if res == None:
            printf(f"         [WRN] result broken or invalid, repeating [{MAX_STEPS}/tot]")
            res = self._runUntilConvergence(MAX_STEPS-1)
        elif not res.properly_finished:
            printf(f"         [WRN] result not finished, continue from last point [{MAX_STEPS}/tot]")
            _base_seed = self.inputObj.seed
            self.inputObj.seed = 1
            res = self._runUntilConvergence(MAX_STEPS-1)
            self.inputObj.seed = _base_seed
        elif not self._acceptanceCriteria(res):
            ## not axial calculation
            printf(f"         [WRN] result is not axial, repeating smaller eta-mu G0 [{MAX_STEPS}/tot]")
            self.inputObj.grad_type = 0
            self.inputObj.eta_grad  = min(0.03, self.inputObj.eta_grad - 0.007)
            self.inputObj.mu_grad   = min(0.2, self.inputObj.mu_grad - 0.01)
            res = self._runUntilConvergence(MAX_STEPS-1)
        return res
    
    def run(self):
        self._checkExecutorSettings()
        
        if self.ITERATIVE_METHOD == self.IterativeEnum.VARIABLE_STEP:
            self._runVariableStep()
        else:
            printf(" ** oblate:")
            for k, val in self._deformations_map[0]: # oblate
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                
                res : self.DTYPE = self._runUntilConvergence()
                # if self.force_converg:
                #     res : self.DTYPE = self._runUntilConvergence()
                # else:
                #     res : self.DTYPE = self._executeProgram()
                                
                self._results[0].append(res)
                self._final_bin_list_data[0][k] = res._exported_filename
            
            if self.ITERATIVE_METHOD == self.IterativeEnum.EVEN_STEP_SWEEPING:
                self._run_backwardsSweeping(oblate_part=True)
            start_from_ = 0
            printf(" ** prolate:   start_from_ = ", start_from_, "")
            for k, val in self._deformations_map[1][start_from_:]: # prolate
                ## exclude the first state since it is the original seed
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                res : self.DTYPE = self._runUntilConvergence()
                
                self._results[1].append(res)
                self._final_bin_list_data[1][k] = res._exported_filename
                # self._results[1].append(self._executeProgram())
            
            if self.ITERATIVE_METHOD == self.IterativeEnum.EVEN_STEP_SWEEPING:
                self._run_backwardsSweeping(oblate_part=False)
    
    def _run_backwardsSweeping(self, oblate_part=None):
        """ 
        Method to reevaluate the execution limits backwards, 
        over-writable method to change the acceptance criteria (impose lower energy)
        """
        if oblate_part == None: 
            raise ExecutionException("Specify the oblate or prolate part")
           
        if oblate_part:
            printf(" ** oblate (back):")
            for k, val in reversed(self._deformations_map[0]): # oblate
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                res  : self.DTYPE = self._executeProgram() # reexecuting
                
                indx_ = -k-1
                res0 : self.DTYPE = self._results[0][indx_]
                if self._backPropagationAcceptanceCriteria(res, res0):
                    self._results[0][indx_] =  res
                    self._final_bin_list_data[0][k] = res._exported_filename
        else:
            printf(" ** prolate (back):")
            ## exclude the first state since it is the original seed
            ## UPDATE, repeat that solution, the 1st minimum could not be the exact one 
            start_from_ = 0
            for k, val in reversed(self._deformations_map[1][start_from_:]): # prolate
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                res  : self.DTYPE = self._executeProgram()
                # indx_ = k - 1
                res0 : self.DTYPE = self._results[1][k]
                if self._backPropagationAcceptanceCriteria(res, res0):
                    self._results[1][k] =  res
                    self._final_bin_list_data[1][k] = res._exported_filename
    
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
            if not self._acceptanceCriteria(result): 
                return False ## the result do not have the correct symmetry
            
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
        printf(f" ** Variable Step Running start point {self.CONSTRAINT}={b20_base:7.3f}")
        
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
                
                self.inputObj.setConstraints(**{self.CONSTRAINT: b20,})
                
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
                        printf("  * reducing b20 increment(1): [{}] Ei{:9.2f} - Eim1{:9.2f} ={:8.5f} > {:8.5f}"
                              .format(div, curr_energ, energ, curr_energ-energ, e_diff))
                        continue
                    else:
                        # accept and continue (DONT copy final function)
                        # increase the step for valid or deformation precision overflow 
                        div = max(0, div - 1) ## smoothly recover the dq
                        e_diff = curr_energ - energ
                        energ = curr_energ
                        b20_i = b20
                        printf("  * Failed but continue: DIV{} DIFF{:10.4f} ENER{:10.4f} B{:5.3f}"
                              .format(div, e_diff, energ, b20_i))
                        continue # cannot evaluate next Step or save results
                
                ## Case 2: the program did'nt broke and the result has values
                # take the E_HFB energy and compare the previous (acceptance criteria)
                curr_energ = float(res.E_HFB)
                self._iter += 1
                _args = [curr_energ, energ, e_diff]
                if ((div < dqDivisionMax)
                    and (self._energyDiffRejectionCriteria(*_args, tol_factor=2.5)
                         or (self._acceptanceCriteria(res))
                         or (not res.properly_finished))):
                    # reject(increase division)
                    div += 1
                    printf("  * reducing b20 increment(2)[i{}]: [{}] Ei{:9.2f} - Eim1{:9.2f} ={:8.5f} >  ({:8.5f}, {:8.5f})"
                          .format(self._iter,div, curr_energ, energ, curr_energ-energ, e_diff,
                                  3.0*e_diff, 2.0*e_diff))
                    continue
                else:
                    printf("  * [OK] step accepted DIV:{} CE{:10.4} C.DIFF:{:10.4}"
                          .format(div, curr_energ, curr_energ - energ))
                    # accept and continue (copy final function)
                    _e = subprocess.call('cp final_wf.bin initial_wf.bin', shell=True)
                    # increase the step for valid or deformation precision overflow 
                    div =  max(0, div - 1) ## smoothly recover the dq
                    e_diff = curr_energ - energ
                    energ = curr_energ
                    b20_i = b20
                    printf("  * [OK] WF directly copied  [i{}]: DIV:{} DIFF{:10.4f} ENER{:10.4f} B{:5.3f}"
                          .format(self._iter, div, e_diff, energ, b20_i))
                
                if prolate == 0: #grow in order [-.5, -.4, .., 0, ..., +.5]
                    self._results[0].insert(0, res) 
                    self._deformations_map[0].append((-len(self._results[0]), b20_i))
                    self._final_bin_list_data[0][self._iter] = res._exported_filename
                else:
                    self._results[1].append(res)
                    self._deformations_map[1].append((len(self._results[1]), b20_i))
                    self._final_bin_list_data[1][self._iter] = res._exported_filename
                
                self.include_header_in_results_file=True
                self.exportResults()
    
    def _auxWindows_executeProgram(self, output_fn):
        """ 
        Dummy method to test the scripts1d in Windows
        """
        # program = """ """
        # exec(program)
        FLD_TEST_ = 'data_resources/testing_files/'
        file2copy = FLD_TEST_+'TEMP_res_z12n12_0-dbase_max_iter.txt'
        file2copy = FLD_TEST_+'TEMP_res_z12n12_0-dbase_broken.txt'
        # file2copy = FLD_TEST_+'TEMP_res_z12n12_0-dbase.txt'
        file2copy = FLD_TEST_+'TEMP_res_z2n1_0-dbase3odd.txt'
        
        txt = ''
        with open(file2copy, 'r') as f:
            txt = f.read()
            txt = txt.format(INPUT_2_FORMAT=self.inputObj)
        
        with open(output_fn, 'w+') as f:
            f.write(txt)
        
        hash_ = hash(random()) # random text to identify wf
        ## wf intermediate
        with open('final_wf.bin', 'w+') as f:
            f.write(str(hash_))
        if self.inputObj.interm_wf == 1:
            with open('intermediate_wf.bin', 'w+') as f:
                f.write(str(hash_))
        
        if self.TRACK_TIME_AND_RAM:
            with open(FLD_TEST_+'TEMP_time_verbose_output.txt', 'r') as f:
                data = f.read()
                with open(self.TRACK_TIME_FILE, 'w+') as f2:
                    f2.write(data)
        
        MAX_TO = max(self._getMinimumIterationTimeTaurus(), 43200) # 12 h timeout
        ## simulates the other .dat prompt
        for file_ in  ("canonicalbasis", "eigenbasis_h", 
                       "eigenbasis_H11", "occupation_numbers"):
            txt = ""
            with open(f"{FLD_TEST_}TEMP_{file_}.txt", 'r') as f:
                txt = f.read()
            with open(f"{file_}.dat", 'w+') as f:
                f.write(txt) 
        
    def _namingFilesToSaveInTheBUfolder(self):
        """ 
        Naming and  tracking of the unconverged and converged results 
        """
        tail = f"z{self.z}n{self.n}_d{self._curr_deform_index}"
        s_list = [x for x in os.listdir(self.DTYPE.BU_folder)]
        s_list = list(filter(lambda x: tail in x, s_list))
        s_list = list(filter(lambda x: x.endswith(".OUT"), s_list))
        s_n = len(s_list)
        
        unconv_n = " ".join(s_list).count("unconv")
        s_n -= unconv_n
        if getattr(self._current_result, 'properly_finished', False)==False:
            s_n = str(s_n)+'unconv'+str(unconv_n)
            ## Note, to keep the first(unconv) and intermediate results
            ## read the "base" results in order until non (unconv) label
        else:
            # if unconv_n > 0:
            #     s_n = str(s_n) ## there are no results converged of any type
            #     ## [res_0unconv0, res_0unconv1] -> append(res_0)
            #    NOTA: esto no vale porque si no se producen convergencias, confundiras
            #    estados de ida (no convergidos) con un resultado valido pero que ocurrio a la vuelta
            # else:
            s_n = str(s_n + unconv_n)
        tail = f"z{self.z}n{self.n}_d{self._curr_deform_index}_{s_n}"
        return tail
    
    def saveFinalWFprocedure(self, result, base_execution=False):
        """ 
        Method to save wf and other objects from _executeProgram outputs
        Args
            :result: dataTaurus result of the calculation
            :base_execution: is to know if the execution were the starter minimum
            localization or if it is in the run process
        """
        if result.broken_execution: return
        
        if base_execution:
            self._base_wf_filename = 'base_initial_wf.bin'
            shutil.copy('final_wf.bin', self._base_wf_filename)
            tail = f"z{self.z}n{self.n}_dbase"
            
            ## save all the 1st blocked seeds
            ## Extend the condition for general repetitive base-input.
            if 1 in self.numberParity or self.GENERATE_RANDOM_SEEDS: 
                s_list = [x for x in os.listdir(self.DTYPE.BU_folder)]
                s_list = list(filter(lambda x: x.endswith("dbase.OUT"), s_list))
                s_n = len(s_list)
                unconv_n = " ".join(s_list).count("unconv")
                s_n -= unconv_n
                if getattr(result, 'properly_finished', False)==False:
                    s_n = str(s_n)+'unconv'+str(self._preconvergence_steps)
                    ## Note, to keep the first(unconv) and intermediate results
                    ## read the "base" results in order until non (unconv) label
                tail = tail.replace("dbase", f"{s_n}-dbase")
        else:
            tail = self._namingFilesToSaveInTheBUfolder()
        
        ## copy the wf to the initial wf always except non 
        if (not self.force_converg) or result.properly_finished:
            shutil.copy('final_wf.bin', 'initial_wf.bin')
        
        shutil.move(self._output_filename, 
                    f"{self.DTYPE.BU_folder}/res_{tail}.OUT")
        shutil.copy('final_wf.bin', 
                    f"{self.DTYPE.BU_folder}/seed_{tail}.bin")
        result._exported_filename = tail
        
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
    
    def _getMinimumIterationTimeTaurus(self):
        """
        Get the minimum estimation time with the DD term on according to the
        cpu time regression formula
        
            t_per_iter / sp_dim^3 = 1.5e-9 * ROmega_dim + 7e-6  [seconds]
        
        setting the time for a maximum of 600 steps.
        """
        if isinstance(self.inputObj, InputAxial):
            return 600
        
        Om  = self.inputObj._DD_PARAMS['omega_dim']
        ROm_dim = LEBEDEV_GRID_POINTS[Om - 1] * self.inputObj._DD_PARAMS['r_dim']
        
        time_  = (1.5e-9 * ROm_dim) + 7e-6 
        time_ *= self._sp_dim ** 3
        return max(self.inputObj.iterations, 600) * time_
    
    def _executeProgram(self, base_execution=False):
        """
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
                order_ = f'./{self.inputObj.PROGRAM} < {_inp_fn} > {_out_fn}'
                if self.TRACK_TIME_AND_RAM: 
                    ## option to analyze the program performance
                    order_ = f'{{ /usr/bin/time -v {order_}; }} 2> {self.TIME_FILE}'
                
                MAX_TO = max(self._getMinimumIterationTimeTaurus(), 
                             self.MAX_ITER_WAITING_TIME)
                _e = subprocess.call(order_, 
                                     shell=True,
                                     timeout=MAX_TO)
            
            res = self.DTYPE(self.z, self.n, _out_fn)
            if self.TRACK_TIME_AND_RAM: 
                res = self._addExecutionPerformanceData(res)
            
            self.saveFinalWFprocedure(res, base_execution)
            self.executionTearDown   (res, base_execution)
            self._current_result = deepcopy(res)
            
            if self.RUN_PROJECTION and not base_execution: self.runProjection()
            
        except Exception as e:
            printf(f"  [FAIL]: {self.__class__}._executeProgram()")
            if isinstance(res, DataTaurus):
                printf(f"  [FAIL]: result=", str(res))
            # TODO:  manage exceptions from execution
            raise e
        
        return res
    
    def _addExecutionPerformanceData(self, result: DataTaurus):
        """
        Reads the information from the time command output and append to it.
        """
        args = parseTimeVerboseCommandOutputFile(self.TRACK_TIME_FILE)
        iters_ = max(1, getattr(result, 'iter_max', 0))
        result.iter_time_cpu = args['user_time']
        result.time_per_iter_cpu = args['user_time']  / iters_
        result.memory_max_KB = args['memory_max']
        return result
        
    def printExecutionResult(self, result : DataTaurus, print_head=False, 
                                  *params2print):
        """
        Standard step information
        """
        HEAD = "  z  n  (st) ( d)       E_HFB        Kin     Pair       b2"
        if print_head:
            printf('\n'+HEAD+LINE_2)
            return
        if result == None or result.broken_execution:
            printf(" {:2} {:2}    {}  {:>4}"
                    .format(self.z, self.n, '(F)', str(self._curr_deform_index)))
            return
        
        ## TODO: check if broken to skip the iter_time get
        status_fin = 'X' if not result.properly_finished  else '.'
        _iter_str = "[{}/{}: {}']".format(result.iter_max, self.inputObj.iterations, 
                                          getattr(result, 'iter_time_seconds', 0) //60 )
        
        txt  =" {:2} {:2}    {}  {:>4}    {:9.3f}  {:8.3f}  {:7.3f}   {:+6.3f} "
        txt = txt.format(result.z, result.n, status_fin, 
                         str(self._curr_deform_index),
                         result.E_HFB, result.kin, result.pair, 
                         result.b20_isoscalar)
        printf(txt, _iter_str)
    
    @property
    def calculationParameters(self):
        """
            Print properties of the calculation to know while running, 
            such as the input object, folders, set up properties, attributes ...
        """
        if not self.PRINT_CALCULATION_PARAMETERS:
            return
        printf(LINE_2)
        printf(f" ** Executor 1D [{self.__class__.__name__}] Parameters:")
        printf(LINE_1)
        priv_attr = ('_1stSeedMinima', '_DDparams', '_deform_base', 
                     '_N_steps', '_iter', '_output_filename')
        priv_attr = dict([(k,getattr(self, k, None)) for k in priv_attr])
        priv_attr = {'PRIVATE_ATTR:': priv_attr}
        pub_attr = dict(list(filter(
            lambda x: not (x[0].startswith('_') or isinstance(x[1], type)),
            self.__dict__.items() )))
        prettyPrintDictionary(pub_attr)
        prettyPrintDictionary(priv_attr)
        printf(LINE_1)
    
    def executionTearDown(self, result : DataTaurus, base_execution, *args, **kwargs):
        """
        Proceedings to do after the execution of a single step.
            copying the wf and output to a folder, clean auxiliary files,
        """
        if self.PRINT_STEP_RESULT:
            self.printExecutionResult(result)
        
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
        dtype_ = None
        for part_ in (0, 1):
            for indx, res in enumerate(self._results[part_]):
                ## NOTE: This conditional is unnecessary, leaved here for secure iter.
                if indx >= self._deformations_map[part_].__len__(): break
                
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
                
                if (dtype_ == None) and (res != None):
                    dtype_ = res.__class__.__name__
        
        txt_ = '\n'.join(res2print)
        head_ = ", ".join([dtype_, self.CONSTRAINT_DT])
        txt_ = head_ + '\n' + txt_
        
        if output_filename == None:
            output_filename = self.EXPORT_LIST_RESULTS
        
        with open(output_filename, 'w+') as f:
            f.write(txt_)
        
    
    def globalTearDown(self, *args, **kwargs):
        """
        Proceedings for the execution to do. i.e:
            zipping files, launch tests on the results, plotting things
        """
        raise ExecutionException("Abstract Method, implement me!")
    

class _Base1DAxialExecutor(_Base1DTaurusExecutor):
    
    EXPORT_LIST_RESULTS = 'export_resultAxial.txt'
    
    DTYPE = DataAxial  # DataType for the outputs to manage
    ITYPE = InputAxial # Input type for the input management
    
        
    def __init__(self, z, n, ipar_int, MZmax, *args, **kwargs):
        """
        :ipar_int: [0:9] choice of program parameters for a Gogny interaction
        """
        _Base1DTaurusExecutor.__init__(self, z, n, ipar_int, *args, **kwargs)
        
        del self._DDparams # not necessary
        self.interaction : int = int(self.interaction)
        self.MzMax = MZmax
    
    
    def _run_backwardsSweeping(self, oblate_part=None):
        raise ExecutionException("Method not implemented for Axial calculation")
    def _runUntilConvergence(self, MAX_STEPS=3):
        raise ExecutionException("Method not implemented for Axial calculation")
    def _runVariableStep(self):
        raise ExecutionException("Method not implemented for Axial calculation")
        
    def _auxWindows_executeProgram(self, output_fn):
        """ 
        Dummy method to test the scripts1d in Windows
        """
        # program = """ """
        # exec(program)
        file2copy = "DATA_RESULTS/axial_output_maxiter.OUT"
        # file2copy = 'TEMP_output_Z10N6_broken.txt'
        file2copy = "DATA_RESULTS/axial_output.OUT"
        
        txt = ''
        with open(file2copy, 'r') as f:
            txt = f.read()
            txt = txt.format(INPUT_2_FORMAT=self.inputObj)
        
        with open(output_fn, 'w+') as f:
            f.write(txt)
        
        hash_ = hash(random()) # random text to identify wf
        ## wf intermediate
        with open('fort.11', 'w+') as f:
            f.write(str(hash_))
            
    def printExecutionResult(self, result : DataTaurus, print_head=False, 
                             *params2print):
        """
        Standard step information
        """
        HEAD = "  z  n  (st) ( d)       E_HFB        Kin     Pair       b2"
        if print_head:
            printf('\n'+HEAD+LINE_2)
            return
        
        status_fin = 'X' if not result.properly_finished  else '.'
        _iter_str = "[{}/{}: {}']".format(result.iter_max, self.inputObj.iterations, 
                                          getattr(result, 'iter_time_seconds', 0) //60 )
        
        txt  =" {:2} {:2}    {}  {:>4}    {:9.3f}  {:8.3f}  {:7.3f}   {:+6.3f} "
        txt = txt.format(result.z, result.n, status_fin, 
                         str(self._curr_deform_index),
                         result.E_HFB, result.kin, result.pair, 
                         result.b20_isoscalar)
        printf(txt, _iter_str)
        
    def saveFinalWFprocedure(self, result,  base_execution=False):
        """ 
        Method to save wf and other objects from _executeProgram outputs
        Args
            :result: dataTaurus result of the calculation
            :base_execution: is to know if the execution were the starter minimum
            localization or if it is in the run process
        """
        if result.broken_execution: return
        
        if base_execution:
            self._base_wf_filename = 'base_initial_wf.bin'
            shutil.copy('fort.11', self._base_wf_filename)
            tail = f"z{self.z}n{self.n}_dbase"
            
            ## save all the 1st blocked seeds
            ## Extend the condition for general repetitive base-input.
            if 1 in self.numberParity or self.GENERATE_RANDOM_SEEDS: 
                s_list = [x for x in os.listdir(self.DTYPE.BU_folder)]
                s_list = list(filter(lambda x: x.endswith("dbase.OUT"), s_list))
                s_n = len(s_list)
                unconv_n = " ".join(s_list).count("unconv")
                s_n -= unconv_n
                if getattr(result, 'properly_finished', False)==False:
                    s_n = str(s_n)+'unconv'+str(self._preconvergence_steps)
                    ## Note, to keep the first(unconv) and intermediate results
                    ## read the "base" results in order until non (unconv) label
                tail = tail.replace("dbase", f"{s_n}-dbase")
        else:
            tail = self._namingFilesToSaveInTheBUfolder()
        
        ## copy the wf to the initial wf always except non 
        if (not self.force_converg) or result.properly_finished:
            shutil.copy('fort.11', 'fort.10')
        
        shutil.move(self._output_filename, 
                    f"{self.DTYPE.BU_folder}/res_{tail}.OUT")
        shutil.copy('fort.11', 
                    f"{self.DTYPE.BU_folder}/seed_{tail}.bin")
        
        
from multiprocessing import cpu_count
from typing import List

class _BaseParallelTaurusExecutor(_Base1DTaurusExecutor):
    
    DTYPE = DataTaurusPAV  # DataType for the outputs to manage
    ITYPE = InputTaurusPAV # Input type for the input management
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.SINGLE_EVALUATION
    
    CONSTRAINT    = None
    CONSTRAINT_DT = None
    
    EXPORT_LIST_RESULTS = 'export_list'
    
    MAX_PARALLEL_NODES = 2
    
    def __init__(self, z, n, interaction, *args, **kwargs):
        
        ## interactions and nucleus
        self.z : int = z
        self.n : int = n
        self.interaction : str = interaction
        
        self.inputObj :         List[self.ITYPE]  = []
        self._current_results : List[self.DTYPE]  = []  # list of self.DTYPE
        
        self.axialSymetryRequired = False ## set up to reject non-axial results
        self.sphericalSymmetryRequired = False
        self._output_filename = self.DTYPE.DEFAULT_OUTPUT_FILENAME
        
        self._sequence_jobs : List = []
        self._deformations_map : List = []
        self._curr_deform_index : List[int]  = None
        
        self._iter    = 0
        self._N_steps = 0
        self._preconvergence_steps = 0
        self._base_wf_filename = None
        self._base_seed_type = None
                
        self.force_converg  = False # requirement of convergence for wf copy
        
        if kwargs:
            for key_, value in kwargs.items():
                setattr(self, key_, value)
        
        self._checkExecutorSettings()
    
    @classmethod
    def setMaximumNumberOfNodes(cls, nodes):
        """
        Setting number of processes
        """
        if nodes >= cpu_count():
            printf("[Warning] The maximum number of nodes for the computer is",
                  cpu_count(), ", setting to that -1.")
            nodes = cpu_count() - 1
        cls.MAX_PARALLEL_NODES = nodes
    
    def setUp(self, **params):
        """
        Set up of any global parameters/variables/wd before the first execution
        (first execution set up for the seed wf, not the parallelizable run)
        """
        pass
    
    def setUpExecution(self, reset_seed=False, *args, **kwargs):
        """
        First Execution, seed production and set up of all variables from 
        """
        pass
    
    def run(self):        
        pass
    
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
        
        self._results = [[], []]
        
        if not keep_1stMinimum:
            
            self._1stSeedMinima : self.DTYPE  = None
            self._base_wf_filename = None
            
            self._deform_lim_max = None
            self._deform_lim_min = None
            self._deform_base    = None
    
    def _acceptanceCriteria(self, result):
        pass
    
    def _runUntilConvergence(self, MAX_STEPS=3):
        pass
    
    def _auxWindows_executeProgram(self, output_fn):
        pass
    
    def _executeProgram(self, base_execution=False):
        pass
    
    def printExecutionResult(self, result:DataTaurus, print_head=False, 
                             *params2print):
        pass
    
    def executionTearDown(self, result:DataTaurus, base_execution, *args, **kwargs):
        pass
    
    def saveFinalWFprocedure(self, result, base_execution=False):
        pass

exe = _BaseParallelTaurusExecutor(2, 1, "B1_MZ1")
exe.setMaximumNumberOfNodes(3)


