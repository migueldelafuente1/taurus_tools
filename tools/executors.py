'''
Created on Jan 10, 2023

@author: Miguel

Module for script setting

'''
import os
import subprocess
import shutil
import numpy as np
from copy import deepcopy, copy
from tools.inputs import Enum, InputTaurus
from tools.data import DataTaurus
from tools.helpers import LINE_2, LINE_1, prettyPrintDictionary, zipBUresults


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
    
    CONSTRAINT    : str = None # InputTaurus Variable to compute
    CONSTRAINT_DT : str = None # DataTaurus (key) Variable to compute
    
    def setUp(self, *args, **kwargs):
        raise ExecutionException("Abstract method, implement parameters for the execution")
    
    def __init__(self, z, n, interaction, *args, **kwargs):
        
        ## interactions and nucleus
        self.z : int = z
        self.n : int = n
        self.interaction : str = interaction
        
        self.inputObj : InputTaurus  = None
        self._DDparams : dict = None
        self._inputCalculationArgs : dict = None
        self._1stSeedMinima : DataTaurus  = None
        self._current_result: DataTaurus  = None ## TODO: might be useless, remove in that case
        
        self.activeDDterm = True
        self._output_filename = DataTaurus.DEFAULT_OUTPUT_FILENAME
        
        self.deform_oblate   : list = []
        self.deform_prolate  : list = []
        self._deformations_map  : list = [[], []] #oblate, prolate
        self._curr_deform_index : int  = None
        self._deform_lim_max = None
        self._deform_lim_min = None
        self._deform_base    = None
        self._iter    = 0
        self._N_steps = 0
        self._base_wf_filename = None
        
        self.save_final_wf  = False # To set for the final wf to be copied as initial
        self.force_converg  = False # 
        
        self._results : list = [[], []] # oblate, prolate
        if kwargs:
            for key_, value in kwargs.items():
                setattr(self, key_, value)
        
        self._checkExecutorSettings()
    
    def _checkExecutorSettings(self):
        """
        Tests for the method to verify the calculation.
        """
        if ((self.CONSTRAINT != None) and 
            (self.CONSTRAINT not in InputTaurus.ConstrEnum.members()) ):
            raise ExecutionException("Main constraint for the calculation is invalid"
                f", given [{self.CONSTRAINT}] but must be in InputTaurus.ConstrEnum" )
        
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
        
        self.inputObj = InputTaurus(self.z, self.n, self.interaction)
        
        if core_calc:
            self.inputObj.setUpValenceSpaceCalculation()
        elif axial_calc:
            self.inputObj.setUpNoCoreAxialCalculation()
        
        _check = [(hasattr(InputTaurus.ArgsEnum,k) or 
                   hasattr(InputTaurus.ConstrEnum,k)) for k in input_kwargs.keys()]
        if not all(_check):
            raise ExecutionException("One of the arguments is invalid for "
                f"InputTaurus:\n {input_kwargs}")
        
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
        """
        if self.ITERATIVE_METHOD == self.IterativeEnum.SINGLE_EVALUATION:
            print("Single evaluation process do not require this process, continue")
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
        
        self._deform_base   = q0
        self.deform_oblate  = deform_oblate
        self.deform_prolate = deform_prolate
        
        self._deformations_map[0] = []
        for k, val in enumerate(deform_oblate):
            self._deformations_map[0].append( (-k - 1, val) )
        self._deformations_map[1] = list(enumerate(deform_prolate))
        self._results[1].append(self._1stSeedMinima)
        
    
    def run(self):
        self._checkExecutorSettings()
        
        if self.ITERATIVE_METHOD == self.IterativeEnum.VARIABLE_STEP:
            raise Exception("Option not implemented")
        else:
            print(" ** oblate:")
            for k, val in self._deformations_map[0]: # oblate
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                self._results[0].append(self._executeTaurus())
            if self.ITERATIVE_METHOD == self.IterativeEnum.EVEN_STEP_SWEEPING:
                self._run_backPropagationSweeping(oblate_part=True) 
            
            print(" ** prolate:")
            for k, val in self._deformations_map[1][1:]: # prolate
                ## exclude the first state since it is the original seed
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                self._results[1].append(self._executeTaurus())
            
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
                res  : DataTaurus = self._executeTaurus() # reexecuting
                
                indx_ = -k-1
                res0 : DataTaurus = self._results[0][indx_]
                if self._backPropagationAcceptanceCriteria(res, res0):
                    self._results[0][indx_] =  res
        else:
            print(" ** prolate (back):")
            for k, val in reversed(self._deformations_map[1][1:]): # prolate
                ## exclude the first state since it is the original seed
                self._curr_deform_index = k
                self.inputObj.setConstraints(**{self.CONSTRAINT: val})
                res  : DataTaurus = self._executeTaurus()
                # indx_ = k - 1
                res0 : DataTaurus = self._results[1][k]
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
        if not isinstance(result, DataTaurus): 
            return False    # cannot accept a new broken result
        if isinstance(prev_result, DataTaurus): 
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
                
    
    def _auxWindows_executeTaurus(self, output_fn):
        """ 
        Dummy method to test the scripts in Windows
        """
        # program = """ """
        # exec(program)
        file2copy = "aux_output_Z10N10_00_00"
        file2copy = "aux_output_Z10N6_23"
        # file2copy = 'aux_output_Z10N6_broken'
        
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
        Method to save wf and other objects from _executeTaurus outputs
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
                    f"{DataTaurus.BU_folder}/res_{tail}.OUT")
        shutil.copy('final_wf.bin', 
                    f"{DataTaurus.BU_folder}/seed_{tail}.bin")
        
        if self.SAVE_DAT_FILES:
            dat_files = filter(lambda x: x.endswith('.dat'), os.listdir())
            dat_files = filter(lambda x: x[:-4] in self.SAVE_DAT_FILES, dat_files)
            for _df in dat_files:
                _df2 = _df.replace('.dat', f'_{tail}.dat')
                shutil.copy(_df, f"{DataTaurus.BU_folder}/{_df2}")
                if not os.getcwd().startswith('C:'):
                    shutil.move(_df, f"{DataTaurus.BU_folder}/{_df2}")
        
        if os.getcwd().startswith('C:'): # Testing on windows
            f_delete = filter(lambda x: x.endswith('.dat') and not 'original' in x, 
                              os.listdir() )
            for f in f_delete:
                os.remove(f)
        else:
            _e = subprocess.call('rm *.dat *.gut', shell=True)
    
    
    def _executeTaurus(self, base_execution=False):
        """
        TODO: Main execution method, prints.
        Input object and its dd-parameters must be already setted at this point
            :base_execution: indicates the minimization to be for the 1 seed
            (configure options to save the wave function)
        """
        assert self.inputObj != None, ExecutionException(
            "Trying to launch an execution without defined InputTaurus object.")
        
        res = None
        
        if self.activeDDterm:
            with open(self.inputObj.INPUT_DD_FILENAME, 'w+') as f:
                f.write(self.inputObj.get_inputDDparamsFile())
        
        try:
            with open(self.inputObj.input_filename, 'w+') as f2:
                f2.write(self.inputObj.getText4file())
            
            _inp_fn = self.inputObj.input_filename
            _out_fn = self._output_filename
            
            if os.getcwd().startswith('C:'): ## Testing purpose 
                self._auxWindows_executeTaurus(_out_fn)
            else:
                _e = subprocess.call(f'./taurus_vap.exe < {_inp_fn} > {_out_fn}', 
                                     shell=True,
                                     timeout=43200) # 12 h timeout
            
            res = DataTaurus(self.z, self.n, _out_fn)
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
            self.exportResults(include_index_header=True)
        
        
    def exportResults(self, include_index_header=False, output_filename=None):
        """
        writes a text file with the results dict-like (before the program ends)
        Order: oblate[-N, -N-1, ..., -1] > prolate [0, 1, ..., N']
        """
        res2print = []
        for part_ in (0, 1):
            for indx, res in enumerate(self._results[part_]):
                key_, dval = self._deformations_map[part_][indx]
                line = []
                if include_index_header:
                    line.append(f"{key_:5}: {dval:+6.3f}")
                line.append(res.getAttributesDictLike)
                line = ' ## '.join(line)
                
                if part_ == 1: ## prolate (add to the final)
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
        """
        
        self._DDparams = self.inputObj._DD_PARAMS
        DataTaurus.BU_folder = f'BU_folder_z{self.z}n{self.n}'
        DataTaurus.setUpFolderBackUp()
        self.SAVE_DAT_FILES = [DataTaurus.DatFileExportEnum.canonicalbasis,]
        
    
    def setUpExecution(self, *args, **kwargs):
        """
        base solution pre-convergence.
            to change after the execution, put by InputTaurus.*Enum new values
        """
        
        self.calculationParameters
        
        ## TODO: see convergece or force convergence to repeat the result or 
        ## save the aproxximation, set deform_bas
        res = None
        self._preconvergence_steps = 0
        self.printTaurusResult(None, print_head=True)
        while not self._preconvergenceAccepted(res):
            res = self._executeTaurus(base_execution=True)
        
        self._setDeformationFromMinimum(self._deform_lim_min, 
                                        self._deform_lim_max, self._N_steps)
        
        
        _new_input_args = dict(filter(lambda x: x[0] in InputTaurus.ArgsEnum.members(), 
                                      kwargs.items() ))
        _new_input_cons = dict(filter(lambda x: x[0] in InputTaurus.ConstrEnum.members(), 
                                      kwargs.items() ))
        self.inputObj.setParameters(**_new_input_args, **_new_input_cons)
        
    
    def _preconvergenceAccepted(self, result: DataTaurus):
        """
        TODO:
        define the steps to accept the result
        """
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
    
    def gobalTearDown(self, *args, **kwargs):
        """
        Proceedings for the execution to do. i.e:
            zipping files, launch tests on the results, plotting things
        """
        zipBUresults(DataTaurus.BU_folder, self.z, self.n, self.interaction)
    


class ExeTaurus1D_DeformB20(ExeTaurus1D_DeformQ20):
    
    CONSTRAINT    = InputTaurus.ConstrEnum.b20
    CONSTRAINT_DT = DataTaurus.getDataVariable(InputTaurus.ConstrEnum.b20,
                                               beta_schm = 1)
    EXPORT_LIST_RESULTS = 'export_TESq20'
    

#===============================================================================
#
#    EXECUTOR DEFINITIONS: ANGULAR MOMENTUM DEFORMATIONS 
#
#===============================================================================

