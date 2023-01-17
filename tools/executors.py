'''
Created on Jan 10, 2023

@author: Miguel

Module for script setting

'''
import os
import subprocess
import numpy as np
from copy import deepcopy, copy
from tools.inputs import Enum, InputTaurus
from tools.data import DataTaurus
from tools.helpers import LINE_2, LINE_1, prettyPrintDictionary

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
    
    CONSTRAINT : str = None # DataTaurus Variable to compute
    
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
        
        self.activeDDterm = True
        self._output_filename = DataTaurus.DEFAULT_OUTPUT_FILENAME
        
        self.deform_oblate   : list = []
        self.deform_prolate  : list = []
        self._deformations_map : list = [{}, {}] #oblate, prolate
        self._deform_lim_max = None
        self._deform_lim_min = None
        self._deform_base    = None
        self._iter    = 0
        self._N_steps = 0
        self._base_wf_filename = None
        
        self.save_final_wf = False # To set for the final wf to be copied as initial
        self.force_converg = False # 
        
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
        if ((self.CONSTRAINT == None) and 
            (self.ITERATIVE_METHOD != self.IterativeEnum.SINGLE_EVALUATION)):
            raise ExecutionException("All Iterative procedures must have a defined"
                                     " constraint, None set.")
        
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
        
        self._deform_lim_max = max_
        self._deform_lim_min = min_
        self._N_steps = N_steps
        if self.ITERATIVE_METHOD == self.IterativeEnum.VARIABLE_STEP:
            ## The results and deformations are unfixed, just between the boundaries
            return
        
        array_ = list(np.linspace(min_, max_, num=N_steps, endpoint=True))
        self.deform_oblate  = []
        self.deform_prolate = array_
        
        self._deformations_map[0] = {}
        self._deformations_map[1] = dict(list(enumerate(array_)))
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
        q0 = getattr(self._1stSeedMinima, self.CONSTRAINT, None)
        if   q0 < p_min: # p0 outside to the left (add the difference to the left)
            p_min = q0 + (q0 - p_min)
        elif q0 > p_max: # p0 outside to the right (add the diff. to the right)
            p_max = q0 + (q0 - p_max)
        
        if  q0 == None: 
            print("[WARNING] _set_pair could not get the main Constraint [",
                  self.CONSTRAINT, "]. Default setting it to 0.00")
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
        
        self._deformations_map[0] = dict(list(enumerate(deform_oblate)))
        self._deformations_map[1] = dict(list(enumerate(deform_prolate)))
        
    
    def run(self):
        self._checkExecutorSettings()
        #TODO: Select running method by ITERATIVE_METHOD ??
        
        if self.ITERATIVE_METHOD == self.IterativeEnum.VARIABLE_STEP:
            raise Exception("Option not implemented")
        else:
            raise Exception("Option not implemented")
        
    
    def _auxWindows_executeTaurus(self):
        """ 
        Dummy method to test the scripts in Windows
        """
        # program = """
        # """
        # "aux_output_Z10N10_00_00"
        # exec(program)
        
        _=0
    
    def _executeTaurus(self):
        """
        TODO: Main execution method, prints.
        Input object and its dd-parameters must be already setted at this point
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
                self._auxWindows_executeTaurus()
            else:
                _e = subprocess.call(f'./taurus_vap.exe < {_inp_fn} > {_out_fn}', 
                                     shell=True,
                                     timeout=43200) # 12 h timeout
            
            res = DataTaurus(self.z, self.n, _out_fn)
            
            self.saveFinalWFprocedure(res) ## TODO: here??
            self.executionTearDown(res)
            
        except Exception as e:
            raise e
            raise Exception("TODO: manage exceptions in taurus execution")
            
    
    def printTaurusResult(self, result : DataTaurus, **params2print):
        """
        Standar step information
        """
        status_fin = 'X' if not result.properly_finished  else '.'
        _ = 0
        
        
    
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
    
    def executionTearDown(self, result : DataTaurus, *args, **kwargs):
        """
        Proceedings to do after the execution of a single step.
            copying the wf and output to a folder, clean auxiliary files,
        """
        pass
    
    def gobalTearDown(self, *args, **kwargs):
        """
        Proceedings for the execution to do. i.e:
            zipping files, launch tests on the results, plotting things
        """
        raise ExecutionException("Abstract Method, implement me!")
    



class ExeTaurus1D_DeformQ20(_Base1DTaurusExecutor):
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.EVEN_STEP_SWEEPING
    
    CONSTRAINT = InputTaurus.ConstrEnum.b20
    
    def setUp(self):
        """
        set up: 
            * back up folder for results
            * dumping filename
        """
        self._DDparams = self.inputObj._DD_PARAMS
        
    
    def setUpExecution(self, *args, **kwargs):
        """
        base solution preconvergence
        """
        
        self.calculationParameters
        
        self._executeTaurus()
        
        self._setDeformationFromMinimum(self._deform_lim_min, 
                                        self._deform_lim_max, self._N_steps)
        
        pass
    
    
    def run(self):
        ## TODO: might require aditional changes
        _Base1DTaurusExecutor.run(self)
