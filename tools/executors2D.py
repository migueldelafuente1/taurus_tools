'''
Created on 14 ago 2023

@author: delafuente
'''
import os, shutil
import numpy as np
import inspect

from .executors import _Base1DTaurusExecutor
from tools.base_executors import ExecutionException
from tools.inputs import InputTaurus
from tools.data import   DataTaurus
from tools.helpers import printf, zipBUresults, getValueCombinationsSorted
from tools.executors import ExeTaurus1D_DeformB20
from tools.Enums import OutputFileTypes

class _Base2DTaurusExecutor(_Base1DTaurusExecutor):
    
    _CONSTRAINT_INP_DAT = (InputTaurus, DataTaurus)
    
    CONSTRAINT    : list = []  # InputTaurus Variable to compute
    CONSTRAINT_DT : list = []  # DataTaurus (key) Variable to compute
    
    
    def __init__(self, z, n, interaction, *args, **kwargs):
        
        _Base1DTaurusExecutor.__init__(self, z, n, interaction, *args, **kwargs)
        
        ## interactions and nucleus
        self._temp_seed_minima      : list = []  # store partial minimaself.DTYPE  = None #  
        self._current_result: self.DTYPE  = None ## TODO: might be useless, remove in that case
        
        self.deform_oblate   : list = []
        self.deform_prolate  : list = []
        self._deformations_map  : list = [] # listed with ([], []) #oblate, prolate
        self._curr_deform_index : list = [] # int-index
        self._deform_lim_max = []
        self._deform_lim_min = []
        self._deform_base    = []
        
        ## Iterations related to the repetition of unconverged solutions
        self._iter    = [] # iter for variable_step method
        self._N_steps = []
        self._preconvergence_steps = 0 # just the last point calculation
        
        self._final_bin_list_data : dict = {} # List of the names for the saved files to export [Cstr][obl/prol:{}]
        self._results : dict = {} # no oblate, prolate criteria, just from index signs
        if kwargs:
            for key_, value in kwargs.items():
                setattr(self, key_, value)
        
        self._setInstanceExecutorConstraints()
        self._checkExecutorSettings()
        
        self._all_def_keys_sorted   = []
        self._all_def_values_by_key = []
    
    @classmethod
    def setExecutorConstraints(cls, constr_list):
        """
            NOTE: no more a @classmethod()
        Launch before the program to set the constraint dimension
        (unset to prompt an exception to avoid default constraint set up)
        """
        if len(constr_list) == 0: raise ExecutionException("Constraint list argument is empty")
        printf("  [executor2D] Clearing previous constraints.")
        cls.CONSTRAINT    = []
        cls.CONSTRAINT_DT = []
        
        constr_list_2 = []
        for constr in constr_list:
            assert constr in cls._CONSTRAINT_INP_DAT[0].ConstrEnum.members(), \
                f"Constraint must come from type {cls._CONSTRAINT_INP_DAT[0]}"
            
            cls.CONSTRAINT   .append(constr)
            cls.CONSTRAINT_DT.append(constr)
            constr_list_2    .append(constr.replace('_', ''))
        n = len(constr_list)
        constr_str = '-'.join(constr_list_2)
        
        cls.EXPORT_LIST_RESULTS = f'export_TES{n}_{constr_str}'
        DataTaurus.BU_folder    = f'export_TES{n}_{constr_str}'
        _=0
    
    def _setInstanceExecutorConstraints(self):
        """ required at instancing. """
        for constr in self.CONSTRAINT:
            self._temp_seed_minima.append(None)
            
            self.deform_oblate .append([])
            self.deform_prolate.append([])
            self._deformations_map.append( [[], [] ] ) #oblate, prolate
            self._curr_deform_index.append(None) # int-index
            self._deform_lim_max   .append(None)
            self._deform_lim_min   .append(None)
            self._deform_base      .append(None)
            
            self._iter   .append(0)
            self._N_steps.append(0)    
    
    def resetExecutorObject(self, keep_1stMinimum=False):
        """
        Clean deformations and results from a previous calculation,
        only instanced attributes generated during the run (inputObject, params, ... keept)
        could keep the first minimum data
        """
        for i in range(len(self.CONSTRAINT)):
            self._current_result[i] = None  ## TODO: might be useless, remove in that case
            
            self.deform_oblate     [i] = []
            self.deform_prolate    [i] = []
            self._deformations_map [i] = [[], []] #oblate, prolate
            self._curr_deform_index[i] = None
            
            self._results = {}
            # self._final_bin_list_data = {}
            
            self._iter    [i] = 0
            self._N_steps [i] = 0
            self._preconvergence_steps = 0
            
            if not keep_1stMinimum:
                self._temp_seed_minima[i] = None
                
                self._deform_lim_max[i] = None
                self._deform_lim_min[i] = None
                self._deform_base   [i] = None     
        
        if not keep_1stMinimum:
            self._1stSeedMinimum : self.DTYPE  = None
            self._base_wf_filename = None  
    
    def setUp(self, *args, **kwargs):
        raise ExecutionException("Abstract method, implement me!")
    
    def _checkExecutorSettings(self):
        """
        Tests for the method to verify the calculation.
        """
        if (self.CONSTRAINT != []): 
            for ctr_ in self.CONSTRAINT:
                if not (ctr_ in self.ITYPE.ConstrEnum.members()):
                    raise ExecutionException(
                        f"Constraint [{ctr_}] for the calculation is invalid",
                        f", given [{self.CONSTRAINT}] but must be in Input***.ConstrEnum" )
        
        if (self.ITERATIVE_METHOD != self.IterativeEnum.SINGLE_EVALUATION):
            for i in range(len(self.CONSTRAINT)):
                if (self.CONSTRAINT[i] == None): 
                    raise ExecutionException("All Iterative procedures must have a defined"
                                             " constraint, None set.")
                if (self.CONSTRAINT_DT[i] == None):
                    raise ExecutionException("Must be a defined observable for the result to",
                                             " evaluate the deformations. None set.")
        if self._base_seed_type == None:
            printf("[WARNING Executor] Seed was not defined, it will be copied (s=1).")
        
    def setUpExecution(self, *args, **kwargs):
        """ Only requires the appending of _binlistdata for each cstr-deformation """
        raise ExecutionException("Abstract method, implement me!")
    
    def defineDeformationRange(self, constr_with_limits):
        """
        Set the arrays for the deformations (prolate, oblate)
        deformation boundaries are required, only SINGLE_EVALUATION ignores them
        """
        if self.ITERATIVE_METHOD == self.IterativeEnum.SINGLE_EVALUATION:
            frame = inspect.currentframe()
            printf(" [NOTE] Single evaluation process do not require this process [{}], continue"
                  .format(frame.f_code.co_name))
            return
        
        for constr, args in constr_with_limits.items():
            i = self.CONSTRAINT.index(constr)
            min_, max_, N_steps = args
            
            assert min_ < max_, "defining range for 1D where min >= max"
            N_steps = N_steps + 1 if (N_steps % 2 == 0) else N_steps
            
            self._deform_lim_max[i] = float(max_)
            self._deform_lim_min[i] = float(min_) # all constraints are float (parse)
            self._N_steps[i]        = N_steps
            
            ## The results and deformations are unfixed, just between the boundaries
            if self.ITERATIVE_METHOD != self.IterativeEnum.VARIABLE_STEP:
                array_ = list(np.linspace(min_, max_, num=N_steps, endpoint=True))
                self.deform_oblate  [i] = []
                self.deform_prolate [i] = array_
                
                self._deformations_map[i] = [[], list(enumerate(array_)) ]
                
    def _setDeformationFromMinimum(self, *args):
        """ 
        :p_min <as list[floats]>, sorted by the different a
        :p_max       '''       '''       '''
        :N_max <as list[floats]>,        '''
        Complete the deformations to the left(oblate) and right(prol)
        for the pairing minimum. dq defined by N_max, final total length= N_max+1
        """
        _def_indx0 = tuple([0 for _ in range(len(self.CONSTRAINT))])
        for i, constr_ in enumerate(self.CONSTRAINT):
            if len(args) > 0:
                p_min = args[0][i]
                p_max = args[1][i]
                N_max = args[2][i]
            else:
                p_min = self._deform_lim_min[i]
                p_max = self._deform_lim_max[i]
                N_max = self._N_steps[i]
            
            if not (p_max!=None and p_min!=None) or N_max == 0:
                ## consider as ITERATIVE_METHOD SINGLE EVALUATION (save the deform)
                q0 = getattr(self._1stSeedMinimum, self.CONSTRAINT_DT[i], 0.0)
                self._deform_base  [i] = q0
                self.deform_prolate[i] = [q0, ]
                self._deformations_map[i][1] = [(0, q0)]
                if i != len(self.CONSTRAINT): 
                    continue
                else:
                    self._results[_def_indx0] = self._1stSeedMinimum
                    return
            
            # p_min = max(p_min, 0.0)
            deform_oblate, deform_prolate = [], []
            dq = round((p_max - p_min) / N_max,  3)
            q0 = getattr(self._1stSeedMinimum, self.CONSTRAINT_DT[i], 0.0)
            if   q0 < p_min: # p0 outside to the left (add the difference to the left)
                p_min = q0 + (q0 - p_min)
            elif q0 > p_max: # p0 outside to the right (add the diff. to the right)
                p_max = q0 + (q0 - p_max)
            
            if  q0 == None: 
                printf(f"[WARNING] _set_pair could not get the main Constraint(i):({i})[",
                       self.CONSTRAINT_DT[i], "]. Default setting it to 0.00")
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
                    
            # self._results [i][1].append(self._1stSeedMinimum)
            self._deform_base[i] = q0
            self._deform_lim_min[i] = min(deform_oblate)  if len(deform_oblate) else p_min
            self._deform_lim_max[i] = max(deform_prolate) if len(deform_prolate)else p_max
            self._N_steps[i]     = len(deform_oblate) + len(deform_prolate)
            ## Variable step fill the 
            if self.ITERATIVE_METHOD != self.IterativeEnum.VARIABLE_STEP:
                self.deform_oblate [i] = deform_oblate
                self.deform_prolate[i] = deform_prolate
                
                self._deformations_map[i][0] = []
                for k, val in enumerate(deform_oblate):
                    self._deformations_map[i][0].append( (-k - 1, val) )
                self._deformations_map[i][1] = list(enumerate(deform_prolate))
                        
        ## Just the last result consist in the main minimum [0,0,0 ...]
        self._results [_def_indx0] = self._1stSeedMinimum
        for i in range(self._temp_seed_minima.__len__()):
            wf_fn_i = f'base_wf_{i}.bin'
            shutil.copy(self._base_wf_filename, wf_fn_i)
            
            self._temp_seed_minima[i] = [ wf_fn_i, self._results[_def_indx0], 
                                          _def_indx0, self._deform_base ]
    
    def run(self, lev=0):
        """
        Iterative Running for each surface 
        """
        if lev == 0: 
            self._checkExecutorSettings()
            self._curr_deform_index = [0 for _ in self.CONSTRAINT]
            self.printExecutionResult(None, print_head=True)
        _last_constr = lev == len(self.CONSTRAINT) - 1
        
        shutil.copy(self._temp_seed_minima[lev][0], 'initial_wf.bin')
                    
        if self.ITERATIVE_METHOD == self.IterativeEnum.VARIABLE_STEP:
            raise ExecutionException(" ITER_METHOD VARIABL_STEP not implemented!")
            self._runVariableStep()
        else:
            start_from_ = 0
            printf(f" ** [lev={lev}] prolate:   start_from_ = ", start_from_, "")
            for k, val in self._deformations_map[lev][1][start_from_:]: # prolate
                ## exclude the first state since it is the original seed
                self._curr_deform_index[lev] = k
                self.inputObj.setConstraints(**{self.CONSTRAINT[lev]: val})
                indx_ = tuple(self._curr_deform_index)
                
                if not _last_constr:
                    if lev == 0:
                        _ = 0
                    self.run(lev=lev + 1)
                    res : self.DTYPE = self._current_result
                else:
                    res : self.DTYPE = self._runUntilConvergence()
                    
                    self._results[indx_] = res
                    self._final_bin_list_data[indx_] = res._exported_filename
                
                if (k == 0):
                    if indx_ in self._results: ## this is not necessary.
                        res_0 = self._results[indx_]
                        b20_0 = getattr(res_0, self.CONSTRAINT_DT[lev])
                        if abs(getattr(res, self.CONSTRAINT_DT[lev]) - b20_0) > 1.0e-3:
                            printf("[WARNING] 1st solution reiterated, values don't match")
                            printf(f"   for constraint value b20_0=[{b20_0}] ")
                    ## copy this solution as the starting for the new minima
                    aux_min_fn = "base_wf_{}.bin".format("_".join([str(x) for x in indx_]))
                    shutil.copy('final_wf.bin', aux_min_fn)
                    self._temp_seed_minima[lev] = [aux_min_fn, res, indx_, 
                                                   self._getCurrentDeformation()]
            
            if self.ITERATIVE_METHOD == self.IterativeEnum.EVEN_STEP_SWEEPING:
                self._run_backwardsSweeping(oblate_part=False, level=lev)
            
            printf(f" ** [lev={lev}] oblate:")
            for k, val in self._deformations_map[lev][0]: # oblate
                self._curr_deform_index[lev] = k
                self.inputObj.setConstraints(**{self.CONSTRAINT[lev]: val})
                
                if not _last_constr:
                    self.run(lev=lev + 1)
                    res : self.DTYPE = self._current_result
                else:
                    res : self.DTYPE = self._runUntilConvergence()
                    
                    _indx = tuple(self._curr_deform_index)
                    self._results[_indx] = res
                    self._final_bin_list_data[_indx] = res._exported_filename
            
            if ((self.ITERATIVE_METHOD == self.IterativeEnum.EVEN_STEP_SWEEPING)
                and (lev)):
                self._run_backwardsSweeping(oblate_part=True, level=lev)
        
        # clear all the base_wf:
        if (lev == 0):
            for f in filter(lambda x: x.startswith('base_wf_'), os.listdir()): 
                os.remove(f)
        ## copy the last solution form the prolate minimum at the level
        # res0_Args = self._temp_seed_minima[lev]
        # shutil.copy(res0_Args[0], 'initial_wf.bin') ## 
    
    # def _runUntilConvergence(self, MAX_STEPS=3):
    #     """ works the same """
    #     return _Base1DTaurusExecutor._runUntilConvergence(self, MAX_STEPS=MAX_STEPS)
    
    # def _executeProgram(self, base_execution=False):
    #     """ does not overwrite any indexable result-value (_current_result is 1)
    #     return res
    
    def _run_backwardsSweeping(self, oblate_part=None, level=False):
        """ 
        Method to reevaluate the execution limits backwards, 
        over-writable method to change the acceptance criteria (impose lower energy)
        """
        if oblate_part == None: 
            raise ExecutionException("Specify the oblate or prolate part")
        if level != self.CONSTRAINT.__len__() - 1: 
            printf(f" ** ! [lev={level}] oblate[{oblate_part}] (back): SKIP")
            return
        
        if oblate_part:
            printf(" ** oblate (back):")
            for k, val in reversed(self._deformations_map[level][0]): # oblate
                self._curr_deform_index[-1] = k
                self.inputObj.setConstraints(**{self.CONSTRAINT[level]: val})
                res  : self.DTYPE = self._executeProgram() # reexecuting
                
                indx_ = (*self._curr_deform_index[:-1], k)
                res0 : self.DTYPE = self._results[indx_]
                if self._backPropagationAcceptanceCriteria(res, res0):
                    self._results[indx_] =  res
                    self._final_bin_list_data[indx_] = res._exported_filename
        else:
            printf(" ** prolate (back):")
            ## exclude the first state since it is the original seed
            ## UPDATE, repeat that solution, the 1st minimum could not be the exact one 
            start_from_ = 0
            for k, val in reversed(self._deformations_map[level][1][start_from_:]): # prolate
                self._curr_deform_index[-1] = k
                self.inputObj.setConstraints(**{self.CONSTRAINT[level]: val})
                res  : self.DTYPE = self._executeProgram()
                # indx_ = k - 1
                
                indx_ = (*self._curr_deform_index[:-1], k)
                res0 : self.DTYPE = self._results[indx_]
                if self._backPropagationAcceptanceCriteria(res, res0):
                    self._results[indx_] =  res
                    self._final_bin_list_data[indx_] = res._exported_filename
    
    ## TODO. Finish!
    # def saveFinalWFprocedure(self, result, base_execution=False):
    #     return _Base1DTaurusExecutor.saveFinalWFprocedure(self, result, base_execution=base_execution)
    
    def _getCurrentDeformation(self):
        """ Auxiliary method to obtain the current deformation value"""
        indx_ = tuple(self._curr_deform_index)
        if self._all_def_values_by_key:
            return self._all_def_values_by_key[indx_]
        
        def_ = ()
        for i in range(self.CONSTRAINT.__len__()):
            part = 0 if indx_[i] < 0 else 1
            def_ = (*def_, dict(self._deformations_map[part])[indx_[i]])
        return def_
    
    def getResultsSortingOrder(self, with_deformation_data = False):
        """
        Get the correct ordering from each constraint (also sorted), from 
        oblate to prolate.
        """
        if not self._all_def_keys_sorted:
            
            keys_sorted = []
            for i in range(len(self.CONSTRAINT)):
                aux = [None,]*2
                for k in (0, 1):
                    aux[k] = [x[0] for x in self._deformations_map[i][k]]
                aux = sorted(aux[0]) + aux[1]
                keys_sorted.append(aux)
            
            self._all_def_keys_sorted = getValueCombinationsSorted(keys_sorted)
        
        keys_sorted = list(filter(lambda x: x in self._results.keys(), 
                                  self._all_def_keys_sorted))
        
        if with_deformation_data:
            if not self._all_def_values_by_key:
                dict_defs = []
                for i in range(len(self.CONSTRAINT)):
                    dict_defs.append( {**dict(self._deformations_map[i][0]),
                                       **dict(self._deformations_map[i][1])} )
                dict_defs_2 = {}
                for indx_ in self._all_def_keys_sorted:
                    vals = [round(dict_defs[i][k], 6) for i, k in enumerate(indx_)]
                    dict_defs_2[indx_] = tuple(vals)
                    
                self._all_def_values_by_key  = dict_defs_2
            
            deform_of_key = []
            for idx_ in keys_sorted:
                deform_of_key.append( self._all_def_values_by_key[idx_] )
            return keys_sorted, deform_of_key
        
        return keys_sorted       # if not with_deformation_data
        
    
    def executionTearDown(self, result:DataTaurus, base_execution, *args, **kwargs):
        return _Base1DTaurusExecutor.executionTearDown(self, result, base_execution, *args, **kwargs)
    
    def exportResults(self, output_filename=None):
        """
        writes a text file with the results dict-like (before the program ends)
        Order: oblate[-N, -N-1, ..., -1] > prolate [0, 1, ..., N']
            for in the order of each constraint:
            key indexing example:
                1D:     1: 0.558        ## DataTaurus as csv
                2D: 1, -2: 0.558, 1.225 ## DataTaurus as csv
                etc.
        """
        res2print = []
        dtype_ = None
        
        keys_and_defs = self.getResultsSortingOrder(with_deformation_data=True)
        for i, indx_ in enumerate(keys_and_defs[0]):
            
            res   = self._results[indx_]
            dvals = keys_and_defs[1][i]
            
            dvals = ','.join([f"{dval:+6.3f}" for dval in dvals])
            keys_ = ','.join([f"{key_:5}" for key_ in indx_])
            line = []
            if self.include_header_in_results_file:
                line.append(f"{keys_}: {dvals}")
            line.append(res.getAttributesDictLike)
            line = self.HEADER_SEPARATOR.join(line)
            
            res2print.append(line) # results are already sorted
            
            if (dtype_ == None) and (res != None):
                dtype_ = res.__class__.__name__
        
        txt_ = '\n' .join(res2print)
        head_ = ", ".join([dtype_, *self.CONSTRAINT_DT])
        txt_ = head_ + '\n' + txt_
        
        if output_filename == None:
            output_filename = self.EXPORT_LIST_RESULTS
        
        with open(output_filename, 'w+') as f:
            f.write(txt_)
    
    def _nameCurrentDeformationIndexString(self):
        """ Overwritable criteria to export the tail of the saved files. """
        deform_str = "_".join([str(x) for x in self._curr_deform_index])
        return deform_str
    #
    # def _namingFilesToSaveInTheBUfolder(self):
    #     """ Valid by overwriting the _nameCurrentDeformationStringIndex
    
class ExeTaurus2D_MultiConstrained(_Base2DTaurusExecutor, ExeTaurus1D_DeformB20):
    
    # CONSTRAINT    : list = []  # InputTaurus Variable to compute
    # CONSTRAINT_DT : list = []  # DataTaurus (key) Variable to compute
    
    def setUp(self, *args, **kwargs):
        """
        :reset_folders = True, give it as key-word argument.
        
        Need to be overwitted to include the option of joining strings as arguments.
            NOTE: It do the same as ExeTaurus1D_DeformB20.setUp(self, *args, **kwargs)
        """
        reset_folder = kwargs.get('reset_folder', True)
        self._DDparams = self.inputObj._DD_PARAMS
        
        args_str = '-'.join(args)
        args_str = '_'+args_str if args_str != '' else ''
        
        self.DTYPE.BU_folder = f'BU_folder{args_str}_{self.interaction}_z{self.z}n{self.n}'
        if reset_folder:
            self.DTYPE.setUpFolderBackUp()
        
        for ext_ in OutputFileTypes.members():
            if os.path.exists(self.interaction+ext_):
                shutil.copy(self.interaction+ext_,  self.DTYPE.BU_folder)
    
    def setUpExecution(self, *args, **kwargs):
        """ Only requires the appending of _binlistdata for each cstr-deformation """
        ExeTaurus1D_DeformB20.setUpExecution(self, *args, **kwargs)
        indx_ = tuple([0,]*self.CONSTRAINT.__len__())
        self._final_bin_list_data[indx_] = self._1stSeedMinimum._exported_filename
    
    def _define_BaseConstraintDeformationAsZero(self):
        """ Set the deformations to zero for all constraints"""
        for i, constr in enumerate(self.CONSTRAINT):
            self._deform_base[i] = 0.0
            setattr(self.inputObj, constr, 0.0)
    
    def globalTearDown(self, zip_bufolder=True, base_calc=False, *args, **kwargs):
        """
        Proceedings for the execution to do. i.e:
            zipping files, launch tests on the results, plotting things
            exporting the list of results-wf final in a list.dat for BMF calculations
            
            export the constrained value as well (from CONSTRAINT_DT)
            
            save results in the PNVAP.
        """
        ## export of the list.dat file
        bins_, outs_ = [], []
        printf(f"\n  [globalTearDown] Export by [{self.CONSTRAINT_DT}]\n")
        ## exportar oblate-reverse order
        for indx_ in self.getResultsSortingOrder():
            tail = self._final_bin_list_data[indx_]
            constr_val = ''
            for j in range(len(self.CONSTRAINT_DT)):
                ctr_j = getattr(self._results[indx_], self.CONSTRAINT_DT[j])
                constr_val += f" {ctr_j:6.3f}"    #.replace('-', '_')
            bins_.append("seed_{}.bin\t{}".format(tail, constr_val))
            outs_.append("res_{}.OUT\t{}" .format(tail, constr_val))
        
        with open('list_dict.dat', 'w+') as f:
            f.write("\n".join(bins_))
        with open('list_outputs.dat', 'w+') as f:
            f.write("\n".join(outs_))
        shutil.copy('list_dict.dat', self.DTYPE.BU_folder)
        shutil.copy('list_outputs.dat', self.DTYPE.BU_folder)
        
        export_fn = self.EXPORT_LIST_RESULTS
        if base_calc:
            args      = ('BASE', *args)
            export_fn = 'BASE-' + self.EXPORT_LIST_RESULTS
        
        args = [self.z,self.n,self.interaction]+list(args)+list(kwargs.values())
        shutil.copy(export_fn, DataTaurus.BU_folder)
        if zip_bufolder:
            if self.CONSTRAINT != None:
                for cnstr in self.CONSTRAINT: args.append(cnstr)
            zipBUresults(DataTaurus.BU_folder, *args)
        
        ## Create a list of wf to do the VAP calculations:
        if self.DTYPE is DataTaurus:
            self._globalTearDown_saveVAPresultsInList(bins_, outs_)
        
        printf( "  [globalTearDown] Done.\n")
    
    def _globalTearDown_saveVAPresultsInList(self, bins_, outs_):
        """
        Auxiliary method to store the mean-field results for further PNPAMP-HWG
        calculations. Requires results as DataTaurus
        """
        os.chdir(self.DTYPE.BU_folder)
        printf(f"\n  [globalTearDown] Saving the results in {os.getcwd()}/PNVAP", )
        # create folder.
        if os.path.exists('PNVAP'): shutil.rmtree('PNVAP')
        os.mkdir('PNVAP')
        
        list_dat = []
        for i, bin_ in enumerate(bins_):
            _aux = bin_.split()
            fn, def_ = _aux[0], "_".join(_aux[1:])
            shutil.copy(fn, 'PNVAP/' + def_ + '.bin')
            fno = outs_[i].split()[0]
            printf(f"     cp: def_=[{def_}] fn[{fn}] fno[{fno}]")
            shutil.copy(fno, 'PNVAP/' + def_ + '.OUT')
            list_dat.append(def_ + '.bin')
        with open('list.dat', 'w+') as f:
            f.write("\n".join(list_dat))
        shutil.move('list.dat', 'PNVAP/')
        os.chdir('..')



