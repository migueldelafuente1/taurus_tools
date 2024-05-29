'''
Created on 19 mar 2024

@author: delafuente

'''

from copy import deepcopy
import shutil
import numpy as np
import os

from tools.helpers import almostEqual, LINE_1, readAntoine, QN_1body_jj,\
    importAndCompile_taurus, printf
from tools.data import DataTaurus, DataTaurusPAV, BaseResultsContainer1D,\
    DataObjectException
from tools.inputs import InputTaurusPAV, InputTaurusMIX
from .executors import ExeTaurus1D_DeformB20
from tools.Enums import OutputFileTypes



class ExeTaurus1D_B20_OEblocking_Ksurfaces_Base(ExeTaurus1D_DeformB20):
    '''
    Protocol to export the different projections of the K momenum in odd-even 
    calculations:
        1. Perform a false ODD-EVEN calculaton (normal outputs etc)
        2. Bock each state in the odd-particle sp-states and minimize again
            2.1 find for each K, in case of not finding it proceed with other seed
            2.2 In case K is not found, ignore K for that deformation
        3. Prepare and export all the components as a surface, also for a later PAV
            in folders "K%%_VAP/"
    '''
    IGNORE_SEED_BLOCKING  = True  ## Option to do False Odd-Even variation
    PARITY_TO_BLOCK       = 1
    FIND_K_FOR_ALL_SPS    = False
    BLOCK_ALSO_NEGATIVE_K = False
    RUN_PROJECTION        = False
    
    _MIN_ENERGY_CRITERIA  = False ## only apply for FIND_K_FOR_ALL_SPS, protocol 
    
    def __init__(self, z, n, interaction, *args, **kwargs):
        
        ExeTaurus1D_DeformB20.__init__(self, z, n, interaction, *args, **kwargs)
        
        self._valid_Ks : list = []
        self._K_results : dict = {}   # not used
        self._K_seed_list : dict = {} # not used
        
        ## Optimization to skip already found sp for a previous K;
        ##   NOTE: for that deformation, we already know that sp will end into 
        ##         the previous K value
        self._sp_blocked_K_already_found : dict = {} # {b20_index: sp_found, ...}
        self._sp_states_obj : dict = {}
        self._current_sp    : int  = 999999
        
        self._blocking_section = False
        self._save_results     = True
        self._previous_bin_path= None
        self._projectionAllowed= self.RUN_PROJECTION
        
        if self.FIND_K_FOR_ALL_SPS:
            self._save_results = False
            self._container    = BaseResultsContainer1D()
            # self._contaienrPAV = BaseResultsContainer1D("TEMP_BU_PAV")
    
    # def setInputCalculationArguments(self, core_calc=False, axial_calc=False, 
    #                                        spherical_calc=False, **input_kwargs):
    #
    #     ExeTaurus1D_DeformB20.setInputCalculationArguments(self, 
    #                                                        core_calc=core_calc, 
    #                                                        axial_calc=axial_calc, 
    #                                                        spherical_calc=spherical_calc, 
    #                                                        **input_kwargs)
    
    def setUpExecution(self, reset_seed=False, valid_Ks=[], *args, **kwargs):
        """
        Set up of the main false OE TES and the arrays for the blocking part.
        """
        ExeTaurus1D_DeformB20.setUpExecution(self, reset_seed=reset_seed, 
                                                   *args, **kwargs)
        
        ## organization only sorted: in the range of valid j
        # self._valid_Ks = [k for k in range(-self._sp_2jmax, self._sp_2jmax+1, 2)]
        # # skip invalid K for the basis, i.e. _sp_2jmin=3/2 -> ..., 5,3,-3,-5 ...
        # self._valid_Ks = list(filter(lambda x: abs(x) >= self._sp_2jmin,
        #                              self._valid_Ks))
        
        valid_states_KP = set()
        for sp_ in range(1, self._sp_dim +1):
            i = 0
            for sh_, deg in self._sp_states.items():
                n, l, j = readAntoine(sh_, l_ge_10=True)
                
                if (-1)**l == self.PARITY_TO_BLOCK:
                    valid_states_KP.add(j) ## add the jz max = K
                
                for mj in range(j, -j -1, -2):
                    i += 1
                    if i == sp_:
                        assert not sp_ in self._sp_states_obj, "Already found"
                        self._sp_states_obj[sp_] = QN_1body_jj(n, l, j, mj)
        
        
        ## optimal organization of the K: 1, -1, 3, -3, ...
        for k in range(self._sp_2jmin, self._sp_2jmax +1, 2):
            if not k in valid_states_KP: 
                continue # skip states with K but other parity
            if valid_Ks != [] and not k in valid_Ks:
                continue
            self._valid_Ks.append(k)
            if self.BLOCK_ALSO_NEGATIVE_K:
                self._valid_Ks.append(-k)
        
        
        for k in self._valid_Ks:
            def_dct = list(map(lambda x: x[0], self._deformations_map[0]))
            def_dct+= list(map(lambda x: x[0], self._deformations_map[1]))
            def_dct.sort()
            def_dct = dict((kk, None) for kk in def_dct)
            
            self._K_results[k]   = def_dct
            self._K_seed_list[k] = deepcopy(def_dct)
            
            self._sp_blocked_K_already_found = deepcopy(def_dct)
            for kk in self._sp_blocked_K_already_found:
                self._sp_blocked_K_already_found[kk] = {}
                for sp_ in range(1, self._sp_dim +1):
                    self._sp_blocked_K_already_found[kk][sp_] = 0 # default
    
    def setUpProjection(self, **params):
        """
        Defines the parameters for the projection of the nucleus.
        The z, n, interaction, com, Fomenko-points, and Jvals from the program
        """
        ExeTaurus1D_DeformB20.setUpProjection(self, **params)
        
        self._list_PAV_outputs = {} # list by K order
        #self.inputObj_PAV.j_max = min( max(self._valid_Ks), self._sp_2jmax)
        
        
    def _KComponentSetUp(self):
        """ 
        Actions common for the creation of K folders or 
        """
        #self.inputObj.grad_type  = 0    if abs(self._current_K) > 3 else 1
        #self.inputObj.iterations = 1000 if abs(self._current_K) > 3 else 500
                
        # Refresh and create folders for vap-blocked results
        _PAR = (1 - self.PARITY_TO_BLOCK)//2
        BU_FLD = self.DTYPE.BU_folder
        BU_FLD_KBLOCK = f"{BU_FLD}/{self._current_K}_{_PAR}_VAP"
        BU_FLD_KBLOCK = BU_FLD_KBLOCK.replace('-', '_')
        # Create new BU folder
        if os.path.exists(BU_FLD_KBLOCK):
            shutil.rmtree(BU_FLD_KBLOCK)
        os.mkdir(BU_FLD_KBLOCK)
        self._exportable_BU_FLD_KBLOCK = BU_FLD_KBLOCK 
        self._exportable_LISTDAT_forK = []
        self._list_PAV_outputs[self._current_K] = []
        printf(f"* Doing 2K={self._current_K} P({self.PARITY_TO_BLOCK}) for TES",
              f"results. saving in [{BU_FLD_KBLOCK}]")
    
    def run(self, fomenko_points=None):
        """
        Modifyed method to obtain the reminization with a blocked state.
        """
        self._blocking_section = False
        ExeTaurus1D_DeformB20.run(self)
        
        if fomenko_points:
            self.inputObj.z_Mphi = fomenko_points[0]
            self.inputObj.n_Mphi = fomenko_points[1]
            self.inputObj_PAV.z_Mphi = fomenko_points[0]
            self.inputObj_PAV.n_Mphi = fomenko_points[1]            
        ##  
        self._blocking_section = True
        if self.numberParity == (0, 0): return
        printf(LINE_1, " [DONE] False Odd-Even TES, begin blocking section")
        printf(f"   Finding all sp-K results: {self.FIND_K_FOR_ALL_SPS}")
        printf(f"   Doing also Projection:    {self.RUN_PROJECTION}")
        printf(f"   Checking also negative K: {self.BLOCK_ALSO_NEGATIVE_K}\n")
        
        self.inputObj.seed = 1
        self.inputObj.eta_grad  = 0.03 
        self.inputObj.mu_grad   = 0.00
        self.inputObj.grad_type = 1
        self.inputObj.iterations = 600
        
        # Perform the projections to save each K component
        for K in self._valid_Ks:
            self._no_results_for_K = True
            self._current_K = K
            self._KComponentSetUp()
            
            self._exportable_txt = {}
            # oblate part
            for prolate in (0, 1):
                for i_def, tail_ in self._final_bin_list_data[prolate].items():
                    
                    self._curr_deform_index = i_def
                    shutil.copy(f"{self.DTYPE.BU_folder}/seed_{tail_}.bin", 
                                "initial_wf.bin")
                    
                    ## NOTE: Projection is not allowed for the _executeProgram,
                    ## in order to do it after checking (or not) all sp-K, 
                    ## PAV is done in _selectStateFromCalculationSetTearDown()
                    self._projectionAllowed = False
                    self._spIterationAndSelectionProcedure(i_def)
                    ## B20 loop
                    printf()
                self._previous_bin_path = None
            if self._no_results_for_K: 
                printf("  [WARNING] No blocked result for 2K=", K)
            # K-loop
        _ = 0
    
    def _spIterationAndSelectionProcedure(self, i_def):
        """
        Iteration over all the single-particle states until getting a correct 
        solution (axial and same K)
        * Getting the first solution or try over all the possible blockings
        """
        ## fix the deformation i to the main constraint
        prolate = int(i_def >= 0)
        b20_ = dict(self._deformations_map[prolate]).get(i_def)
        setattr(self.inputObj, self.CONSTRAINT, b20_)
        
        isNeu = self._sp_dim if self.numberParity[1] else 0
        set_energies = {}
        ## block the states in order
        for sp_ in range(1, self._sp_dim +1):
            self._current_sp = sp_
            ## OPTIMIZATION:
            # * if state has a previous K skip
            # * if state has different jz initial skip
            # * the parity is necessary to match a pattern
            K_prev  = self._sp_blocked_K_already_found[i_def][sp_]
            parity_ = (-1)**self._sp_states_obj[sp_].l
            
            if (K_prev != 0) and (K_prev  != self._current_K): continue
            if self._sp_states_obj[sp_].m != self._current_K : continue
            if parity_ != self.PARITY_TO_BLOCK : continue 
            
            self.inputObj.qp_block = sp_ + isNeu
            
            ## minimize and save only if 2<Jz> = K
            res : DataTaurus = self._executeProgram()
            if not (res.properly_finished and res.isAxial()):
                continue
            if almostEqual(2 * res.Jz, self._current_K, 1.0e-5):
                ## no more minimizations for this deformation
                self._no_results_for_K *= False
                self._K_foundActionsTearDown(res)
                set_energies[sp_] = f"{res.E_HFB:6.4f}"
                if not self.FIND_K_FOR_ALL_SPS: break
            elif sp_ == self._sp_dim:
                printf(f"  [no K={self._current_K}] no state for def[{i_def}]={b20_:>6.3f}")
            else:
                self._K_notFoundActionsTearDown(res)
        
        if (self.FIND_K_FOR_ALL_SPS and not self._no_results_for_K):
            # only apply if multiple E
            self._selectStateFromCalculationSetTearDown(set_energies)
    
    def runProjection(self, **params):
        """
            NOTE: Projection can only be called for blocked functions, occurring
            in the blocking section. This method is called from _executeProgram()
        """
        if self._projectionAllowed:
            if self._blocking_section:
                return ExeTaurus1D_DeformB20.runProjection(self, **params)
            elif self.numberParity == (0, 0):
                return ExeTaurus1D_DeformB20.runProjection(self, **params)
        return
    
    def saveFinalWFprocedure(self, result:DataTaurus, base_execution=False):
        """
        overwriting of exporting procedure:
            naming, dat files, output in the BU folder
        """
        if self._blocking_section:
            if not self._save_results:
                id_ = "sp{}_d{}K{}".format(self._current_sp, 
                                           self._curr_deform_index,
                                           self._current_K)
                self._container.append(result, id_=id_, binary='final_wf.bin',
                                       datfiles=self.SAVE_DAT_FILES)
                return
                
            # copy the final wave function and output with the deformation
            # naming (_0.365.OUT, 0.023.bin, etc )
            #
            # NOTE: if the function is not valid skip or ignore
            i = self._curr_deform_index
            b20_  = self.deform_oblate[-i-1] if (i < 0) else self.deform_prolate[i]
            # prolate = 0 if i < 0 else 1 
            # b20_ = dict(self._deformations_map[prolate]).get(i)
            fndat = f"{b20_:6.3f}".replace('-', '_').strip()
            fnbin = f"{fndat}.bin"
            
            _invalid = result.broken_execution or not result.properly_finished
            if _invalid:
                shutil.move(self.DTYPE.DEFAULT_OUTPUT_FILENAME, 
                            f"{self._exportable_BU_FLD_KBLOCK}/broken_{fndat}.OUT")
                for dat_f in self.SAVE_DAT_FILES:
                    dat_f += '.dat'
                    dat_fn = f"broken_{fndat}_{dat_f}"
                    shutil.move(dat_f, f"{self._exportable_BU_FLD_KBLOCK}/{dat_fn}")
                return
            
            shutil.copy("final_wf.bin", 
                        f"{self._exportable_BU_FLD_KBLOCK}/{fnbin}")
            shutil.move(self.DTYPE.DEFAULT_OUTPUT_FILENAME, 
                        f"{self._exportable_BU_FLD_KBLOCK}/{fndat}.OUT")
            for dat_f in self.SAVE_DAT_FILES:
                dat_f += '.dat'
                dat_fn = f"{fndat}_{dat_f}"
                shutil.move(dat_f, f"{self._exportable_BU_FLD_KBLOCK}/{dat_fn}")
            
            if not fnbin in self._exportable_LISTDAT_forK:
                self._exportable_LISTDAT_forK.append(fnbin)
        else:
            ## Normal execution
            ExeTaurus1D_DeformB20.saveFinalWFprocedure(self, 
                                                       result, base_execution)
    
    def saveProjectedWFProcedure(self, result: DataTaurusPAV):
        """
        Default copy of the function for the deformation into the PAV BU-folder.
        
        After ._save_results option, the PAV is just done after all sp done
        """
        out_args = (self._current_K, self._curr_deform_index)
        outfn    = "K{}_d{}.OUT".format(*out_args)
        
        outpth = "{}/{}".format(self._curr_PAV_result.BU_folder, outfn)
        
        if result.broken_execution or not result.properly_finished:
            outpth = "{}/broken_{}".format(self._curr_PAV_result.BU_folder, outfn)
            shutil.move(self.DTYPE.DEFAULT_OUTPUT_FILENAME, outpth)
            return
        
        if not outfn in self._list_PAV_outputs[self._current_K]:
            self._list_PAV_outputs[self._current_K].append(outfn)
        
        shutil.move(DataTaurusPAV.DEFAULT_OUTPUT_FILENAME, outpth)
    
    def executionTearDown(self, result:DataTaurus, base_execution, *args, **kwargs):
        """
            Separate the two parts for obtaining the 'exportResult()' for 
            minimization after blocking.
        """
        if self._blocking_section:
            if not self._save_results: 
                return
            ## save the list dat into folder
            with open(f'{self._exportable_BU_FLD_KBLOCK}/list.dat', 'w+') as f:
                if len(self._exportable_LISTDAT_forK):
                    f.write("\n".join(self._exportable_LISTDAT_forK) )
            
            K = self._current_K
            with open(f'{self._exportable_BU_FLD_KBLOCK}/' + 
                      self.EXPORT_LIST_RESULTS.replace('TESb20', f'TESb20_K{K}')
                      , 'w+') as f:
                exportable_txt = [self._exportable_txt[k] 
                                  for k in sorted(self._exportable_txt.keys())]
                exportable_txt.insert(0, "{}, {}".format('DataTaurus', 
                                                         self.CONSTRAINT_DT))
                if len(self._exportable_txt):
                    f.write("\n".join(exportable_txt))
        else:
            ## Normal execution
            ExeTaurus1D_DeformB20.executionTearDown(self, result, base_execution, 
                                                    *args, **kwargs)
    
    def projectionExecutionTearDown(self):
        """
        Process to save result after calling runProjection()
        """
        ## save the list dat into folder/ one per K value
        for K in self._list_PAV_outputs.keys():
            with open(f'{self._curr_PAV_result.BU_folder}/list_k{K}_pav.dat', 
                      'w+') as f:
                if len(self._list_PAV_outputs[K]):
                    sort_ = self._list_PAV_outputs[K]
                    sort_ = self._sortListDATForPAVresults(sort_)
                    f.write("\n".join(sort_))
    
    def _K_foundActionsTearDown(self, res: DataTaurus):
        """
        Other exportable actions for founded result in K
        """
        i   = self._curr_deform_index
        prolate = 0 if i < 0 else 1 
        b20 = dict(self._deformations_map[prolate]).get(i)
        sp_index = self._current_sp
        
        if self._save_results:
            id_, res = self._choosen_state_data
            sp_index = int(id_.split('_')[0].replace('sp', '')) if id_ else 0
            printf("   [OK] {:>3} {:>11} <jz>= {:4.1f}, b20={:>6.3f}  E_hfb={:6.3f}"
                  .format(sp_index, self._sp_states_obj[sp_index].shellState,  
                          res.Jz, res.b20_isoscalar, res.E_HFB))
        else:
            printf("      . {:>3} {:>11} <jz>= {:4.1f}, b20={:>6.3f}  E_hfb={:6.3f}"
                  .format(sp_index, self._sp_states_obj[sp_index].shellState,  
                          res.Jz, res.b20_isoscalar, res.E_HFB))
        ## Append the exportable result file
        line = []
        if self.include_header_in_results_file:
            line.append(f"{i:5}: {b20:+6.3f}")
        line.append(res.getAttributesDictLike)
        self._exportable_txt[i] = self.HEADER_SEPARATOR.join(line)
        
        self._sp_blocked_K_already_found[i][sp_index] = self._current_K
    
    def _K_notFoundActionsTearDown(self, res: DataTaurus):
        """
        Actions to discard or notice invalid result.
        """
        sp_index = self._current_sp
        
        if not self._save_results:
            printf("   [xx] {:>3} {:>11} <jz>= {:4.1f}, b20={:>6.3f}  E_hfb={:6.3f} axial={}"
                  .format(sp_index, self._sp_states_obj[sp_index].shellState,  
                          res.Jz, res.b20_isoscalar, res.E_HFB, res.isAxial()))
            return
        else:
            printf("      X {:>3} {:>11} <jz>= {:4.1f}, b20={:>6.3f}  E_hfb={:6.3f} axial={}"
                  .format(sp_index, self._sp_states_obj[sp_index].shellState,  
                          res.Jz, res.b20_isoscalar, res.E_HFB, res.isAxial()))
        invalid_fn = self._list_PAV_outputs[self._current_K].pop()
        invalid_fn = f"{self._curr_PAV_result.BU_folder}/{invalid_fn}"
        
        outpth = invalid_fn.replace(".OUT", "_invalid.OUT")
        shutil.move(invalid_fn, outpth)
        
    def _selectionCriteriaForState(self):
        """
        When several blocked K states lead to different energies (FIND_ALL_K=True),
        stablish a selection for the correct wf for PAV.
        """
        if self._MIN_ENERGY_CRITERIA:
            ## Energy_criteria
            id_min, res, bin_, datfiles = self._energy_CriteriaForStateSelection()
        else:
            ## Overlap citeria: <def_1, def_2(K)> max or not broken for MF-PAV
            id_min, res, bin_, datfiles = self._overlap_CriteriaForStateSelection()
        
        self._current_result = res
        ## Copy into main folder as in a normal result execution
        ## id_ format: "sp{}_d{}K{}"
        if id_min != None:
            self._current_sp = int(id_min.split('_')[0].replace('sp', ''))
        else:
            self._current_sp = None
        return id_min, res, bin_, datfiles
    
    
    def _energy_CriteriaForStateSelection(self):
        
        id_min = None
        e_min : DataTaurus  = 99999999
        printf("     Energy criteria begins ---------------------")
        for id_, res in self._container.getAllResults().items():
            if res.E_HFB < e_min:
                id_min = id_
                e_min  = res.E_HFB
                printf(f"   *( pass) energy_{id_}= {e_min:6.5f}")
            else:
                printf(f"   *(large) energy_{id_}= {res.E_HFB:6.5f}")
        printf("     Final state selected =", id_min, "\n -----------------")
        res, bin_, datfiles  = self._container.get(id_min)
        return id_min, res, bin_, datfiles
    
    def _overlap_CriteriaForStateSelection(self):
        """
        Overlap between different states <L | R> from the surface,
        """
        if not self._previous_bin_path:
            ## No result, use the minimum energy result or first result
            id_sel, res, bin_, datfiles = self._energy_CriteriaForStateSelection()
            printf("   * Initial state selected =", id_sel, "\n ---------------")
        else:
            printf("   * Overlap criteria begins ---------------")
            shutil.copy(self._previous_bin_path, 'left_wf.bin')
            
            id_sel, overlapm1_min = None, 999999
            for id_, res in self._container.getAllResults().items():
                bin = self._container.get(id_)[1]
                shutil.copy(f'{self._container.BU_folder}/{bin}', 'right_wf.bin')
                
                ## No PAV projection (only PN if VAP wf have it)
                inp_pav = self.inputObj_PAV.copy()
                inp_pav.setUpNoPAVCalculation()
                if self.inputObj.z_Mphi > 0: inp_pav.z_Mphi = self.inputObj.z_Mphi
                if self.inputObj.n_Mphi > 0: inp_pav.n_Mphi = self.inputObj.n_Mphi 
                
                res_ov: DataTaurusPAV = None
                for empty_states_case in (0, 1):
                    inp_pav.empty_states = empty_states_case
                    
                    fn_ = 'check_overlap'
                    with open(f'{fn_}.inp', 'w+') as f: 
                        f.write(inp_pav.getText4file())
                    
                    try:
                        os.system(f'./taurus_pav.exe < {fn_}.inp > {fn_}.out')
                        res_ov = DataTaurusPAV(self.z, self.n, f'{fn_}.out')
                    except BaseException:
                        if res_ov == None:
                            printf("      [Err] No solution")
                        continue
                    
                    if res_ov.broken_execution or len(res_ov.proj_norm) == 0:
                        printf("      [Err] Solution is broken, emptyst =",
                              empty_states_case)
                        continue
                    overl  = res_ov.proj_norm[0]
                    if 1 - abs(overl) < overlapm1_min:  ## ovelap has an arbitrary phase 
                        overlapm1_min = 1 - abs(overl)
                        id_sel = id_
                        printf(f"     *( pass) overlap_{id_}= {overl:6.5f}")
                    else:
                        printf(f"     *(large) overlap_{id_}= {overl:6.5f}")
                    break
            
            ## Highly 
            if id_sel == None:
                id_sel = self._container.get(None, list_index_element=0)[-1]
                ## TODO: Other idea is to use the again the E_min criteria.
                printf("     Final state selected = None / using the first value", 
                      id_sel, "\n ---------------")
            else:
                printf("     Final state selected =", id_sel, "\n ---------------")
            os.remove(self._previous_bin_path)
            res, bin_, datfiles  = self._container.get(id_sel)
        
        ## Update the previous result binary
        self._previous_bin_path = f"previous_wf_{id_sel}.bin"
        src = "{}/{}".format(self._container.BU_folder, bin_)
        shutil.copy(src, self._previous_bin_path)
        
        return id_sel, res, bin_, datfiles
    
    def _runningAfterSelection(self, id_sel, bin_, datfiles):
        """
            To be run in _selectStateFromCalculationSetTearDown()
        
        Process of copying the resutls from fully converged results in the main
        folder for normal process (PAV and teardown organization in BU subfolders
        
        Method to be overwritten for unconverged solutions.
        """
        ## This block export the chosen solution as if the program where 
        ## executed. 
        src = "{}/{}.OUT".format(self._container.BU_folder, id_sel)
        shutil.copy(src, self.DTYPE.DEFAULT_OUTPUT_FILENAME)
        src = "{}/{}".format(self._container.BU_folder, bin_)
        shutil.copy(src, 'final_wf.bin')
        if datfiles:
            for file_ in datfiles:
                if not '.dat' in file_: file_ += '.dat'
                file_2 = file_.replace(f"_{id_sel}", "")
                shutil.copy("{}/{}".format(self._container.BU_folder, file_), 
                            file_2)
        if self.RUN_PROJECTION:
            self.runProjection()
    
    def _selectStateFromCalculationSetTearDown(self, set_energies):
        """
        :set_energies <dict[sp] : E_sp> apply when no selection criteria is needed.
        After all the results evaluated for each sp for a def/K case:
            1. choose the best result (minimal E_hfb/Overlap) if not unique 
            2. Copy the binary/ output/ selected to main folder.
                VAP
                PAV if used.
            3. Apply the methods save_WF modified as in the 1st solution case.
        """
        self._save_results = True
        dont_select = len(set(set_energies.values())) == 1
        ## NOTE: It's not necessary, attr _container has the results in order.
                
        ## Select the wavefunciton and files to save..
        if dont_select:
            aux_ = self._container.get(None, list_index_element=0)
            res, bin_, datfiles, id_sel = aux_
            printf("     (All equal) No selection, got:", id_sel, 
                  f" E= {res.E_HFB:6.5f}  -------------")
            self._choosen_state_data = (id_sel, res)
            # prepare / update the auxiliary wf for Overlap-criteria
            if not self._MIN_ENERGY_CRITERIA:
                self._previous_bin_path = f"previous_wf_{id_sel}.bin"
                src = "{}/{}".format(self._container.BU_folder, bin_)
                shutil.copy(src, self._previous_bin_path)
        else:
            printf("      Different results after blocking ... selecting ")
            id_sel, res, bin_, datfiles = self._selectionCriteriaForState()
            self._choosen_state_data = (id_sel, res)
        
        if id_sel != None:
            self._runningAfterSelection(id_sel, bin_, datfiles)
            
            ## Execute exporting functions as it where the normal run
            self.saveFinalWFprocedure(res, False)
            self.executionTearDown(res, False)
            self._K_foundActionsTearDown(res)
            
            ## Copy all elements in TEMP_BU folder to BU
            if not dont_select:
                dst = '/BU_states_d{}K{}'.format(self._curr_deform_index,
                                                 self._current_K)
                shutil.copytree(self._container.BU_folder, self.DTYPE.BU_folder+dst)
        else:
            printf(" [No Result] found, PAV and coping ignored.")
            
        ## Reset all
        self._container.clear()
        self._save_results = False
    
    def globalTearDown(self, zip_bufolder=True, *args, **kwargs):
        """
        Same exporting, but also including a map for the blocked states
        """
        lines = [f"{len(self._sp_blocked_K_already_found)} {self._sp_dim}"]
        for i in self._sp_blocked_K_already_found.keys():
            for sp_ in range(1, self._sp_dim +1):
                
                lines.append((i, sp_, self._sp_blocked_K_already_found[i][sp_]))
                lines[-1] = "{} {} {}".format(*lines[-1]) 
        
        with open("blocked_state_def_K.txt", 'w+') as f:
            f.write("\n".join(lines))
        
        shutil.copy("blocked_state_def_K.txt", self.DTYPE.BU_folder)
        
        ## main export
        ExeTaurus1D_DeformB20.globalTearDown(self, zip_bufolder=zip_bufolder, 
                                                  *args, **kwargs)
        
        ## Remove previous_wf files in main folder
        prev_bins = filter(lambda x: x.startswith('previous_wf_sp'), os.listdir())
        for file_ in prev_bins: os.remove(file_)



class ExeTaurus1D_B20_OEblocking_Ksurfaces(ExeTaurus1D_B20_OEblocking_Ksurfaces_Base):
    
    """
    Accelerated way to obtain the blocking-overlap comparison,
        1. first swap of sp-states is done for rather small number of iterations
           minimization is not achieved but enought to evaluate <ref | 1 | sp phi>
           this is ommited for starting case.
        2. With the overlap results, reevaluate the optimal solution for the 
           convergence precission. After that evaluate the PAV over itself.
    """
    
    FULLY_CONVERGE_BLOCKING_ITER = False  ## Get the final blocked-states solution
    
    def run(self, fomenko_points=None, fully_converge_blocking_sts=False):
        """
        Modifyed method to obtain the reminization with a blocked state.
        """
        self._blocking_section = False
        ExeTaurus1D_DeformB20.run(self)
        
        self.FULLY_CONVERGE_BLOCKING_ITER = fully_converge_blocking_sts
        
        if fomenko_points:
            self.inputObj.z_Mphi = fomenko_points[0]
            self.inputObj.n_Mphi = fomenko_points[1]
            self.inputObj_PAV.z_Mphi = fomenko_points[0]
            self.inputObj_PAV.n_Mphi = fomenko_points[1]            
        ##  
        self._blocking_section = True
        if self.numberParity == (0, 0): return
        printf(LINE_1, " [DONE] False Odd-Even TES, begin blocking section")
        printf(f"   Finding all sp-K results: {self.FIND_K_FOR_ALL_SPS}")
        printf(f"   Doing also Projection:    {self.RUN_PROJECTION}")
        printf(f"   Checking also negative K: {self.BLOCK_ALSO_NEGATIVE_K}\n")
        printf(f"   Valid Ks = {self._valid_Ks}")
        
        self.inputObj.seed = 1
        self.inputObj.eta_grad  = 0.03 
        self.inputObj.mu_grad   = 0.00
        self.inputObj.grad_type = 1
        _default_vap_iters = self.inputObj.iterations
        
        # Perform the projections to save each K component
        for K in self._valid_Ks:
            self._no_results_for_K = True
            self._current_K = K
            self._KComponentSetUp()
            
            self._exportable_txt = {}
            # oblate part
            for prolate in (0, 1):
                for i_def, tail_ in self._final_bin_list_data[prolate].items():
                    
                    self._curr_deform_index = i_def
                    shutil.copy(f"{self.DTYPE.BU_folder}/seed_{tail_}.bin", 
                                "initial_wf.bin")
                    
                    self.inputObj.iterations = _default_vap_iters
                    if not i_def in (-1, 0): self.inputObj.iterations = 100
                    ## NOTE: Projection is not allowed for the _executeProgram,
                    ## in order to do it after checking (or not) all sp-K, 
                    ## PAV is done in _selectStateFromCalculationSetTearDown()
                    self._projectionAllowed = False
                    self._spIterationAndSelectionProcedure(i_def)
                    
                self._previous_bin_path = None
            if self._no_results_for_K: 
                printf("  [WARNING] No blocked result for 2K=", K)
            # K-loop
        _ = 0
    
    def _spIterationAndSelectionProcedure(self, i_def):
        """
        Iteration over all the single-particle states until getting a correct 
        solution (axial and same K)
        * Getting the first solution or try over all the possible blockings
        """
        if self.FULLY_CONVERGE_BLOCKING_ITER:
            ExeTaurus1D_B20_OEblocking_Ksurfaces_Base.\
                _spIterationAndSelectionProcedure(self, i_def)
            return
        
        ## fijar la deformacion i en la coordenada a constrain
        prolate = int(i_def >= 0)
        b20_ = dict(self._deformations_map[prolate]).get(i_def)
        setattr(self.inputObj, self.CONSTRAINT, b20_)
        
        isNeu = self._sp_dim if self.numberParity[1] else 0
        set_energies = {}
        ## block the states in order
        for sp_ in range(1, self._sp_dim +1):
            self._current_sp = sp_
            ## OPTIMIZATION:
            # * if state has a previous K skip
            # * if state has different jz initial skip
            # * the parity is necessary to match a pattern
            K_prev  = self._sp_blocked_K_already_found[i_def][sp_]
            parity_ = (-1)**self._sp_states_obj[sp_].l
            
            if (K_prev != 0) and (K_prev  != self._current_K): continue
            if self._sp_states_obj[sp_].m != self._current_K : continue
            if parity_ != self.PARITY_TO_BLOCK : continue 
            
            self.inputObj.qp_block = sp_ + isNeu
            
            ## minimize and save only if 2<Jz> = K
            res : DataTaurus = self._executeProgram()
            valid_ = abs(res._evol_obj.e_hfb[-2] - res._evol_obj.e_hfb[-1]) < 0.1
            valid_ = valid_ or res._evol_obj.grad[-1] < 0.1
            if self._curr_deform_index in (0, -1):
                valid_ = valid_ and res.properly_finished
            
            if not (valid_ and res.isAxial()):
                continue
            if almostEqual(2 * res.Jz, self._current_K, 1.0e-5):
                ## no more minimizations for this deformation
                self._no_results_for_K *= False
                self._K_foundActionsTearDown(res)
                set_energies[sp_] = f"{res.E_HFB:6.4f}"
                if not self.FIND_K_FOR_ALL_SPS: break
            elif sp_ == self._sp_dim:
                printf(f"  [no K={self._current_K}] no state for def[{i_def}]={b20_:>6.3f}")
            else:
                self._K_notFoundActionsTearDown(res)
        
        if (self.FIND_K_FOR_ALL_SPS and not self._no_results_for_K):
            # only apply if multiple E
            self._selectStateFromCalculationSetTearDown(set_energies)
    
    def _runningAfterSelection(self, id_sel, bin_, datfiles):
        """
            to be run in _selectStateFromCalculationSetTearDown()
        
        Process of copying the resutls from fully converged results in the main
        folder for normal process (PAV and teardown organization in BU subfolders
        
        Method to be overwritten for unconverged solutions.
        """
        
        if self.FULLY_CONVERGE_BLOCKING_ITER or self._curr_deform_index in (0,-1):
            ## First states are converged to ensure a good starting value
            ExeTaurus1D_B20_OEblocking_Ksurfaces_Base.\
                _runningAfterSelection(self, id_sel, bin_, datfiles)
            return
        ## This block export the chosen solution as if the program where 
        ## executed.
        src = "{}/{}".format(self._container.BU_folder, bin_)
        shutil.copy(src, 'initial_wf.bin')
                
        self.inputObj.iterations = 1000
        self.inputObj.qp_block   = 0 # The 1st result has already been blocked
        with open(self.ITYPE.DEFAULT_INPUT_FILENAME, 'w+') as f:
            f.write(self.inputObj.getText4file())
        ## NOTE: Run taurus_vap this way to avoid save-results tear down stuff.
        os.system('./taurus_vap.exe < {} > {}'
                  .format(self.ITYPE.DEFAULT_INPUT_FILENAME,
                          self.DTYPE.DEFAULT_OUTPUT_FILENAME))
        ## NOTE: .dat files are generated and not to be copied
        
        if self.RUN_PROJECTION: self.runProjection()

class _SlurmJob1DPreparation():
    
    """
        This auxiliary class complete the script templates to keep along with 
    the 1-Dimension wf- PAV calculation:
        SUB_1: group job for the paralelization
        JOB_1: unit job for SUB_1
        CAT  : preparation to extract all the resultant projected matrix elements
        HWG  : sub-job to iterate the HWG calculation by J
    
    Usage:
        Giving all necessary arguments, the instance saves the script texts in
        the attributes:
            job_1
            sub_1
            script_cat
            script_hwg
    """
    
    __TEMPLATE_SLURM_JOB_1 = """#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-ARRAY_JOBS_LENGTH
#
## max = N*(N+1)/2 being N the number of q-states (prompt from preparegcm.f)

export OMP_NUM_THREADS=1
ulimit -s unlimited
var=$SLURM_ARRAY_TASK_ID

full_path=$PWD"/"
cd $full_path
cd $var

workdir=$PWD
mkdir -p /scratch/delafuen/
mkdir -p /scratch/delafuen/$SLURM_JOB_ID
chmod 777 PROGRAM
cp -r  PROGRAM INPUT_FILE left_wf.bin right_wf.bin /scratch/delafuen/$SLURM_JOB_ID
cp -r  HAMIL.* /scratch/delafuen/$SLURM_JOB_ID

cd  /scratch/delafuen/$SLURM_JOB_ID

./PROGRAM < INPUT_FILE > OUT

mv /scratch/delafuen/$SLURM_JOB_ID/OUT $workdir/
#mv /scratch/delafuen/$SLURM_JOB_ID/*.me $workdir/
mv /scratch/delafuen/$SLURM_JOB_ID/*.bin $workdir/
rm /scratch/delafuen/$SLURM_JOB_ID/*
rmdir /scratch/delafuen/$SLURM_JOB_ID/
"""
    
    __TEMPLATE_SLURM_SUB_1 = """#!/bin/bash
## max = N*(N+1)/2 being N the number of q-states (prompt from preparegcm.f)

tt="1-23:59:59"
rang="1-ARRAY_JOBS_LENGTH"

sbatch --output=/dev/null --array $rang --time $tt  $PWD/job_1.x
"""
    
    __TEMPLATE_CAT_ME_STATES = """#!/bin/bash
rm projmatelem_states.bin
mkdir outputs_PAV
rm outputs_PAV/* 

for var in {1..ARRAY_JOBS_LENGTH}; do
var3=$(cat gcm_3 | head -$var | tail -1 | awk '{print$1}')
cat $var3"/projmatelem_states.bin" >> projmatelem_states.bin

cp $var3"/OUT" outputs_PAV"/OUT_"$var3
done

cp gcm* outputs_PAV"""
    
    __TEMPLATE_CAT_ME_STATES_PYTHON = """import shutil, os

OUT_fld  = 'outputs_PAV'
pme_file = 'projmatelem_states.bin'
if os.path.exists(OUT_fld):  shutil.rmtree(OUT_fld)
if os.path.exists(pme_file): os.    remove(pme_file)
os.mkdir(OUT_fld)

def copy_stuff(dir_list):
    for fld_ in dir_list:
        fld_ = fld_.strip()
        if os.path.exists(fld_):
            if pme_file in os.listdir(fld_): 
                os.system("cat {}/{} >> {}".format(fld_, pme_file, pme_file))
            else: print(" [ERROR] not found for {}".format(fld_))
            if 'OUT' in os.listdir(fld_):
                shutil.copy("{}/OUT".format(fld_), 
                            "{}/OUT_{}".format(OUT_fld, fld_ ))
            else: print("     [ERROR 2] not found OUT for {}".format(fld_))
    print("* done for all files")

if os.path.exists('gcm_3'):  # gcm_file
    with open('gcm_3', 'r') as f:
        dir_list = f.readlines()
        print(dir_list)
        copy_stuff(dir_list)
else: # without gcm_file
    dir_list = list(filter(lambda x: os.path.isdir(x) and x.isdigit(), os.listdir()))
    copy_stuff(dir_list.sort())"""
    
    __TEMPLATE_JOB_HWX = """#!/bin/bash

ulimit -s unlimited 

LIST_JVALS
chmod 777 PROGRAM
for var in $LIST; do
sed s/"J_VAL"/$var/ INPUT_FILE > INP0
./PROGRAM < INP0 > $var".dat"
done
"""
    
    __TEMPLATE_PREPARE_PNPAMP = """#!/bin/bash
## max = N*(N+1)/2 being N the number of q-states (prompt from preparegcm.f)

for var in {1..ARRAY_JOBS_LENGTH}; do
var1=$(cat gcm | head -$var | tail -1 | awk '{print$1}')
var2=$(cat gcm | head -$var | tail -1 | awk '{print$2}')
var3=$(cat gcm | head -$var | tail -1 | awk '{print$3}')
var4=$(cat gcm | head -$var | tail -1 | awk '{print$4}')

cp $var1 left_wf.bin
cp $var2 right_wf.bin
mkdir $var
cp PROGRAM INPUT_FILE left_wf.bin right_wf.bin HAMIL.* $var
cd $var
chmod 777 PROGRAM
cd ../
done"""
    
    
    TAURUS_PAV = 'taurus_pav.exe'
    TAURUS_HWG = 'taurus_mix.exe'
    
    class ArgsEnum:
        JOBS_LENGTH = 'ARRAY_JOBS_LENGTH'
        INPUT_FILE = 'INPUT_FILE'
        PROGRAM    = 'PROGRAM'
        LIST_JVALS = 'LIST_JVALS'
        HAMIL = 'HAMIL'
    
    def __init__(self, interaction, number_of_wf, valid_J_list, 
                 PAV_input_filename='', HWG_input_filename=''):
        """
        Getting all the jobs for
        """
        self.hamil = interaction
        self.jobs_length = str(number_of_wf * (number_of_wf + 1) // 2)
        if (HWG_input_filename == ''):
            HWG_input_filename = self.ArgsEnum.INP_hwg
        if (PAV_input_filename == ''):
            PAV_input_filename = self.ArgsEnum.INP_pav
        
        
        ## JOB-PARALLEL
        self.job_1 = self.__TEMPLATE_SLURM_JOB_1
        self.job_1 = self.job_1.replace(self.ArgsEnum.JOBS_LENGTH,
                                        self.jobs_length)
        self.job_1 = self.job_1.replace(self.ArgsEnum.HAMIL, self.hamil)
        self.job_1 = self.job_1.replace(self.ArgsEnum.INPUT_FILE, 
                                        PAV_input_filename)
        self.job_1 = self.job_1.replace(self.ArgsEnum.PROGRAM, self.TAURUS_PAV)
        
        self.sub_1 = self.__TEMPLATE_SLURM_SUB_1
        self.sub_1 = self.sub_1.replace(self.ArgsEnum.JOBS_LENGTH,
                                        self.jobs_length)
        ## PREPARE PNAMP
        self.prepare_pnpamp = self.__TEMPLATE_PREPARE_PNPAMP
        self.prepare_pnpamp = self.prepare_pnpamp.replace(self.ArgsEnum.JOBS_LENGTH,
                                                          self.jobs_length)
        self.prepare_pnpamp = self.prepare_pnpamp.replace(self.ArgsEnum.HAMIL,
                                                          self.hamil)
        self.prepare_pnpamp = self.prepare_pnpamp.replace(self.ArgsEnum.INPUT_FILE,
                                                          PAV_input_filename)
        self.prepare_pnpamp = self.prepare_pnpamp.replace(self.ArgsEnum.PROGRAM, 
                                                          self.TAURUS_PAV)
        ## CAT
        self.script_cat = self.__TEMPLATE_CAT_ME_STATES
        self.script_cat = self.script_cat.replace(self.ArgsEnum.JOBS_LENGTH,
                                                  self.jobs_length)
        
        ## HWG
        J_vals = " ".join([str(j) for j in valid_J_list])
        J_vals = f'LIST="{J_vals}"'
        self.script_hwg = self.__TEMPLATE_JOB_HWX
        self.script_hwg = self.script_hwg.replace(self.ArgsEnum.LIST_JVALS,
                                                  J_vals)
        self.script_hwg = self.script_hwg.replace(self.ArgsEnum.INPUT_FILE,
                                                  HWG_input_filename)
        self.script_hwg = self.script_hwg.replace(self.ArgsEnum.PROGRAM,
                                                  self.TAURUS_HWG)
    
    def getScriptsByName(self):
        
        scripts_ = {
            'sub_1.x': self.sub_1, 
            'job_1.x': self.job_1,
            'hw.x': self.script_hwg,
            'cat_states.me.x': self.script_cat,
            'cat_states.py'  : self.__TEMPLATE_CAT_ME_STATES_PYTHON,
            'run_pnamp.x': self.prepare_pnpamp,
        }
        
        return scripts_

class ExeTaurus1D_B20_KMixing_OEblocking(ExeTaurus1D_B20_OEblocking_Ksurfaces):
    
    """
    Protocol to evaluate the K mixing of states of different b_20, as in the
    parent executor class. This process do the same false odd-even meanfield,
    block each state of each J-Pi.
    
    As a final step, it reorganizes the results for b20 folders with each 
    <K|H|K'> matrix elements in order to operate in parallel the program 
    "taurus_pav.exe" for PAV projections.
    \b20_1
        \1 <1/2| 1/2>, input_pav.INP, hamil_files..., taurus_pav.exe
        \2 <1/2| 3/2>, " "
        \3 <1/2| 5/2>, " "
        \4 <3/2| 3/2>, " "
        ...
        \6 <5/2| 5/2>, " "
    \b20_2
        ...
    ...
    
    The script might or not to execute or not each node, however, it can prepare
    all the scripts for the slurm execution and the concatenation of the 
    "projmatelem_states.bin", HWG folder and its bash-scripts.
    
    """
    
    def _KComponentSetUp(self):
        ExeTaurus1D_B20_OEblocking_Ksurfaces._KComponentSetUp(self)
        
        if self._current_K != self._valid_Ks[0]: 
            return
        
        self._export_PAV_Folders = {}
        # Proceed to create the folders for each deformation
        for prolate in (0, 1):
            for i in self._final_bin_list_data[prolate].keys():
                
                BU_FLD_DEF_BLOCK = f"{self.DTYPE.BU_folder}/def{i}_PAV"
                BU_FLD_DEF_BLOCK = BU_FLD_DEF_BLOCK.replace('-', '_')
                # Create new BU folder
                if os.path.exists(BU_FLD_DEF_BLOCK):
                    shutil.rmtree(BU_FLD_DEF_BLOCK)
                os.mkdir(BU_FLD_DEF_BLOCK)
                self._export_PAV_Folders[i] = BU_FLD_DEF_BLOCK + '/'
                    
    
    def _K_foundActionsTearDown(self, res:DataTaurus):
        """ Overwriting to save the results in d """        
        ExeTaurus1D_B20_OEblocking_Ksurfaces.\
            _K_foundActionsTearDown(self, res)
        
        if not self._save_results:  
            ## ALL SP results require not to store until all done
            return
        _PAR = (1 - self.PARITY_TO_BLOCK)//2
        dest_ = f"{self._current_K}_{_PAR}.bin"
        shutil.copy('final_wf.bin', 
                    self._export_PAV_Folders[self._curr_deform_index] + dest_)
    
    def _get_gcm_filetexts(self, list_wf):
        """
        Creates auxiliary files for PNVAP_preparation scripts in linux.
                gcm      : "file1   file2   i1 i2" ...
                gcm_3    : "k" ...
                gcm_diag : "k" ... for k where file1 == file2
        
            :list_wf <list>: wf names in order
            
            :Returns: gcm, gcm_3, gcm_diag
        """
        gcm = []
        gcm_3 = []
        gcm_diag = []
        
        k = 0
        for i, file_1 in enumerate(list_wf):
            for j in range(i, len(list_wf)):
                file_2 = list_wf[j]
                
                k += 1
                gcm_3.append(str(k))
                gcm.append("{:<36}\t{:<36}\t{:3.0f} {:3.0f}"
                           .format(file_1, file_2, i+1, j+1))
                if i == j:
                    gcm_diag.append(str(k))
        
        gcm = "\n".join(gcm)
        gcm_3 = "\n".join(gcm_3)
        gcm_diag = "\n".join(gcm_diag)
        gcm_files = {'gcm': gcm, 'gcm_3': gcm_3, 'gcm_diag': gcm_diag}
        return gcm_files
    
    def globalTearDown(self, zip_bufolder=True, *args, **kwargs):
        
        ExeTaurus1D_B20_OEblocking_Ksurfaces\
            .globalTearDown(self, zip_bufolder=zip_bufolder, *args, **kwargs)       
        
        ## Import the programs if they do not exist
        importAndCompile_taurus(pav= not os.path.exists(InputTaurusPAV.PROGRAM),
                                mix= not os.path.exists(InputTaurusMIX.PROGRAM))
            
        ## Introduce the jobfiles in case of using slurm.
        for prolate in (0, 1):
            for i in self._final_bin_list_data[prolate].keys():
                FLD_ = self._export_PAV_Folders[i]
                
                k_list = []
                for K in self._valid_Ks:
                    _PAR = (1 - self.PARITY_TO_BLOCK)//2
                    dest_ = f"{K}_{_PAR}.bin"
                    if dest_ in os.listdir(FLD_): 
                        k_list.append(dest_)
                # k_list = k_list + ['3_0.bin', '5_0.bin']
                
                ## save auxiliary gcm, gcm_3, gcm_diag files
                gcm_files = self._get_gcm_filetexts(k_list)
                for f_, txt_ in gcm_files.items():
                    with open(FLD_+f_, 'w+') as f:
                        f.write(txt_)
                
                valid_J_list = list(filter(lambda x: x>0, self._valid_Ks))
                scr_x = _SlurmJob1DPreparation(self.interaction,  
                                               len(k_list), valid_J_list, 
                                               InputTaurusPAV.DEFAULT_INPUT_FILENAME, 
                                               InputTaurusMIX.DEFAULT_INPUT_FILENAME)
                scr_files = scr_x.getScriptsByName()
                
                ## HWG program and script prepared
                FLD_3 = FLD_+'HWG'
                os.mkdir(FLD_3)
                f_ = 'hw.x'
                with open(FLD_3+'/'+f_, 'w+') as f:
                    f.write(scr_files.pop(f_))
                shutil.copy(InputTaurusPAV.PROGRAM, FLD_3)
                
                ## PNPAMP files
                for f_, txt_ in scr_files.items():
                    with open(FLD_+f_, 'w+') as f:
                        f.write(txt_)
                
                ## create all the folders for PNPAMP
                inp_pav = InputTaurusPAV.DEFAULT_INPUT_FILENAME
                with open(FLD_+'gcm', 'r') as fr:
                    folder_list = fr.readlines()
                                     
                for k_fold, line in enumerate(folder_list):
                    f1, f2, i, j = [f_.strip() for f_ in line.split()]
                    
                    FLD_2 = f"{FLD_}/{k_fold+1}"
                    os.mkdir(FLD_2)
                    shutil.copy(FLD_+f1, f"{FLD_2}/left_wf.bin")
                    shutil.copy(FLD_+f2, f"{FLD_2}/right_wf.bin")
                    for hty in OutputFileTypes.members() + ['.red', ]:
                        if os.path.exists(self.interaction+hty):
                            shutil.copy(self.interaction+hty, FLD_2)
                    
                    inp_pav = f"{FLD_2}/{InputTaurusPAV.DEFAULT_INPUT_FILENAME}"
                    with open(inp_pav, 'w+') as f_:
                        f_.write(self.inputObj_PAV.getText4file())
                    shutil.copy(InputTaurusPAV.PROGRAM, FLD_2)
                    
                    
                _  = 0
                
                
        ## Execute projection.
        _ = 0


