'''
Created on Jan 10, 2023

@author: Miguel

Module for script setting

'''
import os
import shutil
from numpy.random import randint
from copy import deepcopy, copy

from tools.inputs import InputTaurus, InputAxial, InputTaurusPAV, InputTaurusMIX
from tools.data import DataTaurus, DataAxial, DataTaurusPAV, DataTaurusMIX
from tools.helpers import LINE_2, LINE_1, prettyPrintDictionary, \
    zipBUresults, readAntoine, OUTPUT_HEADER_SEPARATOR, LEBEDEV_GRID_POINTS,\
    ValenceSpacesDict_l_ge10_byM, TAURUS_EXECUTABLE_BY_FOLDER,\
    GITHUB_TAURUS_PAV_HTTP, TAURUS_SRC_FOLDERS, printf
from tools.Enums import OutputFileTypes

from tools.base_executors import _Base1DAxialExecutor, _Base1DTaurusExecutor, \
    ExecutionException
from random import random

#===========================================================================
#
#    EXECUTOR DEFINITIONS: MULTIPOLE DEFORMATIONS 
#
#===========================================================================

class ExeTaurus1D_DeformQ20(_Base1DTaurusExecutor):
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.EVEN_STEP_SWEEPING
    
    CONSTRAINT    = InputTaurus.ConstrEnum.b20
    CONSTRAINT_DT = DataTaurus.getDataVariable(InputTaurus.ConstrEnum.b20,
                                               beta_schm = 0)
    
    EXPORT_LIST_RESULTS = 'export_TESq20'
        
    def setUp(self, reset_folder=True):
        """
        set up: 
            * back up folder for results
            * dumping filename
            * save the hamiltonian files in BU folder for recovery
        """
        
        self._DDparams = self.inputObj._DD_PARAMS
        self.DTYPE.BU_folder = f'BU_folder_{self.interaction}_z{self.z}n{self.n}'
        if reset_folder:
            self.DTYPE.setUpFolderBackUp()
        
        for ext_ in OutputFileTypes.members():
            if os.path.exists(self.interaction+ext_):
                shutil.copy(self.interaction+ext_,  self.DTYPE.BU_folder)
        _=0
    
    def setUpExecution(self, reset_seed=False,  *args, **kwargs):
        """
        base solution pre-convergence.
        *   to change after the execution, put by InputTaurus.*Enum new values 
            as keyword arguments.
        
        :reset_seed: If there is/or not a seed minimum for the calculation and
                     we want to repeat the convergence process, turn it  <True>
        
        """
        
        self.calculationParameters
        
        res = None
        self._preconvergence_steps = 0
        self.printExecutionResult(None, print_head=True)
        
        _DO_ODD = 1 in self.numberParity and (not self.IGNORE_SEED_BLOCKING)
        if self._1stSeedMinima == None or reset_seed:                
            if self.GENERATE_RANDOM_SEEDS:
                if _DO_ODD:
                    ## procedure to evaluate odd-even nuclei
                    printf("  ** 1st seed minima_ odd, random convergences")
                    self._oddNumberParitySeedConvergence()
                    ## NOTE: The state blocked is in the seed, no more blocking during the process
                else:
                    ## Procedure to evaluate a set of random states (for even-even)
                    ## to get the lowest energy.
                    printf("  ** 1st seed minima_ even-even, random convergences")
                    self._evenNumberParitySeedConvergence()
            else:
                if not self.DO_BASE_CALCULATION:
                    ## Force convergence to point=0
                    self._deform_base = 0.0
                    setattr(self.inputObj, self.CONSTRAINT, 0.0)
                
                if _DO_ODD:
                    ## procedure to evaluate odd-even nuclei
                    printf("  ** 1st seed minima_ odd, NO convergences (constr_deform=0)")
                    self._oddNumberParitySeedConvergence()
                else:
                    ## even-even general case
                    printf("  ** 1st seed minima_ even-even, NO convergences (constr_deform=0)")
                    while not self._preconvergenceAccepted(res):
                        res = self._executeProgram(base_execution=True)
        
        self._final_bin_list_data[1][0] = self._1stSeedMinima._exported_filename
        
        ## negative values might be obtained        
        self._setDeformationFromMinimum(self._deform_lim_min, 
                                        self._deform_lim_max, self._N_steps)
        
        _new_input_args = dict(filter(lambda x: x[0] in self.ITYPE.ArgsEnum.members(), 
                                      kwargs.items() ))
        _new_input_cons = dict(filter(lambda x: x[0] in self.ITYPE.ConstrEnum.members(), 
                                      kwargs.items() ))
        self.inputObj.setParameters(**_new_input_args, **_new_input_cons)
        
        if ((self.ITYPE is InputTaurus) and 
            (InputTaurus.ArgsEnum.seed in _new_input_args)):
            self._base_seed_type = _new_input_args.get(InputTaurus.ArgsEnum.seed, 
                                                       None)
        
        self.printExecutionResult(None, print_head=True)
    
    def setUpProjection(self, **params):
        """
        Defines the parameters for the projection of the nucleus.
        The z, n, interaction, com, Fomenko-points, and Jvals from the program
        """
        if len(params) == 0: return
        self.inputObj_PAV = InputTaurusPAV(self.z, self.n, self.interaction)
        
        self.inputObj_PAV.com = self.inputObj.com
        self.inputObj_PAV.red_hamil = 1 # first computation will export it
        
        self.inputObj_PAV.z_Mphi = 9
        self.inputObj_PAV.n_Mphi = 9
        if self.inputObj.z_Mphi > 1: 
            self.inputObj_PAV.z_Mphi = self.inputObj.z_Mphi
        if self.inputObj.n_Mphi > 1: 
            self.inputObj_PAV.n_Mphi = self.inputObj.n_Mphi
        
        if (self.numberParity in [(0, 0), (1, 1)]):
            djmax = min(16, 2*self._sp_2jmax)
            djmin = 0
        else:
            djmax = 2 * self._sp_2jmax - 1
            djmin = self._sp_2jmin
        
        # if (InputTaurusPAV.ArgsEnum.j_max or InputTaurusPAV.ArgsEnum.j_min):
        self.inputObj_PAV.j_max = params.get(InputTaurusPAV.ArgsEnum.j_max, djmax)
        self.inputObj_PAV.j_min = params.get(InputTaurusPAV.ArgsEnum.j_min, djmin)            
            
        self.inputObj_PAV.setParameters(**params)
        self._list_PAV_outputs = []
        
        ## Create a folder in BU_ to store the projected results after a VAP-HFB
        if self.RUN_PROJECTION:
            if self.IGNORE_SEED_BLOCKING and 1 in self.numberParity:
                w = "\t[WARNING] Verify runProjection() before performing PAV over NON-Blocked wf!!!\n"*3
                printf(LINE_2, w, LINE_2)
            
            DataTaurusPAV.BU_folder = self.DTYPE.BU_folder + '/PNAMP'
            DataTaurusPAV.setUpFolderBackUp()
        
        printf("Will use the following PAV input optiones:")
        printf(self.inputObj_PAV)
        printf("EOF.\n\n")
    
    def runProjection(self, **params):
        """
        Base PAV-diagonal execution if VAP-HFB were executed: 
            1. Copy final_wf.bin-> left_wf.bin and right_wf.bin
            2. Run and save the PAV output.
            3. 
        """
        program_ = TAURUS_SRC_FOLDERS[GITHUB_TAURUS_PAV_HTTP]
        program_ = TAURUS_EXECUTABLE_BY_FOLDER[program_]
        
        if not (self._current_result and self._current_result.properly_finished):
            printf(" [WARNING] Previous result is invalid for PAV. Skipping")
            return
        if not os.path.exists(program_):
            printf(" [WARNING] PAV executable is missing. Skipping")
            return
            
        shutil.copy('final_wf.bin', 'left_wf.bin')
        shutil.copy('final_wf.bin', 'right_wf.bin')
        
        inp_fn = self.inputObj_PAV.DEFAULT_INPUT_FILENAME
        out_fn = DataTaurusPAV.DEFAULT_OUTPUT_FILENAME
        with open(inp_fn, 'w+') as f2:
                f2.write(self.inputObj_PAV.getText4file())
        
        if os.getcwd().startswith('C:'): ## Testing purpose 
            self._auxWindows_executeProgram_PAV(out_fn)
        else:
            os.system('./{} < {} > {}'.format(program_, inp_fn, out_fn))
        
        ## In case o
        try:
            res = DataTaurusPAV(self.z, self.n, out_fn)
            ## 
            if res.nanComponentsInResults:
                if not getattr(self, '_calling_co_PAV_exception', False):
                    self._calling_co_PAV_exception = True
                    printf("  [PAV error], changing cutoff-n.ovelap solve Nan components issue")
                    ies_0 = self.inputObj_PAV.empty_states
                    self.inputObj_PAV.cutoff_overlap = 1.e-9 # (change the value)
                    self.runProjection(**params)
                    self.inputObj_PAV.cutoff_overlap = 0 # (revert the change)
                    del self._calling_co_PAV_exception
                    return
                else:
                    printf("  [PAV error 3], cannot do anything. procced")
                    res = DataTaurusPAV(self.z, self.n, empty_data=True)
            
        except BaseException:
            if not getattr(self, '_calling_PAV_exception', False):
                self._calling_PAV_exception = True
                printf("  [PAV error], changing the empty states to solve the issue")
                ies_0 = self.inputObj_PAV.empty_states
                self.inputObj_PAV.empty_states = -ies_0 + 1 # (change the value)
                self.runProjection(**params)
                self.inputObj_PAV.empty_states = ies_0 # (revert the change)
                del self._calling_PAV_exception
                return
            else:
                printf("  [PAV error 2], cannot do anything. procced")
                res = DataTaurusPAV(self.z, self.n, empty_data=True)
        
        self._curr_PAV_result  = res
        ## move the projection results to another folder
        ## TODO. Exportable for printing as for K projection
        if self._curr_PAV_result.properly_finished:
            self.saveProjectedWFProcedure(res)
            self.projectionExecutionTearDown()
        else:
            printf(" [ERROR-PAV] PAV could not be achieved. Skipping")
    
    def saveProjectedWFProcedure(self, result: DataTaurusPAV):
        """
        Default copy of the function for the deformation into the PAV BU-folder
        """
        outfn = "d{}.OUT".format(self._curr_deform_index)
        outpth = "{}/{}".format(self._curr_PAV_result.BU_folder, outfn)
        
        if result.broken_execution or not result.properly_finished:
            outpth = "{}/broken_{}".format(self._curr_PAV_result.BU_folder, outfn)
            shutil.move(self.DTYPE.DEFAULT_OUTPUT_FILENAME, outpth)
            return
        
        if not outfn in self._list_PAV_outputs:
            self._list_PAV_outputs.append(outfn)
        
        shutil.move(DataTaurusPAV.DEFAULT_OUTPUT_FILENAME, outpth)        
    
    def _oddNumberParitySeedConvergence(self):
        """
        Procedure to select the sp state to block with the lowest energy:
         * Selects state randomly for a random sh-state in the odd-particle space
         * Repeat the convergence N times and get the lower energy
         * export to the BU the (discarded) blocked seeds (done in saveWF method)  
        """
        ## get a sp_space for the state to block 
        odd_p, odd_n = self.numberParity
        ## this was set in setUpExecution._getStatesAndDimensionsOfHamiltonian
        sh_states = self._sh_states
        sp_states = self._sp_states
        sp_dim    = self._sp_dim
        
        ## randomization of the blocked state and repeat the convergence
        ## several times to get the lower energy
        blocked_states  = []
        blocked_sh_states     = {}
        blocked_seeds_inputs  = {}
        blocked_seeds_results = {}
        blocked_energies      = {}
        bk_min, bk_E_min      = 0, 1.0e+69
        bu_results = {}
        double4OO = sum(self.numberParity) # repeat twice for odd-odd
        printf("  ** Blocking minimization process (random sp-st 2 block). MAX iter=", 
              double4OO*self.SEEDS_RANDOMIZATION, 
              " #-par:", self.numberParity, LINE_2)
        for rand_step in range(double4OO * self.SEEDS_RANDOMIZATION):
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
                printf(rand_step, f"  * Blocked state [{bk_sp}] is already calculated [SKIP]")
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
                if res != None:
                    ## otherwise the state is "re- blocked"
                    printf(" ** * [Not Converged] Repeating loop.")
                    self.inputObj.qp_block = None
                res = self._executeProgram(base_execution=True)
            printf(" ** * [OK] Result accepted. Saving result.")
            
            blocked_seeds_results[bk_sp] = deepcopy(res)
            blocked_energies     [bk_sp] = res.E_HFB
            # Move the base wf to a temporal file, then copied from the bk_min
            shutil.move(self._base_wf_filename, f"{bk_sp}_{self._base_wf_filename}")
            bu_results[(bk_sh_p, bk_sh_n)] = res
            
            ## actualize the minimum result
            if res.E_HFB < bk_E_min:
                bk_min, bk_E_min = bk_sp, res.E_HFB
        
            printf(rand_step, f"  * Blocked state [{bk_sp}] done, Ehfb={res.E_HFB:6.3f}")
            
            ## NOTE: If convergence is iterated, inputObj seed is turned 1, refresh!
            self.inputObj.seed = self._base_seed_type
        
        printf("\n  ** Blocking minimization process [FINISHED], Results:")
        printf(f"  [  sp-state]  [    shells    ]   [ E HFB ]  sp/sh_dim={sp_dim}, {len(sp_states)}")
        for bk_st in blocked_states:
            printf(f"  {str(bk_st):>12}  {str(blocked_sh_states[bk_st]):>16}   "
                  f"{blocked_energies[bk_st]:>9.4f}")
        printf("  ** importing the state(s)", bk_min, "with energy ", bk_E_min)
        printf(LINE_2)
        
        ## after the convegence, remove the blocked states and copy the 
        # copy the lowest energy solution and output.
        self.inputObj.qp_block = 0
        self._1stSeedMinima = blocked_seeds_results[bk_min]
        shutil.move(f"{bk_min}_{self._base_wf_filename}", self._base_wf_filename)
        self._exportBaseResultFile(bu_results)    
    
    def _evenNumberParitySeedConvergence(self):
        """
        Procedure to find the lowest energy for the even-even nuclei:
         * Taurus_ seeds start randomly (with its corresponding symmetry restrictions)
         * Repeat the convergence N times and get the lower energy
         * export to the BU the (discarded) blocked seeds (done in saveWF method)
        """
        ## randomization of the blocked state and repeat the convergence
        ## several times to get the lower energy
        converged_seeds = []
        seeds_results   = {}
        energies        = {}
        pairing_E       = {}
        beta_gamma      = {}
        bk_min, bk_E_min = 0, 1.0e+69
        bu_results      = {}
        
        for rand_step in range(self.SEEDS_RANDOMIZATION):
            bk_sp = rand_step
            
            seeds_results[bk_sp] = None
            energies     [bk_sp] = 4.20e+69
            
            self._preconvergence_steps = 0
            self._1stSeedMinima = None
            
            res = None
            while not self._preconvergenceAccepted(res):
                if res != None:
                    ## otherwise the state is "re- blocked"
                    printf(" ** * [Not Converged] Repeating loop.")
                res = self._executeProgram(base_execution=True)
            #printf(" ** * [OK] Result accepted. Saving result.")
            bu_results[bk_sp] = res
            converged_seeds.append(bk_sp)
            
            seeds_results[bk_sp] = deepcopy(res)
            energies     [bk_sp] = res.E_HFB
            pairing_E    [bk_sp] = res.pair
            beta_gamma   [bk_sp] = (res.beta_isoscalar, res.gamma_isoscalar)
            # Move the base wf to a temporal file, then copied from the bk_min
            shutil.move(self._base_wf_filename, f"{bk_sp}_{self._base_wf_filename}")
            
            ## actualize the minimum result
            if res.E_HFB < bk_E_min:
                bk_min, bk_E_min = bk_sp, res.E_HFB
        
            printf(f" ** * [OK] Seed [{bk_sp}] done, Ehfb={res.E_HFB:6.3f}")
            
            ## NOTE: If convergence is iterated, inputObj seed is turned 1, refresh!
            self.inputObj.seed = self._base_seed_type
        
        printf("\n  ** Blocking minimization process [FINISHED], Results:")
        printf(f"  [  ]     [ E HFB ] [ E pair] [beta, gamma]")
        for bk_st in converged_seeds:
            printf(f"   {bk_st:>2}      {energies[bk_st]:>9.4f} {pairing_E[bk_st]:>9.4f}"
                  f" ({beta_gamma[bk_st][0]:>4.3f}, {beta_gamma[bk_st][1]:>4.1f})")
        printf("  ** importing the state(s)", bk_min, "with energy ", bk_E_min)
        printf(LINE_2)
        
        ## after the convegence, remove the blocked states and copy the 
        # copy the lowest energy solution and output.
        self._1stSeedMinima = seeds_results[bk_min]
        shutil.move(f"{bk_min}_{self._base_wf_filename}", self._base_wf_filename)
        
        self._exportBaseResultFile(bu_results)
    
    def _exportBaseResultFile(self, bu_results):
        """ sExport in a results file the wf indexed by blocked states."""
        
        count = 0
        res : DataTaurus = None
        write_lines = []
        for indx, res in bu_results.items():
            if isinstance(indx, tuple):
                head = f'{indx[0]}_{indx[1]}'
            else:
                head = f'0_0'
            
            vals = res.getAttributesDictLike
            line = f'{count}: {head}' + OUTPUT_HEADER_SEPARATOR + vals
            write_lines.append(line)                
            count += 1
        
        with open('BASE-'+self.EXPORT_LIST_RESULTS, 'w+') as f:
            f.writelines('\n'.join(write_lines))
        printf(f" [DONE] Exported [{len(write_lines)}] convergence results in "
              f"[BASE-{self.EXPORT_LIST_RESULTS}]")
        
    def _preconvergenceAccepted(self, result: DataTaurus):
        """
        define the steps to accept the result.
            Modification: it there is a _1sSeedMinima and _preconvergence_steps=0
            skip (to run another constraint from a previous minimum)
        """
        
        if self._preconvergence_steps == 0 and self._1stSeedMinima != None:
            self.inputObj.seed = 1
            if   isinstance(self.inputObj, InputTaurus):
                shutil.copy(self._base_wf_filename, 'initial_wf.bin')
            elif  isinstance(self.inputObj, InputAxial):
                shutil.copy(self._base_wf_filename, 'fort.10')
            return True 
        
        self._preconvergence_steps += 1
        MAX_REPETITIONS = 4
        str_rep = f"[{self._preconvergence_steps} / {MAX_REPETITIONS}]"
        
        if self._preconvergence_steps >= MAX_REPETITIONS:
            ## iteration for preconvergence stops
            raise ExecutionException(f" !! {str_rep} Could not converge to the "
                                     "initial wf., execution process stops.")
        
        if result == None or result.broken_execution:
            printf(f" ** {str_rep} Seed convergence procedure:")
            return False
        else:
            ## there is no critical problem for the result, might be garbage or
            ## not enough iterations, the result is copied and changed the input
            ## to import it (if result is crazy the process will stop in if #1)
            self.inputObj.seed = 1
            if   isinstance(self.inputObj, InputTaurus):
                shutil.copy('final_wf.bin', 'initial_wf.bin')
            elif  isinstance(self.inputObj, InputAxial):
                shutil.copy('fort.11', 'fort.10')
            
            if result.properly_finished:
                pass
                #printf(f" ** {str_rep} Seed convergence procedure [DONE]:")
            else:
                printf(f" ** {str_rep} Seed convergence procedure [FAIL]: repeating ")
            return result.properly_finished
        
        return False # if not valid 2 repeat
    
    def saveFinalWFprocedure(self, result, base_execution=False):
        _Base1DTaurusExecutor.saveFinalWFprocedure(self, result, base_execution)
    
    def run(self):
        ## TODO: might require additional changes
        _Base1DTaurusExecutor.run(self)
    
    def globalTearDown(self, zip_bufolder=True, *args, **kwargs):
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
        printf("self._final_bin_list_data[0]=\n", self._final_bin_list_data[0])
        printf("self._final_bin_list_data[1]=\n", self._final_bin_list_data[1])
        for k in range(-len(self._final_bin_list_data[0]), 0, 1):
            tail = self._final_bin_list_data[0][k]
            constr_val = getattr(self._results[0][-k-1], self.CONSTRAINT_DT)
            constr_val = f"{constr_val:6.3f}".replace('-', '_')
            bins_.append("seed_{}.bin\t{}".format(tail, constr_val))
            outs_.append("res_{}.OUT\t{}".format(tail, constr_val))
        ## exportar prolate en orden
        for k in range(len(self._final_bin_list_data[1])):
            tail = self._final_bin_list_data[1][k]
            constr_val = getattr(self._results[1][k], self.CONSTRAINT_DT)
            constr_val = f"{constr_val:6.3f}".replace('-', '_')
            bins_.append("seed_{}.bin\t{}".format(tail, constr_val))
            outs_.append("res_{}.OUT\t{}".format(tail, constr_val))
        
        with open('list_dict.dat', 'w+') as f:
            f.write("\n".join(bins_))
        with open('list_outputs.dat', 'w+') as f:
            f.write("\n".join(outs_))
        shutil.copy('list_dict.dat', self.DTYPE.BU_folder)
        shutil.copy('list_outputs.dat', self.DTYPE.BU_folder)
        
        args = [self.z,self.n,self.interaction]+list(args)+list(kwargs.values())
        shutil.copy(self.EXPORT_LIST_RESULTS, DataTaurus.BU_folder)
        if zip_bufolder:
            if self.CONSTRAINT != None:
                args.append(self.CONSTRAINT)
            zipBUresults(DataTaurus.BU_folder, *args)
        
        ## Create a list of wf to do the VAP calculations:
        if self.DTYPE is DataTaurus:
            os.chdir(self.DTYPE.BU_folder)
            printf(f"\n  [globalTearDown] Saving the results in {os.getcwd()}/PNVAP", )
            os.mkdir('PNVAP')
            list_dat = []
            for i, bin_ in enumerate(bins_):
                fn, def_ = bin_.split()
                shutil.copy(fn, 'PNVAP/' + def_ + '.bin')
                fno, _ = outs_[i].split()
                printf(f"     cp: def_=[{def_}] fn[{fn}] fno[{fno}]")
                shutil.copy(fno, 'PNVAP/' + def_ + '.OUT')
                list_dat.append(def_ + '.bin')
            with open('list.dat', 'w+') as f:
                f.write("\n".join(list_dat))
            shutil.move('list.dat', 'PNVAP/')
            os.chdir('..')
        printf( "  [globalTearDown] Done.\n")
    
    def projectionExecutionTearDown(self):
        """
        Process to save result after calling runProjection()
        """
        ## save the list dat into folder
        with open('{}/list_pav.dat'.format(self._curr_PAV_result.BU_folder), 
                  'w+') as f:
            if len(self._list_PAV_outputs):
                sort_ = self._sortListDATForPAVresults(self._list_PAV_outputs)
                f.write("\n".join(sort_))
                
    def _sortListDATForPAVresults(self, list_):
        """
        Sorting procedure by deformations:
            >>> [ 'K1_d3.OUT', 'K1_d-1.OUT', 'K1_d-2.OUT']
            <<< ['K1_d-2.OUT', 'K1_d-1.OUT', 'K1_d3.OUT']
        """
        initial_ = [int(x.replace('.OUT', '').split('_d')[-1]) for x in list_]
        return [list_[initial_.index(i)] for i in sorted(initial_)]
        
        
        
class ExeAxial1D_DeformQ20(_Base1DAxialExecutor, ExeTaurus1D_DeformQ20):
    
    CONSTRAINT    = InputAxial.ConstrEnum.b20
    CONSTRAINT_DT = DataAxial .getDataVariable(InputAxial.ConstrEnum.b20,
                                               beta_schm = 0)
    EXPORT_LIST_RESULTS = 'exportAx_TESq20'
    
    EXPORT_LIST_RESULTS = 'export_resultAxial.txt'
    
    DTYPE = DataAxial  # DataType for the outputs to manage
    ITYPE = InputAxial # Input type for the input management

    def setUp(self, reset_folder=True):
        """
        set up: 
            * back up folder for results
            * dumping filename
            * save the hamiltonian files in BU folder for recovery
        """
        
        self.DTYPE.BU_folder = f'BU_folder_D1S_z{self.z}n{self.n}'
        if reset_folder:
            self.DTYPE.setUpFolderBackUp()
        # No Hamil files.
    
    def _getStatesAndDimensionsOfHamiltonian(self):
        """
        Read the hamiltonian and get the sp states/shell for the calculation
        """
        ## the hamiltonian is already copied in CWD for execution
        sh_states, l_ge_10 = [], True
        for m in range(self.MzMax +1):
            sh_states = sh_states + list(ValenceSpacesDict_l_ge10_byM[m])
        sh_states = [int(st) for st in sh_states]
        
        ## construct sp_dim for index randomization (sh_state, deg(j))
        sp_states = map(lambda x: (int(x), readAntoine(x, l_ge_10)[2] + 1), sh_states)
        sp_states = dict(list(sp_states))
        sp_dim    = sum(list(sp_states.values()))
        
        self._sh_states = sh_states
        self._sp_states = sp_states
        self._sp_dim    = sp_dim  

class ExeTaurus1D_DeformB20(ExeTaurus1D_DeformQ20):
    
    CONSTRAINT    = InputTaurus.ConstrEnum.b20
    CONSTRAINT_DT = DataTaurus.getDataVariable(InputTaurus.ConstrEnum.b20,
                                               beta_schm = 1)
    EXPORT_LIST_RESULTS = 'export_TESb20'
    
class ExeAxial1D_DeformB20(ExeAxial1D_DeformQ20):
    
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
    
class ExeAxial0D_EnergyMinimum(ExeAxial1D_DeformB20):
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.SINGLE_EVALUATION
    
    CONSTRAINT = None
    ## default value to see, 
    CONSTRAINT_DT = InputAxial.ConstrEnum.b20
    
    EXPORT_LIST_RESULTS = 'export_HOminimums_axial'
    
    """ Finish, just get _1sMinimum and export to a file"""


class ExeTaurus1D_FromBuckUpFiles(ExeTaurus1D_DeformB20):
    
    pass

#===============================================================================
# 
#    EXECUTOR - SINGLE STEP / PROJECTIONS AND GCM
#
#===============================================================================

class ExeTaurus0D_PAVProjection(_Base1DTaurusExecutor):
    
    DTYPE = DataTaurusPAV  # DataType for the outputs to manage
    ITYPE = InputTaurusPAV # Input type for the input management
    
    ITERATIVE_METHOD = _Base1DTaurusExecutor.IterativeEnum.SINGLE_EVALUATION
    
    CONSTRAINT    = None
    CONSTRAINT_DT = None
    
    EXPORT_LIST_RESULTS = 'export_PAV'
    
    def __init__(self, z, n, interaction, *args, **kwargs):
        
        ## interactions and nucleus
        self.z : int = z
        self.n : int = n
        self.interaction : str = interaction
        
        self.inputObj  : self.ITYPE  = None
        self._current_result: self.DTYPE  = None ## TODO: might be useless, remove in that case
        
        self.axialSymetryRequired = False ## set up to reject non-axial results
        self.sphericalSymmetryRequired = False
        self._output_filename = self.DTYPE.DEFAULT_OUTPUT_FILENAME
        
        
        self._curr_deform_index : int  = None
        
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
    
    def setUp(self, **params):
        pass
    
    def setUpExecution(self, reset_seed=False, *args, **kwargs):
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
        """ 
        Dummy method to test the scripts1d in Windows
        """
        # program = """ """
        # exec(program)
        FLD_TEST_ = 'data_resources/testing_files/'
        file2copy = FLD_TEST_+'TEMP_res_PAV_z2n1_odd.txt'
        
        txt = ''
        with open(file2copy, 'r') as f:
            txt = f.read()
            txt = txt.format(INPUT_2_FORMAT=self.inputObj_PAV)
        
        with open(output_fn, 'w+') as f:
            f.write(txt)
    
    def _executeProgram(self, base_execution=False):
        pass
    
    def printExecutionResult(self, result:DataTaurus, print_head=False, 
                             *params2print):
        pass
    
    def executionTearDown(self, result:DataTaurus, base_execution, *args, **kwargs):
        pass
    
    def saveFinalWFprocedure(self, result, base_execution=False):
        pass