'''
Created on 19 mar 2024

@author: delafuente

'''

from copy import deepcopy
import shutil
import numpy as np
import os

from tools.helpers import almostEqual, LINE_1
from tools.data import DataTaurus
from .executors import ExeTaurus1D_DeformB20



class ExeTaurus1D_B20_OEblocking_Ksurfaces(ExeTaurus1D_DeformB20):
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
    IGNORE_SEED_BLOCKING = True

    
    def __init__(self, z, n, interaction, *args, **kwargs):
        
        ExeTaurus1D_DeformB20.__init__(self, z, n, interaction, *args, **kwargs)
        
        self._valid_Ks : list = []
        self._K_results : dict = {}   # not used
        self._K_seed_list : dict = {} # not used
        
        ## Optimization to skip already found sp for a previous K;
        ##   NOTE: for that deformation, we already know that sp will end into 
        ##         the previous K value
        self._sp_blocked_K_already_found : dict = {} # {b20_index: sp_found, ...}
        
        self._blocking_section = False
    
    def setInputCalculationArguments(self, core_calc=False, axial_calc=False, 
                                           spherical_calc=False, **input_kwargs):
        
        ExeTaurus1D_DeformB20.setInputCalculationArguments(self, 
                                                           core_calc=core_calc, 
                                                           axial_calc=axial_calc, 
                                                           spherical_calc=spherical_calc, 
                                                           **input_kwargs)
    
    def setUpExecution(self, reset_seed=False, *args, **kwargs):
        """
        Set up of the main false OE TES and the arrays for the blocking part.
        """
        ExeTaurus1D_DeformB20.setUpExecution(self, reset_seed=reset_seed, *args, **kwargs)
        
        ## organization only sorted: in the range of valid j
        # self._valid_Ks = [k for k in range(-self._sp_2jmax, self._sp_2jmax+1, 2)]
        # # skip invalid K for the basis, i.e. _sp_2jmin=3/2 -> ..., 5,3,-3,-5 ...
        # self._valid_Ks = list(filter(lambda x: abs(x) >= self._sp_2jmin,
        #                              self._valid_Ks))
        
        ## optimal organization of the K: 1, -1, 3, -3, ...
        for k in range(self._sp_2jmin, self._sp_2jmax +1, 2):
            self._valid_Ks.append(k)
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
        
    
    def run(self):
        """
        Modifyed method to obtain the reminization with a blocked state.
        """
        self._blocking_section = False
        ExeTaurus1D_DeformB20.run(self)
        
        ##  
        self._blocking_section = True
        if self.numberParityOfIsotope == (0, 0): return
        print(LINE_1, " [DONE] False Odd-Even TES, begin blocking section")
        
        def __get_mz_str(sp_):
            i_ = 0
            for sh_, deg in self._sp_states.items():
                jz = deg - 1                        # jz = int(f"{sh_:03}"[2:])
                for mj in range(-jz, jz +1, 2):
                    i_ += 1
                    if i_ == sp_: return f"{sh_:03}({mj:+2})"
        
        # Perform the projections to save each K component
        BU_FLD = self.DTYPE.BU_folder
        no_results_for_K = True
        for K in self._valid_Ks:
            self._current_K = K
            # TODO: Could be done for the variable step
            
            # Refresh and create folders for vap-blocked results
            BU_FLD_KBLOCK = f"{BU_FLD}/{K}_VAP"
            BU_FLD_KBLOCK = BU_FLD_KBLOCK.replace('-', '_')
            # Create new BU folder
            if os.path.exists(BU_FLD_KBLOCK):
                shutil.rmtree(BU_FLD_KBLOCK)
            os.mkdir(BU_FLD_KBLOCK)
            self._exportable_BU_FLD_KBLOCK = BU_FLD_KBLOCK 
            self._exportable_LISTDAT = []
            print(f"* Doing 2K={K} for TES results. saving in [{BU_FLD_KBLOCK}]")
            
            self._exportable_txt = {}
            # oblate part
            for prolate in (0, 1):
                for i, tail_ in self._final_bin_list_data[prolate].items():
                    
                    self._curr_deform_index = i
                    
                    shutil.copy(f"{BU_FLD}/seed_{tail_}.bin", "initial_wf.bin")
                    self.inputObj.seed = 1
                    ## fijar la deformacion i en la coordenada a constrain
                    if prolate:
                        b20_ = self.deform_prolate[i] 
                    else: 
                        b20_ = self.deform_oblate[i]
                    setattr(self.inputObj, self.CONSTRAINT, b20_)
                    
                    in_neu = self._sp_dim if self.numberParityOfIsotope[1] else 0
                    ## block the states in order
                    for sp_ in range(1, self._sp_dim +1):
                        ## OPTIMIZATION: if state has a previous K skip
                        K_prev = self._sp_blocked_K_already_found[i][sp_]
                        if (K_prev != 0) and (K_prev != K): continue
                        
                        self.inputObj.qp_block = sp_ + in_neu
                        
                        ## minimize and save only if 2<Jz> = K
                        res : DataTaurus = self._executeProgram()
                        if not (res.properly_finished and res.isAxial()): 
                            continue
                        if almostEqual(2 * res.Jz, K, 1.0e-5):
                            print("   [OK] {} {} <jz>= {:3.1f}, b20={:>6.3f}  E_hfb={:6.3f}"
                                  .format(sp_, __get_mz_str(sp_),  
                                          res.Jz, res.b20_isoscalar, res.E_HFB))
                            
                            ## Append the exportable result file
                            line = []
                            if self.include_header_in_results_file:
                                line.append(f"{i:5}: {b20_:+6.3f}")
                            line.append(res.getAttributesDictLike)
                            self._exportable_txt[i] = self.HEADER_SEPARATOR.join(line)
                            
                            self._sp_blocked_K_already_found[i][sp_] = K
                            
                            no_results_for_K *= False
                            ## no more minimizations for this deformation
                            break
                        elif sp_ == self._sp_dim:
                            print(f"  [no K={K}] no state for def[{i}]={b20_:>6.3f}")
            if no_results_for_K: 
                print("  [WARNING] No blocked result for 2K=", K)
            # K-loop
        _ = 0
    
    def saveFinalWFprocedure(self, result:DataTaurus, base_execution=False):
        """
        overwriting of exporting procedure:
            naming, dat files, output in the BU folder
        """
        if self._blocking_section:
            # copy the final wave function and output with the deformation
            # naming (_0.365.OUT, 0.023.bin, etc )
            #
            # NOTE: if the function is not valid skip or ignore
            i = self._curr_deform_index
            b20_ = self.deform_oblate[i] if (i < 0) else self.deform_prolate[i]
            fndat = f"{b20_:6.3f}".replace('-', '_').strip()
            fnbin = f"{fndat}.bin"
            
            _invalid = result.broken_execution or not result.properly_finished
            if _invalid:
                shutil.move(self.DTYPE.DEFAULT_OUTPUT_FILENAME, 
                            f"{self._exportable_BU_FLD_KBLOCK}/broken_{fndat}.bin")
                return
            
            shutil.move("final_wf.bin", 
                        f"{self._exportable_BU_FLD_KBLOCK}/{fnbin}")
            shutil.move(self.DTYPE.DEFAULT_OUTPUT_FILENAME, 
                        f"{self._exportable_BU_FLD_KBLOCK}/{fndat}.OUT")
            
            if not fnbin in self._exportable_LISTDAT:
                # if i < 0:
                #     self._exportable_LISTDAT.insert(0, fnbin)
                # else:
                self._exportable_LISTDAT.append(fnbin)
            
        else:
            ## Normal execution
            ExeTaurus1D_DeformB20.saveFinalWFprocedure(self, 
                                                       result, base_execution)
    
    def executionTearDown(self, result:DataTaurus, base_execution, *args, **kwargs):
        """
            Separate the two parts for obtaining the 'exportResult()' for 
            minimization after blocking.
        """
        if self._blocking_section:
            ## save the list dat into folder
            with open(f'{self._exportable_BU_FLD_KBLOCK}/list.dat', 'w+') as f:
                if len(self._exportable_LISTDAT):
                    f.write("\n".join(self._exportable_LISTDAT) )
            
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
    
    def gobalTearDown(self, zip_bufolder=True, *args, **kwargs):
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
        
        ## main 
        ExeTaurus1D_DeformB20.gobalTearDown(self, zip_bufolder=zip_bufolder, 
                                                  *args, **kwargs)
        