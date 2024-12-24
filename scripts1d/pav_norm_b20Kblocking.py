'''
Created on 5 dic 2024

@author: delafuente
'''
from pathlib import Path
import os, shutil
from tools.data import DataTaurus, DataTaurusPAV
from copy import deepcopy
from tools.Enums import OutputFileTypes
from tools.helpers import importAndCompile_taurus, printf,\
    OUTPUT_HEADER_SEPARATOR
from tools.executors import ExeTaurus1D_DeformQ20
from tools.inputs import InputTaurusPAV

class EvaluatePAVNormOver1dByKforAllQuasiparticles():
    '''
    Over the results of a VAP calculations evaluated for every b20 deformation
    and for each quasiparicle value:
    
    BU_folder_B1_MZ4_z12n13/
        1_0_VAP/
            001/
                binaries(.bin) named by deformations: _0.606.bin, 1.253.bin, etc
                outputs vap (.out)
                export_TESb20_K1_z12n13_B1_MZ4.txt
            10001/
            ...
        3_0_VAP/
        ...
    '''
    INP_FN = 'aux_pav.INP'
    
    def __init__(self, z, n, interaction, valid_Ks=[], bu_folder=''):
        '''
        
        '''
        self.z = z
        self.n = n
        self.inter = interaction
        self.input_pav = InputTaurusPAV(z, n, interaction)
        self.input_pav.setParameters(**self.params_pav)
        
        self.bu_folder = Path(bu_folder)
        if bu_folder != '':
            assert self.bu_folder.exists(), f"ERROR, folder does not exist [{bu_folder}]"
        fld_ = f"BU_folder_{self.inter}_z{self.z}n{self.n}/"
        self.bu_folder = self.bu_folder / fld_
        assert self.bu_folder.exists(), f"ERROR, folder does not exist [{self.bu_folder}]"
        
        assert isinstance(valid_Ks, (list, tuple)), "Valid Ks is not list"
        assert all(map(lambda x: isinstance(x,int), valid_Ks)), \
            f"Ks must be integers, got[{valid_Ks}]"
        self.valid_Ks = valid_Ks
        
        ## process data from folder
        self._processData()
        
        ## evaluate in order the <b20(i)| b20(i+1)> for each particle and
        self._iteratePAVOverSolutions()
        
        ## export norm solutions
        self._exportNorms()
    
    @classmethod
    def setUpPAVparameters(cls, **params):
        cls.params_pav = params
    
    def _sort_deforms(self, list_files):
        if not isinstance(list_files, list): list_files = list(list_files)
        
        pos_files = filter(lambda x: x[0] != '_', list_files)
        neg_files = filter(lambda x: x[0] == '_', list_files)
        
        pos_files = [(x, float(x[:-4])) for x in pos_files]
        neg_files = [(x, float(x[:-4].replace('_', '-'))) for x in neg_files]
        
        pos_files = sorted(pos_files, key = lambda x: x[1])
        neg_files = sorted(neg_files, key = lambda x: x[1])
        
        return neg_files + pos_files
    
    def _processExportFileForTheDeformIndexes(self, K=None):
        
        if K:
            fn = f"{K}_0_VAP/export_TESb20_K{K}_z{self.z}n{self.n}_{self.inter}.txt"
            _k_list = [K, ]
        else:
            fn = f"export_TESb20_z{self.z}n{self.n}_{self.inter}.txt"
            _k_list = self.valid_Ks
        
        exp_file = self.bu_folder / fn
        with open(exp_file, 'r') as f:
            data = f.readlines()[1:]
            printf("Importing deforms for K=", K)
            for i, l in enumerate(data):
                args, _ = l.split(OUTPUT_HEADER_SEPARATOR)
                args = args.split(':')
                args[0] = int(args[0])
                args[1] = args[1].strip().replace('+','').replace('-','_')
                
                for K in _k_list:
                    self.deform_index_by_K[K][args[1]] = args[0]
        
                printf("  *",i+1, args)
        _ = 0
    
    def _clearanceOfBUstatesFolders(self):
        """
        This operation is required for all qp blocking separated in folders,
        the program will refresh the BU folders to be constructed from differtnt
        Energies depending in the blocked state.
        """
        
        bu_flds = os.listdir(self.bu_folder)
        bu_flds = filter(lambda x: os.path.isdir(self.bu_folder / x), bu_flds )
        bu_flds = filter(lambda x: x.startswith('BU_states_d'), bu_flds)
        printf("  Cleaning BU-states folders.")
        for fld_ in bu_flds:
            printf("  Cleaning BU-states folders: ", fld_)
            shutil.rmtree(self.bu_folder / fld_)
    
    def _processData(self):
        """ """
        self.b20_K_sorted = {}
        self.surf_b20 = dict()
        self.surfaces = dict()
        self.surf_b20_qp = dict()
        self.energies = dict()
        self.results  = dict()
        self.deform_index_by_K = dict([(K, dict()) for K in self.valid_Ks])
        
        ## clear BU_deform folders if exists
        self._clearanceOfBUstatesFolders()
        
        self._processExportFileForTheDeformIndexes()
        printf("")
        
        for K in self.valid_Ks:
            printf(" Getting data for K",K)
            self.surf_b20[K] = dict()
            self.surfaces[K] = dict()
            self.surf_b20_qp[K] = dict()
            self.energies[K] = dict()
            self.results [K] = dict()
            fld_k = self.bu_folder / f"{K}_0_VAP/"
            
            qp_folders = filter(lambda x: os.path.isdir(fld_k / x), os.listdir(fld_k))
            qp_folders = list  (filter(lambda x: x.isdigit(), qp_folders))
            qp_folders_sorted = dict([(int(qp), qp) for qp in qp_folders])
            
            for qp in sorted(qp_folders_sorted.keys()):
                files_b20 = os.listdir(fld_k / qp_folders_sorted[qp])
                files_b20 = filter(lambda x: x.endswith('.OUT'), files_b20)
                files_ded = filter(lambda x: 'broken' in x,  files_b20)
                files_b20 = filter(lambda x: not 'broken' in x,  files_b20)
                files_b20 = list(files_b20)
                
                b20_sorted = self._sort_deforms(files_b20)
                qp_fld     = fld_k / qp_folders_sorted[qp]
                
                printf(" ** Broken-uncompleted results in K,qp=",K, qp, list(files_ded))
                
                for k_b20, b20 in b20_sorted:
                    obj  = DataTaurus(self.z, self.n, qp_fld / f"{k_b20}")
                    Ehfb = obj.E_HFB
                    bin_k_b20 = k_b20.replace('.OUT', '.bin')
                    if not k_b20 in self.surf_b20[K]:
                        self.surf_b20[K][k_b20] = b20
                        self.surfaces[K][k_b20] = [qp_fld / f"{bin_k_b20}", ]
                        self.surf_b20_qp[K][k_b20] = [qp, ]
                        self.energies[K][k_b20] = [Ehfb, ]
                        self.results [K][k_b20] = [deepcopy(obj), ]
                    else:
                        assert not qp in self.surf_b20[K], f"Error, qp already [{qp}]"
                        
                        if self._EnergyAlreadyStored(K, k_b20, Ehfb): continue
                        
                        self.surfaces[K][k_b20].append(qp_fld / f"{bin_k_b20}")
                        self.surf_b20_qp[K][k_b20].append(qp)
                        self.energies[K][k_b20].append(Ehfb)
                        self.results [K][k_b20].append(deepcopy(obj))
            
            self.b20_K_sorted[K] = self._sort_deforms(self.energies[K].keys())
        
        if os.getcwd().startswith('C'): self._createADummyExportfile() 
    
    def _createADummyExportfile(self):
        _ = 0
        keys_ = []
        for K in self.valid_Ks: keys_ += list(self.results[K].keys())
        keys_ = list(set(keys_))
        keys_ = self._sort_deforms(keys_)
        
        K0 = 1
        printf(" [TEST] Creating export file, K0=", K0)
        
        energies, b20list = [], []
        for i, kb20 in enumerate(keys_):
            if not kb20[0] in self.results[K0]: 
                printf("  ** not found index,deform=", i, kb20)
                continue
            res : DataTaurus = self.results[K0][kb20[0]][0]            
            energies.append(res.E_HFB)
            b20list.append(res.b20_isoscalar)
        
        Emin = min(energies)
        k0   = energies.index(Emin)
        
        for K in self.valid_Ks:
            self.deform_index_by_K[K] = {}
            for i, b20 in enumerate(b20list):
                b20s = f"{b20:5.3f}".replace('-', '_')
                if not b20s+'.OUT' in self.results[K]: continue
                self.deform_index_by_K[K][b20s] = i - k0
        
    
    def _EnergyAlreadyStored(self, K, k_b20, Ehfb):
        """ skip the quasiparticle blocked if the energy is already found.  """
        return all(map(lambda x: abs(x-Ehfb) < 1.0e-6, self.energies[K][k_b20]))
    
    def _emptyStatesSetUpByPairingSituation(self, K, k_b20, i):
        """
        For Mean-field results, the absence of pairing in proton or neutron part
        lead to some problems with the pfaffian-norm evaluation (getting N>=1.0).
        The problem should be considered with option empty_states=1 only in that
        cases, 0 by default.
        """
        obj : DataTaurus = self.results[K][k_b20][i]
        empty_states = (obj.var_p < 1.0e-7) or (obj.var_n < 1.0e-7 )
        self.input_pav.empty_states = int(empty_states)
    
    def _execute_PAV_norm(self, output_fn):
        """
        Problems related to the EmptyStates Switching, in cases where ES=1, even
        with zero-pairing. Revert the option if there 0-norm PAV exception.
        """
        for i in range(3):
            ## switch cutoff -> 1e-6 if IE=1, switch IE if tried
            if i == 0: 
                self.input_pav.cutoff_overlap = 0 ## reset cutoff norm
            if i == 1:
                if self.input_pav.empty_states == 1: 
                    self.input_pav.cutoff_overlap = 1.0e-6
                else: 
                    self.input_pav.empty_states   = 1
                    self.input_pav.cutoff_overlap = 0
            else: 
                if self.input_pav.empty_states == 1:
                    self.input_pav.empty_states   = 0
                    self.input_pav.cutoff_overlap = 0
                else: 
                    self.input_pav.cutoff_overlap = 1.0e-6
            ## run
            try:
                with open(self.INP_FN, 'w+') as f: 
                    f.write(self.input_pav.getText4file())
                    # run PAV
                if os.getcwd().startswith('C:'): ## Testing purpose 
                    _auxWindows_executeProgram_PAV(output_fn)
                else:
                    os.system('./taurus_pav.exe < {} > {}'.format(self.INP_FN, output_fn))
                
                obj = DataTaurusPAV(self.z, self.n, output_fn)
                norm_i = obj.proj_norm[0]
                if norm_i > 1: continue
                
                return obj
            except BaseException as e:
                if i == 1: raise e
                printf(f"       ** failed PAV projection, switching EmptyStates")
    
    def _iteratePAVOverSolutions(self):
        """ Execute for the different b20 - K, """
        ## copy the hamiltonian files
        exe_fld = Path('TEMP_BU') if os.getcwd().startswith('C') else Path()
        fld_ = self.bu_folder
        for tail_ in OutputFileTypes.members():
            if not os.path.exists(fld_ / f'{self.inter}{tail_}'): continue
            shutil.copy(fld_ / f'{self.inter}{tail_}', exe_fld)
        if exe_fld != Path(): shutil.copy('taurus_pav.exe', exe_fld)
                
        self.norms = dict()
        self.pav_results = dict()
        for K in self.valid_Ks:
            b20_ref   = "{:5.3f}".format(self.b20_K_sorted[K][0][1])
            k_b20_ref = self.b20_K_sorted[K][0][0]
            shutil.copy(self.surfaces[K][k_b20_ref][0], exe_fld / 'left_wf.bin')
            
            ## clear PAV saving folders,
            bu_k_fld = self.bu_folder / f'{K}_0_VAP/norm_continuity'
            if os.path.exists(bu_k_fld):
                for x in os.listdir(bu_k_fld): os.remove(bu_k_fld / x)
            else:
                os.mkdir(bu_k_fld)
            
            self.norms[K] = dict()
            self.pav_results[K] = dict()
            printf(f"  Running K={K}")
            printf("       >> new-ref state:", self.surfaces[K][k_b20_ref][0])
            for k_b20, b20 in self.b20_K_sorted[K][1:]:
                # copy contiguous wf.
                self.norms[K][k_b20] = []
                self.pav_results[K][k_b20] = []
                _create_BU_fld = len(self.surfaces[K][k_b20]) > 1
                
                if _create_BU_fld: 
                    bu_sts = self._createBUstatesForMultiMinima(K, k_b20)
                
                _lenqp = len(self.surf_b20_qp[K][k_b20])
                for i, kb20i_pth in enumerate(self.surfaces[K][k_b20]):
                    
                    self._emptyStatesSetUpByPairingSituation(K, k_b20, i)
                    shutil.copy(kb20i_pth, exe_fld / 'right_wf.bin')
                    
                    printf(f"    <{b20_ref} | {b20:5.3f}> : ", kb20i_pth)
                    
                    os.chdir(exe_fld)
                    args = K, self.surf_b20_qp[K][k_b20][i], b20_ref, k_b20.replace("_","-")
                    out_fn = 'overlap_K{}_qp{}_{}_{}'.format(*args)
                    
                    try:
                        obj : DataTaurusPAV = self._execute_PAV_norm(out_fn)
                        
                        norm_i = obj.proj_norm[0]
                        self.norms[K][k_b20].append(norm_i)
                        self.pav_results[K][k_b20].append(deepcopy(obj))
                        
                        printf(f"      = {norm_i:5.3f} [OK] i:[{i}/{_lenqp}] "
                               f"EmptySt={self.input_pav.empty_states}" )
                        
                    except BaseException as e:
                        printf(f"      [Error] PAV-HFB not obtained i:[{i}/{_lenqp}] "
                               f"EmptySt={self.input_pav.empty_states}")
                        self.norms[K][k_b20].append(0.0)
                        self.pav_results[K][k_b20].append(None)
                        
                        shutil.move(out_fn, f"broken-{out_fn}")
                        out_fn = f"broken-{out_fn}"
                        
                    if os.getcwd().startswith('C') : os.chdir('..')
                    ## save PAV files
                    if _create_BU_fld:
                        self._moveBUstatesForMultiMinima(exe_fld/out_fn, bu_sts)
                    shutil.move(exe_fld / out_fn, bu_k_fld)
                    
                # if several compare and stablish the next b20_ref as left_wf
                _ = 0
                Nmin = min(self.norms[K][k_b20])
                i = self.norms[K][k_b20].index(Nmin)
                
                b20_ref = b20
                if _lenqp > 1: printf("       >> new-ref state:", self.surfaces[K][k_b20][i])
                shutil.copy(self.surfaces[K][k_b20][i], exe_fld / 'left_wf.bin')    
    
    def _createBUstatesForMultiMinima(self, K, k_b20):
        """ Create in main BU folder BU-states folder with tuple-qp files. """
        _=0
        def_ = self.deform_index_by_K[K][k_b20[:-4]]
        bu_sts_fld = self.bu_folder / f"BU_states_d{def_}K{K}"
        os.mkdir(bu_sts_fld)
        
        for i, pth_ in enumerate(self.surfaces[K][k_b20]):
            qp     = self.surf_b20_qp[K][k_b20][i]
            parent = pth_.parent
            head_ = f'sp{qp}_d{def_}K{K}'
            
            shutil.copy(pth_, bu_sts_fld / f"{head_}.bin")
            shutil.copy(parent / k_b20, bu_sts_fld / f"{head_}.OUT")
            
            for dat_ in ('eigenbasis_h', 'eigenbasis_H11', 'occupation_numbers', 'canonicalbasis'):
                f = f"{dat_}_{k_b20}".replace('OUT', 'dat')
                if not f in os.listdir(parent): continue
                shutil.copy(parent / f, bu_sts_fld / f"{dat_}_{head_}.dat")
        
        return bu_sts_fld
    
    def _moveBUstatesForMultiMinima(self, src_fn, bu_sts):
        """ src_fn = exe_fld/out_fn """
        shutil.copy(src_fn, bu_sts)
    
    def _exportNorms(self):
        
        for K in self.valid_Ks:
            
            lines = []
            for b20_k, b20 in self.b20_K_sorted[K][1:]:
                indx = self.deform_index_by_K[K][b20_k[:-4]]
                
                line = f'{indx}: {b20:5.3}{OUTPUT_HEADER_SEPARATOR}'
                vals = []
                for i in range(len(self.energies[K][b20_k])):
                    qp    = self.surf_b20_qp[K][b20_k][i]
                    ener  = self.energies[K][b20_k][i]
                    norm  = self.norms   [K][b20_k][i]
                    vals.append(f'{qp} : {ener:8.4f} : {norm:5.3f}')
                vals = ','.join(vals)
                lines.append(line + vals)
            
            bu_k_fld = self.bu_folder / f'{K}_0_VAP/norm_continuity'
            with open(bu_k_fld / 'norm_overlaps.txt', 'w+') as f:
                f.write('\n'.join(lines))
            printf(f" ! Exported data K={K} in {bu_k_fld / 'norm_overlaps.txt'}")

class EvaluatePAVNormOver1dByKforStandardCalculation(EvaluatePAVNormOver1dByKforAllQuasiparticles):
    
    """
    Same as EvaluatePAVNormOver1dByKforAllQuasiparticles but for a usual multiK 
    evaluation:
        Not all the QP have to be blocked independently, and the QP global folder
        does not exists
        
    * The change in storaging folders goes in BU_states_d[defindex]K[K] that identify
        the quasiparticle,
    * All other valid converging states goes to the same minimum stored as usual (deform.*extension)
    
    BU_folder_B1_MZ4_z12n13/
        1_0_VAP/
            0.125.bin 0.125.OUT canonicalbasis_0.125.dat ...
            0.333.bin 0.333.bin canonicalbasis_0.333.dat ...
            export_TESb20_K1_z15n14_B1_MZ4.txt
        BU_states_d-5K1/
            repeated files with prefix: (*)sp[qp]_d[def_index]K[K].**
            (*) = eigenbasis_h_, canonicalbasis_, occupation_numbers_ ...
            **  = .OUT, .bin, .dat
        
    """
    
    def _processData(self):
        """ """
        self.b20_K_sorted = {}
        self.surf_b20 = dict()
        self.surfaces = dict()
        self.surf_b20_qp = dict()
        self.energies = dict()
        self.results  = dict()
        self.deform_index_by_K = dict([(K, dict()) for K in self.valid_Ks])
        
        set_EN_tuplets = filter(lambda x: x.startswith('BU_states_'), 
                                os.listdir(self.bu_folder))
        set_EN_tuplets = list(set_EN_tuplets)
        
        for K in self.valid_Ks:
            printf(" Getting data for K",K)
            self.surf_b20[K] = dict()
            self.surfaces[K] = dict()
            self.surf_b20_qp[K] = dict()
            self.energies[K] = dict()
            self.results [K] = dict()
            fld_k = self.bu_folder / f"{K}_0_VAP/"
            
            self._processExportFileForTheDeformIndexes(K)
            list_out = map(lambda x: x+'.OUT', self.deform_index_by_K[K].keys())
            self.b20_K_sorted[K] = self._sort_deforms(list_out)
            ## read the converged-selected values, as list
            
            files_ded = []
            for k_b20, idx in self.deform_index_by_K[K].items():
                k_b20 = f"{k_b20}.OUT"
                if not os.path.exists(fld_k / k_b20):
                    if os.path.exists(fld_k / f"broken_{k_b20}"):
                        files_ded.append(f"broken_{k_b20}")
                    printf(f" [WARN] Deformation [{k_b20}] not found or broken. SKIPPING")
                    continue
                obj  = DataTaurus(self.z, self.n, fld_k / f"{k_b20}")
                Ehfb = obj.E_HFB
                bin_k_b20 = k_b20.replace('.OUT', '.bin')
                
                if not k_b20 in self.surf_b20[K]:
                    self.surf_b20[K][k_b20] = self.b20_K_sorted[K][1]
                    self.surfaces[K][k_b20] = [fld_k / f"{bin_k_b20}", ]
                    self.surf_b20_qp[K][k_b20] = [0, ]
                    self.energies[K][k_b20] = [Ehfb, ]
                    self.results [K][k_b20] = [deepcopy(obj), ]
            
            printf(" ** Broken-uncompleted results in K,qp=",K, list(files_ded))
            ## Overwrite the with the BU states and the index of qp.
            for bu_sts in set_EN_tuplets:
                idx, K2 = bu_sts.replace('BU_states_d','').split('K')
                idx, K2 = int(idx), int(K2)
                if K2 != K: continue
                
                bu_sts_p = self.bu_folder / bu_sts
                
                files_ = filter(lambda x: x.endswith('.OUT'),os.listdir(bu_sts_p))
                files_ = list  (files_)
                
                for k_b20, idx2 in self.deform_index_by_K[K].items():
                    if idx2 == idx: break
                k_b20 = f"{k_b20}.OUT"
                
                self.surfaces[K][k_b20] = []
                self.surf_b20_qp[K][k_b20] = []
                self.energies[K][k_b20] = []
                self.results [K][k_b20] = []
                
                for f in files_:
                    qp = int(f.replace('sp', '').split('_') [0])
                    
                    obj  = DataTaurus(self.z, self.n, bu_sts_p / f)
                    Ehfb = obj.E_HFB
                    bin_k_b20 = f.replace('.OUT', '.bin')
                    
                    self.surfaces[K][k_b20]   .append(bu_sts_p / f"{bin_k_b20}")
                    self.surf_b20_qp[K][k_b20].append(qp)
                    self.energies[K][k_b20].append(Ehfb)
                    self.results [K][k_b20].append(deepcopy(obj))
    
    def _iteratePAVOverSolutions(self):
        """ The same procedure, just do not copy into BU-states folders """
        EvaluatePAVNormOver1dByKforAllQuasiparticles._iteratePAVOverSolutions(self)
    
    def _createBUstatesForMultiMinima(self, K, k_b20):
        return 
    
    def _moveBUstatesForMultiMinima(self, src_fn, bu_sts):
        """ src_fn = exe_fld/out_fn """
        return
    
def _auxWindows_executeProgram_PAV(output_fn):
        """ 
        Dummy method to test the scripts1d - PAV in Windows
        """
        
        FLD_TEST_ = '../data_resources/testing_files/'
        file2copy = FLD_TEST_+'TEMP_res_PAV_z2n1_odd.txt'
        file2copy = FLD_TEST_+'TEMP_res_PAV_z8n9_1result.txt'
        # file2copy = FLD_TEST_+'TEMP_res_PAV_z12n19_nan_norm_components.txt'
        
        txt = ''
        with open(file2copy, 'r') as f:
            txt = f.read()        
        with open(output_fn, 'w+') as f:
            f.write(txt)

def run_b20_calculatePAVnormForKindependently(nuclei, valid_Ks=[]):
    """
    Over results evaluated for many blocked states (independently from a False OE)
    calculate the norm between the contiguos deformations - qp.
    
    # Notes:
        1. sorting order from oblate to prolate (1st oblate excluded)
        2. In case of tuplet, the reference state for the next step is the one with 
        larger norm in the tuplet.
    """
    
    #os.chdir('../') # this script is not in the main folder
    
    importAndCompile_taurus(use_dens_taurus=False, pav=True,
                            force_compilation=not os.path.exists('taurus_pav.exe'))
    
    input_args_projection = {
        InputTaurusPAV.ArgsEnum.red_hamil : 0,
        InputTaurusPAV.ArgsEnum.com   : 1,
        InputTaurusPAV.ArgsEnum.z_Mphi: 1,
        InputTaurusPAV.ArgsEnum.n_Mphi: 1,
        InputTaurusPAV.ArgsEnum.disable_simplifications_NZA: 1,
        # InputTaurusPAV.ArgsEnum.alpha : 0,
        # InputTaurusPAV.ArgsEnum.beta  : 0,
        # InputTaurusPAV.ArgsEnum.gamma : 0,
        InputTaurusPAV.ArgsEnum.disable_simplifications_JMK: 1,
        InputTaurusPAV.ArgsEnum.disable_simplifications_P : 1,
        InputTaurusPAV.ArgsEnum.empty_states : 0,
        InputTaurusPAV.ArgsEnum.cutoff_overlap : 1.0e-10,
        # PN-PAV and J bound arguments set by the program, P-PAV = no
    }
    if os.getcwd().startswith('C'):
        MAIN_FLD = 'DATA_RESULTS/SD_Kblocking_multiK/P/' 
    else: 
        MAIN_FLD = ''
    
    for zn, inter in nuclei.items():
        EvaluatePAVNormOver1dByKforAllQuasiparticles.setUpPAVparameters(**input_args_projection)
        EvaluatePAVNormOver1dByKforAllQuasiparticles(*zn, inter, valid_Ks, MAIN_FLD)
        
        printf(" Finished the PAV norm evaluation z,n=", zn,"!")

def run_b20_calculatePAVnormForStandardRunKBlocking(nuclei, valid_Ks=[]):
    """
    Over results evaluated for blocked states considering the tuples from a
    direct evaluation (from a False OE) from the folders BU_states
    calculate the norm between the contiguos deformations - qp.
    
    # Notes:
        1. sorting order from oblate to prolate (1st oblate excluded)
        2. In case of tuplet, the reference state for the next step is the one with 
        larger norm in the tuplet.
    """
    
    #os.chdir('../') # this script is not in the main folder
    
    importAndCompile_taurus(use_dens_taurus=False, pav=True,
                            force_compilation=not os.path.exists('taurus_pav.exe'))
    
    input_args_projection = {
        InputTaurusPAV.ArgsEnum.red_hamil : 0,
        InputTaurusPAV.ArgsEnum.com   : 1,
        InputTaurusPAV.ArgsEnum.z_Mphi: 1,
        InputTaurusPAV.ArgsEnum.n_Mphi: 1,
        InputTaurusPAV.ArgsEnum.disable_simplifications_NZA: 1,
        # InputTaurusPAV.ArgsEnum.alpha : 0,
        # InputTaurusPAV.ArgsEnum.beta  : 0,
        # InputTaurusPAV.ArgsEnum.gamma : 0,
        InputTaurusPAV.ArgsEnum.disable_simplifications_JMK: 1,
        InputTaurusPAV.ArgsEnum.disable_simplifications_P : 1,
        InputTaurusPAV.ArgsEnum.empty_states : 0,
        InputTaurusPAV.ArgsEnum.cutoff_overlap : 1.0e-10,
        # PN-PAV and J bound arguments set by the program, P-PAV = no
    }
    if os.getcwd().startswith('C'):
        MAIN_FLD = '' #DATA_RESULTS/SD_Kblocking_multiK/P/' 
    else: 
        MAIN_FLD = ''
    
    for zn, inter in nuclei.items():
        EvaluatePAVNormOver1dByKforStandardCalculation.setUpPAVparameters(**input_args_projection)
        EvaluatePAVNormOver1dByKforStandardCalculation(*zn, inter, valid_Ks, MAIN_FLD)
        
        printf(" Finished the PAV norm evaluation z,n=", zn,"!")
    