'''
Created on 11 dic 2024

@author: delafuente
'''
import os, shutil
from pathlib import Path
from tools.hamiltonianMaker import TBME_HamiltonianManager, TBMEXML_Setter
from tools.helpers import printf, LINE_2, OUTPUT_HEADER_SEPARATOR
from tools.Enums import OutputFileTypes
from tools.inputs import InputTaurus, InputTaurusPAV
from tools.data import DataTaurus, DataTaurusPAV
from copy import deepcopy

class ExeTaurus1D_AfterRun_HamilDecomposition(object):
    '''
    After Run evaluation (from some ExeTaurus1D previous execution).
    
    From PNVAP folder in an stored BU_folder_{interaction}_z{z}n{n}/
    
    evaluates for every solution different hamiltonians with a void step 
    (only iters=0, the other option grad_type=0, eta=0, mu=1e-50, iters=1 rotates
    somehow the wavefunction so it is not valid) 
        -> TODO: the files for iter=0 does not have the header for the 
                iterative minimization, some error could arrise.
    
    Every binary is copied, reexecuted for every part defined, the outputs 
    are moved to HAMIL_PART_{identification} with also an export_file for better 
    processing.    
    
    Can evaluate the recombination of all forces to get a bech hamiltonian to 
    test the results in PNVAP (or comparing directly the matrix elements).
    
    Example:
    
        BU_folder_{interaction}_z{z}n{n}/
            export_TESb20_z15n14_{interaction}.txt
            PNVAP/
            HAMIL_PART_Gaussian1Majorana/
                *.out,
                export_TESb20_z15n14_Gaussian1Majorana.txt
                Gaussian1Majorana.sho (2b, com, etc)
            HAMIL_PART_Gaussian2Majorana/
                *.out,
                export_TESb20_z15n14_Gaussian2Majorana.txt
                Gaussian2Majorana.sho (2b, com, etc)
            ...
            HAMIL_BENCH/
                *.out,
                export_TESb20_z15n14_bench.txt
                bench.sho (2b, com, etc)
    
    Notes: 
    1.  This scripts require full comprehension of the initial Hamiltonian from
        the first calculation, such as oscilator length, parts and parameters
        for those parts.
    2.  The program does not asume blocking, therefore it interprets PNVAP 
        solutions as proper HFB vacum in odd nuclei. To extend this consideration
        Extend this class for a certain K-P folder (K_P_VAP)
    3.  Write the names of the interactions in camel form.
    4.  The executor relies in TBME_HamiltonianManager and TBMEXML_Setter,
        verify if changes are required for "exotic" interactions non accounted.
    
    '''
    HAMIL_NAME   = 'hamil'
    VAP_INP_ARGS = {}
    PAV_INP_ARGS = {}
    DO_PROJECTION= False
    IGNORE_FOMENKO_POINTS_FROM_BU_SOLUTIONS = False

    def __init__(self, z, n, interaction_params, bu_folder=''):
        '''
        interaction_params: (Mzmin, Mzmax)
        '''
        self.z = z
        self.n = n
        
        Mzmax, Mz0, b_len  = interaction_params
        self.b_length = b_len
        self.MZmin    = Mz0
        self.MZmax    = Mzmax
        
        if Mz0 == 0:
            self.interaction = self.HAMIL_NAME + f'_MZ{self.MZmax}'
            self.HAMIL_NAME += f'_MZ{self.MZmax}'
        else:
            raise Exception("Method not implemented!")
        
        self.bu_folder = Path(bu_folder)
        
        self._interaction_parts = []
        self.hamil_dest_folders = {}
        self.input_vap : InputTaurus    = None
        self.input_pav : InputTaurusPAV = None
        self.constraint = None
        self.export_filenames = {}
    
    def setUpHamiltonians(self, interactions, combine_all=False):
        """
        From Hamiltonian
        """
        
        for name, interaction_runable in interactions.items():
            exe_ = TBME_HamiltonianManager(self.b_length, self.MZmax, self.MZmin, 
                                           set_com2=True)
            exe_.hamil_filename = name
            if not isinstance(interaction_runable, list): 
                interaction_runable = [interaction_runable, ]
                         
            exe_.setAndRun_ComposeInteractions(interaction_runable)
            self._interaction_parts.append(exe_.hamil_filename)
            printf(f"  Hamil part [{exe_.hamil_filename}] available.\n")
        
        ## Get all the results and combine them from the files
        if combine_all:
            bench_combination = []
            for name in self._interaction_parts:
                # reference from the 2B_MatrixElements folder
                kwargs = {'filename': f"results/{name}.2b",} 
                bench_combination.append((TBMEXML_Setter.set_file_force, kwargs ))
            
            exe_ = TBME_HamiltonianManager(self.b_length, self.MZmax, self.MZmin, 
                                           set_com2=True)
            exe_.hamil_filename = 'bench'
            exe_.setAndRun_ComposeInteractions(bench_combination)
            self._interaction_parts.append(exe_.hamil_filename)
            
            interactions['bench'] = bench_combination
        
        printf(LINE_2, "All parts obtained:")
        for name in self._interaction_parts:
            printf(" * ", name, ": ", list(filter(lambda x: x.startswith(name), 
                                                  os.listdir())) )
        printf(LINE_2)
    
    def setUpVAPparameters(self, **input_args):
        self.input_vap    = InputTaurus(self.z, self.n, '', None, **input_args)
        self.VAP_INP_ARGS = input_args
    
    def setUpPAVparameters(self, **input_args):
        self.input_pav    = InputTaurusPAV(self.z, self.n, '', **input_args)
        self.PAV_INP_ARGS = input_args
        self.DO_PROJECTION = True
    
    def _sort_deforms(self, list_files):
        if not isinstance(list_files, list): list_files = list(list_files)
        
        pos_files = filter(lambda x: x[0] != '_', list_files)
        neg_files = filter(lambda x: x[0] == '_', list_files)
        
        pos_files = [(x, float(x[:-4])) for x in pos_files]
        neg_files = [(x, float(x[:-4].replace('_', '-'))) for x in neg_files]
        
        pos_files = sorted(pos_files, key = lambda x: x[1])
        neg_files = sorted(neg_files, key = lambda x: x[1])
        
        return neg_files + pos_files
    
    def processData(self, folder2import, observable=None):
        """
        :folder2import: folder in main self.bu_folder to evaluate
        :reset_data=True: remove al files and data from previous calculations
        """
        fld_bu = self.bu_folder / folder2import
        if not fld_bu.exists():
            printf(f" [ERROR] Folder [{fld_bu}] not found. SKIP")
            return False
        self.observable = observable
        
        ## Folder to import: from PNVAP:
        files_list = os.listdir(fld_bu / 'PNVAP')
        files_list = filter(lambda x: x.endswith('.bin'), files_list)        
        self.b20_sorted = self._sort_deforms(files_list)
        
        ## deform index critera from export_TES_file
        export_fn = folder2import.replace('BU_folder_','')
        export_fn = export_fn.replace(self.interaction,'').replace(f'z{self.z}n{self.n}', '')
        export_fn = export_fn.replace('_', '')
        export_fn = f"export_TES_{export_fn}_z{self.z}n{self.n}_{self.interaction}.txt"
        self.deform_indexes = {}
        if os.path.exists(fld_bu / export_fn):
            with open(export_fn, 'r') as f:
                for l in f.readlines()[1:]:
                    args, _ = l.split(OUTPUT_HEADER_SEPARATOR)
                    i, v = args.split(':')
                    i = int(i)
                    v = v.strip().replace('+','') + '.bin'
                    self.deform_indexes[v] = i
        else:
            for v, _ in self.b20_sorted:
                self.deform_indexes[len(self.deform_indexes)] = v
        
        ## create surfaces-binary paths for execution and exporting
        self.surfaces = dict()
        self.results  = dict([(int_, dict()) for int_ in self._interaction_parts])
        self.hamil_dest_folders = {}
        
        for inter in self._interaction_parts:
            ## Create the storage folders for each inter
            dest_fld = fld_bu / f'HAMIL_DECOMP_{inter}'
            if dest_fld.exists():
                shutil.rmtree(dest_fld)
            os.mkdir(dest_fld)
            self.hamil_dest_folders[inter] = dest_fld
            for tail_ in OutputFileTypes.members():
                f = inter + tail_
                if os.path.exists(f): shutil.copy(f, self.hamil_dest_folders[inter])
            
            ## fix the exporting_fn for each interactions
            export_fn_2 = export_fn.replace(self.interaction, inter)
            self.export_filenames[inter] = dest_fld / export_fn_2
            
        ## Read the files to be copied
        verify_out = True
        for f, _ in self.b20_sorted:
            self.surfaces[f] = fld_bu / f"PNVAP/{f}"
            if verify_out:
                fout = fld_bu / f"PNVAP/{f}".replace('.bin', '.OUT')
                self._verifyIfSolutionsAreVAPprojectedAndReadTheFomenko(fout)
                verify_out = False
        
        return True
        
    def _verifyIfSolutionsAreVAPprojectedAndReadTheFomenko(self, file_out):
        """ 
        Having different VAP fomenko points could lead to inconsistent energy
        outcome, if required (by default) check with the original solution.
        """
        if self.IGNORE_FOMENKO_POINTS_FROM_BU_SOLUTIONS: return
            
        curr_fom = self.input_vap.z_Mphi, self.input_vap.n_Mphi
        
        _HHM = 'Master name hamil. files      '
        _HZ  = 'Number of active protons      '
        _HN  = 'Number of active neutrons     '
        _HFZ = 'No. of gauge angles protons   '
        _HFN = 'No. of gauge angles neutrons  '
        with open(file_out, 'r') as f:
            lines = f.readlines()[:90]  ## These lines are difficult to be moved
            for i,h in {22: _HHM, 29: _HZ, 30: _HN, 31: _HFZ, 32: _HFN}.items():
                assert lines[i].startswith(h), "Something wrong with the taurus-vap Output!"
            inter = lines[22].replace(_HHM, '').strip()
            z     = lines[29].replace(_HZ , '').replace('.00', '').strip()
            n     = lines[30].replace(_HN , '').replace('.00', '').strip()
            fomz  = lines[31].replace(_HFZ, '').strip()
            fomn  = lines[32].replace(_HFN, '').strip()
        ref_fom  = int(fomz), int(fomn)
        
        ok_ = [
            inter   == self.HAMIL_NAME,
            int(z)  == self.z,
            int(n)  == self.n,
            ref_fom == curr_fom,
        ]       
        if all(ok_):  return
        else:
            _ERR = [
                 'Error for current VAP parameters and BU-states ones:',
                f' * hamil name [{self.HAMIL_NAME}/{inter}]  [status={ok_[0]}]',
                f' * Z [{self.z}/{z}]  [status={ok_[1]}]',
                f' * N [{self.n}/{n}]  [status={ok_[2]}]',
                f' * Fomenko points [{curr_fom}/{ref_fom}]  [status={ok_[3]}]\n'                
            ]
            _ERR = '\n'.join(_ERR)
            raise BaseException(f" [STOP] {_ERR}")
    
    def run(self):
        """
        Copy every solution in order and evaluate each hamiltonian with 0 steps
        Store results for export file and move into new BU.
        """
        assert self.input_vap.iterations == 0, "Fix iterations to 0!!"
        
        for i, k_b20 in enumerate(self.b20_sorted):
            k_b20, b20 = k_b20
            if i > 0: self.input_vap.red_hamil = 1 # already executed
            
            shutil.copy(self.surfaces[k_b20], 'initial_wf.bin')
            
            printf(f"  exe b20 = {b20:5.3f} / {k_b20}")
            printf("   * interaction *:   [   E HFB  ]  [   E HF   ]  [ E pairing]   ------")
            for inter in self._interaction_parts:
                inp_fn = self.input_vap.input_filename
                out_fn = 'aux_output.OUT'
                
                ## create the input file
                self.input_vap.interaction = inter
                with open(inp_fn, 'w+') as f:
                    f.write(self.input_vap.getText4file())
                
                ## execute void step
                try:
                    # NOTE: for test Windows, put a output in main folder
                    if not os.getcwd().startswith('C'): 
                        os.system(f'./taurus_vap.exe < {inp_fn} > {out_fn}')
                except BaseException as e:
                    printf(f' [Error] execution failed for [{inter}], skipping!')
                    continue
                
                res = DataTaurus(self.z, self.n, out_fn)
                self.results[inter][k_b20] = deepcopy(res)
                shutil.copy(out_fn, self.hamil_dest_folders[inter] / k_b20.replace('bin', 'OUT'))
                
                printf(f"   {inter: >15}:  {res.E_HFB:12.4f}  {res.hf:12.4f}  {res.pair:12.4f}")
                ## teardown results into 
                self._exportFileFromEachInteraction(inter)
            
    
    def _exportFileFromEachInteraction(self, inter):
        """
        Print onrun the results from void step 
        """
        lines = [f'DataTaurus, {self.observable}', ]
        for k_b20, _ in self.b20_sorted:
            indx = self.deform_indexes[k_b20]
            b20  = k_b20[:-4]
            b20  = '+'+b20 if not '-' in b20 else b20
            head = f"{indx: >5}:{b20: >7}"
            
            if k_b20 in self.results[inter]:
                line = self.results[inter][k_b20].getAttributesDictLike
                lines.append(OUTPUT_HEADER_SEPARATOR.join([head, line]))
        
        lines = '\n'.join(lines)
        
        with open(self.export_filenames[inter], 'w+') as f:
            f.write(lines)
        
        