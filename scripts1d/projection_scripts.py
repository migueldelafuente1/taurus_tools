# encoding: utf-8
'''
Project with the PAV program over results in a folder:

    Complete Kernels
    Only diagonal terms
'''
import os, shutil
import subprocess
from tools.inputs import InputTaurusPAV
from copy import deepcopy
from tools.data import DataTaurusPAV
from tools.base_executors import _Base1DTaurusExecutor
from tools.helpers import printf


def run_diagonal_pavResults_even_even_nuclei(FLD_path, Z, N, INTER,
                                             default_pav_input = {},
                                             sort_by_seed_deform_index = False,
                                             range_2J = (0, 16),
                                             fomenko_points=(1, 1)):
    """
    Get the files from a folder with binaries /txt wf.
    
    :args:
        FLD_path, Z, N, INTER
    
    :kwargs:
        :default_pav_input
        :sort_by_seed_deform_index . If True, reads from a list_dat.txt file.
        :range_2J = (2J minimum, 2J maximum)
        :fomenko_points = (Z dim, N dim))
    """
    
    if not os.path.exists(FLD_path):
        printf("WARNING: FLD_path does not exist or contain PNVAP, exit:", FLD_path)
        return
    
    FLD_results = 'PNAMP'
    
    __DEFAULT_PAV_INPUT = {
        InputTaurusPAV.ArgsEnum.red_hamil : 1,
        InputTaurusPAV.ArgsEnum.com: 1,
        InputTaurusPAV.ArgsEnum.alpha : 12,
        InputTaurusPAV.ArgsEnum.beta  : 15,
        InputTaurusPAV.ArgsEnum.gamma : 12,
        InputTaurusPAV.ArgsEnum.empty_states : 0,
        InputTaurusPAV.ArgsEnum.disable_simplifications_P : 0,
        InputTaurusPAV.ArgsEnum.cutoff_overlap : 1.0e-10,
        InputTaurusPAV.ArgsEnum.read_as_txt : 0,
        # PN-PAV and J bound arguments set by the program, P-PAV = no
    }
    
    if not default_pav_input:
        default_pav_input = deepcopy(__DEFAULT_PAV_INPUT)
    else:
        for k, v in __DEFAULT_PAV_INPUT.items():
            if not k in default_pav_input: default_pav_input[k] = v
    
    default_pav_input[InputTaurusPAV.ArgsEnum.j_min]  = range_2J[0]
    default_pav_input[InputTaurusPAV.ArgsEnum.j_max]  = range_2J[1]
    default_pav_input[InputTaurusPAV.ArgsEnum.z_Mphi] = fomenko_points[0]
    default_pav_input[InputTaurusPAV.ArgsEnum.n_Mphi] = fomenko_points[1]
    
    inputObj_PAV = InputTaurusPAV(Z, N, INTER)
    inputObj_PAV.setParameters(**default_pav_input)
    
    deform_index_list = []
    deform_wf_files   = {}
    
    list_wf, bin_case = [], True
    if sort_by_seed_deform_index:
        ## You are in a BU_folder_INTER_Z_N, 
        ## 1. if list_dict.dat -> use it to store the files
        if os.path.exists(FLD_path + '/list_dict.dat'):
            with open(FLD_path + '/list_dict.dat', 'r') as f:
                for l in f.readlines():
                    file_, b20 = l.split()
                    
                    d_index = 0
                    if file_.startswith('seed_'):
                        _, _, d_index, case_ = file_.split('_')
                    
                    deform_wf_files[f"{d_index}_{case_}"] = file_
                    deform_index_list.append(d_index)
                    deform_wf_files[d_index] = b20
                    
                    list_wf.append(file_)
                    bin_case = '.bin' in file_
    else:
        list_wf = os.listdir(FLD_path)
        list_wf = list(filter(lambda x: x.startswith('seed_'), list_wf))
        list_wf  = list(filter(lambda x: x.endswith('.bin'), list_wf))
        bin_case = len(list_wf) > 0
        if not bin_case:
            list_wf  = list(filter(lambda x: x.endswith('.txt'), list_wf))
            if len(list_wf) > 0:
                default_pav_input[InputTaurusPAV.ArgsEnum.read_as_txt] = 1
            else:
                printf(" WARNING : No files found, Exiting.")
                return
        
        raise Exception("Sorting not specified, use a list_dat file or Implement me!")
    
    if os.path.exists(FLD_path + '/' + FLD_results):
        shutil.rmtree(FLD_path + '/' + FLD_results)
    os.mkdir(FLD_path + '/' + FLD_results)
    
    _fmt = '.bin' if bin_case else '.txt' 
    for i, wf in enumerate(list_wf):
        
        shutil.copy(FLD_path + '/' + wf, f'left_wf{_fmt}' )
        shutil.copy(FLD_path + '/' + wf, f'right_wf{_fmt}' )        
        
        inp_fn = inputObj_PAV.DEFAULT_INPUT_FILENAME
        out_fn = DataTaurusPAV.DEFAULT_OUTPUT_FILENAME
        with open(inp_fn, 'w+') as f2:
            f2.write(inputObj_PAV.getText4file())
        
        
        d_index = deform_index_list[i]
        printf(f" * [{i+1: >2}/{len(list_wf)}] : {wf} -> {d_index: >6}  {deform_wf_files[d_index]}")
        if os.getcwd().startswith('C:'): ## Testing purpose 
            _Base1DTaurusExecutor._auxWindows_executeProgram_PAV(None, out_fn)
        else:
            os.system('./taurus_pav.exe < {} > {}'.format(inp_fn, out_fn))
        
        shutil.move( out_fn,  f'{FLD_path}/{FLD_results}/{d_index}.OUT')
    
    with open(f'{FLD_path}/{FLD_results}/list_pav.dat', 'w+') as f:
        f.writelines('\n'.join([x+'.OUT' for x in deform_index_list]))
    
    printf(f" Finished.")

