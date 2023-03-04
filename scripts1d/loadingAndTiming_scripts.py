'''
Created on Mar 1, 2023

@author: Miguel
'''

import os
from datetime import datetime
import shutil
import subprocess

from tools.executors import ExecutionException, ExeTaurus1D_AngMomentum
from tools.inputs import InputTaurus
from scripts1d.script_helpers import getInteractionFile4D1S,\
    parseTimeVerboseCommandOutputFile
from tools.data import DataTaurus
from tools.helpers import ValenceSpacesDict_l_ge10_byM, getSingleSpaceDegenerations,\
    prettyPrintDictionary, importAndCompile_taurus



def _exportResultsOrSummary(results, path_, key_order=None):
    """ 
    Save the result zips from the summary case or the Output Data results"""
    case = 0
    exp_sum_obj = []
    exp_dat_obj = []
    for MZ in range(len(results)):
        data_dict = results[MZ]
        
        for typ, vals in data_dict.items():
            if vals == None or len(vals)==0: continue
            
            header = [str(MZ), str(typ)]
            if isinstance(vals, tuple):
                ## Summary case
                case = 0
                vals_str = [str(v) for v in vals]
                line = "##".join([*header, ', '.join(vals_str)]) + '\n'
                exp_sum_obj.append(line)
                
            elif isinstance(vals[0], DataTaurus):
                ## Results case
                case = 1
                for i, val in vals.items():
                    if val== None: continue
                    line_csv = val.getAttributesDictLike
                    line = "##".join([*header, str(i), line_csv]) + '\n'
                    exp_dat_obj.append(line)
                exp_dat_obj.append('\n')
    
    with open(path_, 'w+') as f:
        if   case == 0:
            f.writelines(exp_sum_obj)
        elif case == 1:
            f.writelines(exp_dat_obj)


def _reset_hamil_files(sps_2_clear, filename, hamilf_out, com_file):
    extension = '.2b' if not com_file else '.com'
    
    final_ = []
    with open(filename + extension, 'r') as f:
        data  = f.read()
        data  = data.split(' 0 5 ')
        title = data[0]
    
    for sp_block in data[1:]:
        vals = sp_block.split('\n')
        sts_head = vals[0]
        sts_head = sts_head.split()[:-2]
        
        ignore_ = False
        for sp_clear in sps_2_clear:
            if sp_clear in sts_head:
                ignore_ = True
                break
        
        if not ignore_:
            final_.append(' 0 5 ' + sp_block)
    
    final_ = title + ''.join(final_)
    with open(hamilf_out + extension, 'w+') as f:
        f.write(final_)

def _removeShellsFromHamilAndCOMfile(remain_sts, MZ_2remove, filename, hamilf_out, 
                                     b_len=1.0):
    TEMPLATE_SHO = [
        " D1S no density . ME evaluated: Brink_Boeker + Coulomb.",
        "4", "", "    0 0",  "2 13.542"
        ]
    sts2rem = ValenceSpacesDict_l_ge10_byM[MZ_2remove]
    for st_ in sts2rem:
        try:
            remain_sts.remove(st_)
        except ValueError:
            pass
    aux_sts = []
    for sts in remain_sts:
        aux_sts = aux_sts + [st for st in sts]
    TEMPLATE_SHO[2] = "    {} {}".format(len(aux_sts), ' '.join(aux_sts)) 
    TEMPLATE_SHO[4] = "2 {:6.3f}".format(41.42460 / (b_len**2))
    
    # set SHO
    with open(hamilf_out+'.sho', 'w+') as f:
        f.write('\n'.join(TEMPLATE_SHO))
    # set .2b
    _reset_hamil_files(sts2rem, filename, hamilf_out, False)
    # set .COM
    _reset_hamil_files(sts2rem, filename, hamilf_out, True)
    
    return remain_sts


def _test_execution(z, n, hamil_fn, repetitions=3, iterations=50, ROmega=(10,10)):
    
    ## Change to not do every process
    DOFILE   = [True       , False       , True] 
    files_ = ['_Full.txt', '_noRea.txt', '_base.txt']
    files_ = [hamil_fn+f_ for f_ in files_]
    
    results_sh = {
        'full' : dict([(i, None) for i in range(repetitions)]),
        'noRea': dict([(i, None) for i in range(repetitions)]),
        'base' : dict([(i, None) for i in range(repetitions)])
    }
    
    params_ = {
        InputTaurus.ArgsEnum.iterations : iterations,
        InputTaurus.ArgsEnum.seed       : 4,
        InputTaurus.ArgsEnum.com        : 1,
        }
    dd_params_ = {
        InputTaurus.InpDDEnum.r_dim     : ROmega[0],
        InputTaurus.InpDDEnum.omega_dim : ROmega[1],
        }
    inp_t= InputTaurus(z, n, hamil_fn, input_filename=None, **params_)
    
    if DOFILE[0]:
        ## D1S_t0 + DD + Rea
        dd_params_[InputTaurus.InpDDEnum.eval_dd ] = 1
        dd_params_[InputTaurus.InpDDEnum.eval_rea] = 1
        InputTaurus.set_inputDDparamsFile(**dd_params_)
        print("   * doing D1S_t0 + DD + Rea")
        for iter_ in range(repetitions):
            out_t = files_[0].replace('.txt', f'_{iter_}.txt')
            res   = __execute_taurus_diagnose(inp_t, out_t)
            results_sh['full'][iter_] = res
            
    if DOFILE[1]:
        ## D1S_t0 + DD
        dd_params_[InputTaurus.InpDDEnum.eval_dd ] = 1
        dd_params_[InputTaurus.InpDDEnum.eval_rea] = 0
        InputTaurus.set_inputDDparamsFile(**dd_params_)
        print("   * doing D1S_t0 + DD")
        for iter_ in range(repetitions):
            out_t = files_[0].replace('.txt', f'_{iter_}.txt')
            res   = __execute_taurus_diagnose(inp_t, out_t)
            results_sh['noRea'][iter_] = res
        
    if DOFILE[2]:
        ## D1S_t0
        dd_params_[InputTaurus.InpDDEnum.eval_dd ] = 0
        dd_params_[InputTaurus.InpDDEnum.eval_rea] = 0
        InputTaurus.set_inputDDparamsFile(**dd_params_)
        print("   * doing D1S_t0")
        for iter_ in range(repetitions):
            out_t = files_[0].replace('.txt', f'_{iter_}.txt')
            res   = __execute_taurus_diagnose(inp_t, out_t)
            results_sh['base'][iter_] = res
        
    # Export temporal values of the sh_ results
    shell_time_ram_results = {}
    for indx, key_ in enumerate(['full', 'noRea', 'base']):
        if not DOFILE[indx]: 
            continue
        shell_time_ram_results[key_] = _summaryOfCalculations(key_, results_sh, ROmega)
    
    return results_sh, shell_time_ram_results

def _summaryOfCalculations(calc_opt, results, ROmega):
    """
    Average the cpu_time per iter, maximum memory_used, integration dims
    Returns:
        MZmax, sh_states, sp_states, 
        int_dims, 
        cpu_time_s, ram_max_kB , 
        best_dens_approx
    """
    cpu_time_s = 0.0
    ram_max_kB = 0
    sh_states  = None
    sp_states  = None
    MZmax      = None
    repet      = len(results)
    
    __nPoints = [6,14,26,38,50,74,86,110,146,170,194,230,266,
                 302,350,434,590,770,974,1202,1454,1730,2030] ## 22
    int_dims =  __nPoints[ROmega[1] - 1] * ROmega[0]
    
    res : DataTaurus = None
    for res in results[calc_opt].values():
        cpu_time_s += res.time_per_iter_cpu
        ram_max_kB += res.memory_max_KB
        if not sp_states:
            sp_states = res.sp_dim
            sh_states = res.sh_dim
            MZmax     = res.MZmax
        
        ## res._dataEvol.sp_dens (last) to see precision approximation
        best_dens_approx = getattr(res._evol_obj, 'sp_dens')
        if best_dens_approx != None:
            best_dens_approx = best_dens_approx[-1]
    

    return (MZmax, sh_states, sp_states, int_dims, 
            cpu_time_s / repet, ram_max_kB / repet, best_dens_approx)

def __execute_taurus_diagnose(inp_taurus: DataTaurus, output_fn):
    """
    auxiliary method to perform safely the program and
    export: DataTaurus result, CPU time and Memory RAM used by the process.
    """
    
    res = None
    with open(inp_taurus.INPUT_DD_FILENAME, 'w+') as f:
        f.write(inp_taurus.get_inputDDparamsFile())
    
    try:
        with open(inp_taurus.input_filename, 'w+') as f2:
            f2.write(inp_taurus.getText4file())
        
        _inp_fn  = inp_taurus.input_filename
        _time_fn = '_time_taurus.log'
        # 
        if os.getcwd().startswith('C:'): ## Testing purpose on Windows
            file2copy = "TEMP_output_Z10N10_max_iter.txt"
            txt = ''
            with open(file2copy, 'r') as f:
                txt = f.read()
                txt = txt.format(INPUT_2_FORMAT=inp_taurus)
            with open(output_fn, 'w+') as f:
                f.write(txt)
            shutil.copy('TEMP_time_verbose_output.txt', _time_fn)
        else:
            if not 'taurus_vap.exe' in os.listdir():
                importAndCompile_taurus()
            order_ = f'./taurus_vap.exe < {_inp_fn} > {output_fn}'
            order_ = f'{{ /usr/bin/time -v {order_}; }} 2> {_time_fn}'
            _e = subprocess.call(order_, 
                                 shell=True,
                                 timeout=43200) # 12 h timeout
        
        res = DataTaurus(inp_taurus.z, inp_taurus.n, output_fn)
        
        ## Process the time log file
        vals = parseTimeVerboseCommandOutputFile(_time_fn)
        iters_ = max(1, getattr(res, 'iter_max', 0))
        res.iter_time_cpu = vals['user_time']
        res.time_per_iter_cpu = res.iter_time_cpu / iters_
        res.memory_max_KB = vals['memory_max']
        
    except Exception as e:
        print(f"  [FAIL]: _executeProgram()")
        if isinstance(res, DataTaurus):
            print(f"  [FAIL]: result=", str(res.getAttributesDictLike))
        
    return res
    


def run_IterTimeAndMemory_from_Taurus_byShellsAndIntegrationMesh(
        ROmegaMax=(10,12), z_numb=2, n_numb=2):
    """
    This script runs dens_taurus_vap program for the D1S shell. 
    Depending on the 
    
    """
    repetitions = 5
    iterations  = 10 
    Mzmax  = 7
    b_len  = 1.75
    
    def_inter = {(z_numb, n_numb) : (Mzmax, 0, b_len), }
    if all([os.path.exists(f'D1S_MZ{Mzmax}.{ext}') for ext in ('2b','com','sho')]):
        ## Option to reuse the hamiltonian
        def_inter = {(z_numb, n_numb) : f'D1S_MZ{Mzmax}', }
    ## 1) Build a very large Hamiltonian_ (Max shell), save the states for 
    ## the procedure to remove from shell to shell.
    hamil_filename = getInteractionFile4D1S(def_inter, z_numb,n_numb)
    rem_sts = [list(ValenceSpacesDict_l_ge10_byM[MZ]) for MZ in range(Mzmax)]
    
    data_times   = []
    data_results = []
    ## 2) Iterate the shells to evaluate (reversed to clean)
    for iter_, MZ in enumerate(range(Mzmax+1, 0, -1)):
        sh_, sp_dim = getSingleSpaceDegenerations(MZ)
        print(f"[STEP {iter_:2}] Running MZ={MZ} dimens({sh_},{sp_dim}) for z{z_numb}n{n_numb}" )
        if z_numb >= sp_dim or n_numb >= sp_dim:
            print("[WARNING] Process STOP!, reached the occupation for the N,Z")
            break
        
        new_hamil_fn = f'hamil_MZ{MZ}' 
        rem_sts  = _removeShellsFromHamilAndCOMfile(rem_sts, MZ, hamil_filename, 
                                                    new_hamil_fn, b_len)
        
        ## Validate the options of using or not the DD term
        ## Get precision markers to validate the integration.
        results_sh = _test_execution(z_numb, n_numb, new_hamil_fn, 
                                     repetitions, iterations, ROmegaMax)
        print()
        prettyPrintDictionary(results_sh[1]) # Times and shit !
        print()
        
        data_results.append(results_sh[0])
        data_times.  append(results_sh[1])
        ## : Save the results in terms of the involved shell
        _exportResultsOrSummary(data_results, 'global_results.txt')
        _exportResultsOrSummary(data_times,   'global_summary.txt')
        
    f"[MAIN END] Main running of substracting shells are done."
    ## TODO: We need the plotters for the results:
    ## 1. The Time/sp_dim    2. Time/integ_dim   3. Regression Time (sp_dim, RO)
    _=0