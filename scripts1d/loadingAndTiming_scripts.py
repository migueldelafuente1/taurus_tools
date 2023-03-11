'''
Created on Mar 1, 2023

@author: Miguel
'''

import os
import re
from datetime import datetime
import shutil
import subprocess
import numpy as np

from tools.executors import ExecutionException, ExeTaurus1D_AngMomentum
from tools.inputs import InputTaurus
from scripts1d.script_helpers import getInteractionFile4D1S,\
    parseTimeVerboseCommandOutputFile
from tools.data import DataTaurus
from tools.helpers import ValenceSpacesDict_l_ge10_byM, getSingleSpaceDegenerations,\
    prettyPrintDictionary, importAndCompile_taurus, linear_regression,\
    LEBEDEV_GRID_POINTS
from copy import deepcopy

if os.getcwd().startswith("C:"):
    import matplotlib.pyplot as plt

__RESULTS_FILENAME = 'global_results'
__SUMMARY_FILENAME = 'global_summary'

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


def _reset_hamil_files(remain_sts, sps_2_clear, filename, hamilf_out, com_file):
    extension = '.2b' if not com_file else '.com'
    valid_states = []
    for sh_sts in remain_sts:
        valid_states = valid_states + sh_sts
    
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
        # for sp_clear in sps_2_clear:
        #     if sp_clear in sts_head:
        #         ignore_ = True
        #         break
        for sp_ in sts_head:
            if sp_ not in valid_states:
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
        for sh_ in range(len(remain_sts)):
            try:
                remain_sts[sh_].remove(st_)
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
    _reset_hamil_files(remain_sts, sts2rem, filename, hamilf_out, False)
    # set .COM
    _reset_hamil_files(remain_sts, sts2rem, filename, hamilf_out, True)
    
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
        cpu_time_s average (per iter), ram_max_kB average,  
        cpu_desv std, ram_desv std, repetitions, 
        best_dens_approx
    """
    cpu_time_s = []
    ram_max_kB = []
    sh_states  = None
    sp_states  = None
    MZmax      = None
    repet      = len(results)
    
    int_dims =  LEBEDEV_GRID_POINTS[ROmega[1] - 1] * ROmega[0]
    
    res : DataTaurus = None
    for res in results[calc_opt].values():
        cpu_time_s.append(res.time_per_iter_cpu)
        ram_max_kB.append(res.memory_max_KB)
        if not sp_states:
            sp_states = res.sp_dim
            sh_states = res.sh_dim
            MZmax     = res.MZmax
        
        ## res._dataEvol.sp_dens (last) to see precision approximation
        best_dens_approx = getattr(res._evol_obj, 'sp_dens')
        if best_dens_approx != None:
            best_dens_approx = best_dens_approx[-1]
    
    if len(cpu_time_s) == 0:
        cpu_time_s_mean, cpu_stdv = 0.0, 0.0
        ram_max_kB_mean, ram_stdv = 0, 0.0
    else:
        cpu_time_s_mean = np.mean(cpu_time_s)
        cpu_stdv = np.std(cpu_time_s) * (repet/(repet-1))**.5
        ram_max_kB_mean = np.mean(ram_max_kB)
        ram_stdv = np.std(ram_max_kB) * (repet/(repet-1))**.5
    
    return (MZmax, sh_states, sp_states, int_dims, 
            cpu_time_s_mean, ram_max_kB_mean, cpu_stdv, ram_stdv, repet,
            best_dens_approx)

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
            _e = subprocess.call(order_, shell=True,
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
        Mzmax=7, ROmegaMax=(10,12), z_numb=2, n_numb=2):
    """
    This script runs dens_taurus_vap program for the D1S shell. 
    Depending on the 
    
    """
    repetitions = 5
    iterations  = 10 
    b_len  = 1.75
    
    def_inter = {(z_numb, n_numb) : (Mzmax, 0, b_len), }
    if all([os.path.exists(f'D1S_MZ{Mzmax}.{ext}') for ext in ('2b','com','sho')]):
        ## Option to reuse the hamiltonian
        def_inter = {(z_numb, n_numb) : f'D1S_MZ{Mzmax}', }
    ## 1) Build a very large Hamiltonian_ (Max shell), save the states for 
    ## the procedure to remove from shell to shell.
    hamil_filename = getInteractionFile4D1S(def_inter, z_numb,n_numb)
    rem_sts = [list(ValenceSpacesDict_l_ge10_byM[MZ]) for MZ in range(Mzmax +1)]
    
    str_rome = 'R{}_O{}'.format(*ROmegaMax)
    data_times   = []
    data_results = []
    ## 2) Iterate the shells to evaluate (reversed to clean)
    for iter_, MZ in enumerate(range(Mzmax+1, 0, -1)):
        sh_, sp_dim = getSingleSpaceDegenerations(MZ)
        print(f"[STEP {iter_:2}] Running MZ={MZ} dimens({sh_},{sp_dim}) for z{z_numb}n{n_numb}" )
        if z_numb >= sp_dim or n_numb >= sp_dim:
            print("[WARNING] Process STOP!, reached the occupation for the N,Z")
            break
        
        new_hamil_fn = f'hamil_noMZ{MZ}' 
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
        _exportResultsOrSummary(data_results, f'{__RESULTS_FILENAME}_{str_rome}.txt')
        _exportResultsOrSummary(data_times,   f'{__SUMMARY_FILENAME}_{str_rome}.txt')
        
    f"[MAIN END] Main running of substracting shells are done."
    ## TODO: We need the plotters for the results:
    ## 1. The Time/sp_dim    2. Time/integ_dim   3. Regression Time (sp_dim, RO)

def getCPUandIterTimeFromCPU(res : DataTaurus):
    """
    Approximate the time for the setUp from the total CPU time from the 
    relative time lapse of the 
    """
    _tcpu_setup, _tcpu_per_iter = 0, 0
    
    tcpu_tot  = res.iter_time_cpu
    treal_tot = res.date_end_iter - res.date_start
    treal_setUp = res.date_start_iter - res.date_start
    if treal_tot.seconds == 0:
        _=0
    ratio_real = (treal_setUp.seconds + 1e-6*treal_setUp.microseconds) / \
        (treal_tot.seconds + 1e-6*treal_tot.microseconds)
    
    _tcpu_setup = tcpu_tot * ratio_real
    _tcpu_per_iter = tcpu_tot * (1 - ratio_real) / res.iter_max

    return _tcpu_setup, _tcpu_per_iter

def _averageOfValues2Print(*datas):
    """ For the list of MZ values"""
    datasAvr = []
    for data_ in datas:
        for RO_key, mz_vals in data_.items():
            for mz, vals_lists in mz_vals.items():
                if not isinstance(vals_lists[0], list): 
                    data_[RO_key][mz] = vals_lists
                    continue  ## this value has been processed
                
                rep = len(vals_lists)
                f_std = (rep / (rep - 1))**0.5
                
                tcpuiter = [x[ 4] for x in vals_lists]
                ram      = [x[ 5] for x in vals_lists]
                tcpuSUp  = [x[10] for x in vals_lists]
                av_vals = [
                    *vals_lists[0][:4], np.mean(tcpuiter), np.mean(ram),
                    f_std * np.std(tcpuiter), f_std * np.std(ram), rep, 0.0,
                    np.mean(tcpuSUp), f_std * np.std(tcpuSUp)
                ]
                data_[RO_key][mz] = av_vals
        datasAvr.append(deepcopy(data_))
    return datasAvr
    
    
def _importDataFromResultsFiles(files2plot, folder_):
    """ Process data from individual DataTaurus results in "results" files """
    data_base = {}
    data_noRea= {}
    data_full = {}
    for file_ in files2plot:
        if not file_.startswith(__RESULTS_FILENAME): continue
        _,R,O = file_.replace(__RESULTS_FILENAME,'').replace('.txt','').split('_')
        R,  O = int( R.replace('R', '')), int( O.replace('O', ''))
        RO_key = (R, O)
        data_base[RO_key] = {}
        data_noRea[RO_key]= {}
        data_full[RO_key] = {}
        integr_points = LEBEDEV_GRID_POINTS[ O - 1 ] * R
        
        with open(folder_+file_, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) == 0: 
                    continue
                ## Process the data
                _, typ_, iter_, vals = line.split('##')
                res = DataTaurus(0, 0, None, True)
                res.setDataFromCSVLine(vals)
                mz = int(res.MZmax)
                _tcpu_setup, _tcpu_per_iter = getCPUandIterTimeFromCPU(res)
                
                vals = [
                    mz, int(res.sh_dim), int(res.sp_dim), integr_points,
                    _tcpu_per_iter, res.memory_max_KB / 1024, 
                    0.0, 0.0, 0, 0.0, _tcpu_setup, 0.0]
                
                ## Save the data (MZ is the second key)
                if   typ_ == 'base':
                    if mz not in data_base[RO_key]: 
                        data_base[RO_key][mz] = [vals, ]
                    else:
                        data_base[RO_key][mz].append(vals)
                elif typ_ == 'noRea':
                    if mz not in data_noRea[RO_key]: 
                        data_noRea[RO_key][mz]= [vals, ]
                    else:
                        data_noRea[RO_key][mz].append(vals)
                elif typ_ == 'full':
                    if mz not in data_full[RO_key]: 
                        data_full[RO_key][mz]= [vals, ]
                    else:
                        data_full[RO_key][mz].append(vals)
            
    ## do the averages from the lists.
    args = data_base, data_noRea, data_full
    data_base, data_noRea, data_full = _averageOfValues2Print(*args)
                
    return {
        'base' : data_base,
        'noRea': data_noRea,
        'full' : data_full
    }

def _importDataFromSummaryFiles(files2plot, folder_):
    """ Process data from Summary files """
    data_base = {}
    data_noRea= {}
    data_full = {}
    for file_ in files2plot:
        if not file_.startswith(__SUMMARY_FILENAME): continue
        _,R,O = file_.replace(__SUMMARY_FILENAME,'').replace('.txt','').split('_')
        R,  O = int( R.replace('R', '')), int( O.replace('O', ''))
        RO_key = (R, O)
        data_base[RO_key] = {}
        data_noRea[RO_key]= {}
        data_full[RO_key] = {}
        
        with open(folder_+file_, 'r') as f:
            
            for line in f.readlines():
                ## Process the data
                _, typ_, vals = line.split('##')
                vals = vals.split(', ')
                for i in range(len(vals)):
                    if  (i < 4) or (i == 8):
                        vals[i] = int  ( vals[i] )
                    elif i != 9:
                        vals[i] = float( vals[i] )
                    else:
                        if typ_ != 'base':
                            vals[i] = float( vals[i] )
                        else:
                            vals[i] = 0                            
                
                ## Save the data (MZ is the second key)
                if   typ_ == 'base':
                    data_base[RO_key][vals[0]] = vals
                elif typ_ == 'noRea':
                    data_noRea[RO_key][vals[0]]= vals
                elif typ_ == 'full':
                    data_full[RO_key][vals[0]] = vals
    
    return {
        'base' : data_base,
        'noRea': data_noRea,
        'full' : data_full }

def __plotEfficiencyShitFromFilesInFolder(DATA_FOLDER, IMPORT_FILENAME, MODE2PRINT,
                                          EXPORT_PDF_IMG=False):
    """
    Print all results [IMPORT_FILENAME] from [DATAFOLDER], being IMPORT_FILENAME
    the "summary files" or explicit DataTaurus "Results_files",
    """
    
    assert MODE2PRINT in ('base', 'noRea', 'full'), "Invalid Mode"
    assert IMPORT_FILENAME in (__SUMMARY_FILENAME, __RESULTS_FILENAME), "Invalid fileTypes"
    
    files2plot = list(filter(lambda f: f.startswith(IMPORT_FILENAME), 
                             os.listdir(DATA_FOLDER)))
    # sort by Rand Omega order
    sorteable_files = []
    for file_ in files2plot:
        k_ = [int(x) for x in re.findall(r'\d+', file_)]
        k_ = [k_[1], k_[0], file_, ]
        sorteable_files.append(k_)
    files2plot = [k_[2] for k_ in sorted(sorteable_files)] # sort R then Omega
    
    ## PROCESS THE DATA TO PLOT
    if   IMPORT_FILENAME == __SUMMARY_FILENAME:
        data_summary = _importDataFromSummaryFiles(files2plot, DATA_FOLDER)
    elif IMPORT_FILENAME == __RESULTS_FILENAME:
        data_summary = _importDataFromResultsFiles(files2plot, DATA_FOLDER)
    
    fig1, ax1 = plt.subplots() # plot sp^3    vs  t_cpu
    fig2, ax2 = plt.subplots() # plot RO      vs  t_cpu
    fig3, ax3 = plt.subplots() # plot RO*sp^2 vs  ram
    fig4, ax4 = plt.subplots() # plot sp_dim  vs  precision on spatial density 
    fig5, ax5 = plt.subplots() # plot the trends of t_cpu_iter/N^x by RO
    fig6, ax6 = plt.subplots() # plot the trends of t_cpu for setUp (ONLY SUMMARY)
    
    _func_sppow = {
        'base' : lambda xx: xx**3 , 
        'noRea': lambda xx: xx**3 ,
        'full' : lambda xx: xx**2.5 ,}
    
    mz_set, RO_set = set(), set()
    trends_MZ_RO_list = []
    for RO_, mzvals in data_summary[MODE2PRINT].items():
        x_mz = {}
        x_RO = []
        x_sp = []
        x_sp_pw     = []
        x_sp_pw_RO  = []
        y_Tcpu_s    = []
        y_ram_MB    = []
        err_dens    = []
        RO_set.add(RO_)
        err_bar_cpu = []
        
        for mz, vals in mzvals.items():
            mz_set.add(mz)
            x_mz[vals[2]] = int(vals[0])
            x_sp.append(int(vals[2]))
            x_sp_pw.append(_func_sppow[MODE2PRINT](x_sp[-1]))
            x_RO.append(int(vals[3]))
            
            y_Tcpu_s.append(vals[4])
            y_ram_MB.append(vals[5] / 1024)
            # err_dens.append(abs(round(vals[6]) - vals[6]))
            err_dens.append(max(abs(round(vals[9]) - vals[9]), 1.e-9))
            
            #err_bar_cpu.append(np.random.rand()*0.05*y_Tcpu_s[-1])
            err_bar_cpu.append(vals[6])           ## import from file  (vals[6])
            if MODE2PRINT != 'base':
                x_sp_pw_RO.append(x_sp[-1]**2 * x_RO[-1]**2)
            else:
                x_sp_pw_RO.append(x_sp[-1]**3)    
        
        A,B = linear_regression(x_sp_pw, y_Tcpu_s)
        RO_label = "R,Om:{}= {:>5}".format(RO_, x_RO[0])
        ax1.plot(x_sp_pw, y_Tcpu_s, '*--', 
                 label=RO_label + "  Reg: A[{:4.2e}] B[{:4.2e}]".format(A,B))
        ax1.errorbar(x_sp_pw, y_Tcpu_s, yerr=err_bar_cpu, 
                     fmt = '.', elinewidth=1.5, capsize=7, color='k')
        trends_MZ_RO_list.append( (x_RO[0], A) )
        ## add the sp-dimensions involved in the calculation (no scaling) -----
        ax12 = ax1.twiny()
        ax12.set_xlim(ax1.get_xlim())
        ax12.set_xticks(x_sp_pw)
        ax12.set_xticklabels([f"{x_mz[a]}:{a}" for a in x_sp])
        ax12.set_xlabel("MZ and sp dimension")
        ## --------------------------------------------------------------------
        
        A,B = linear_regression(x_sp_pw_RO, y_ram_MB)
        ax3.plot(x_sp_pw_RO, y_ram_MB, '.-', 
                 label=RO_label + "  Reg: A[{:4.2e}] B[{:4.2e}]".format(A,B))
        
        ax4.plot(x_mz.values(), np.log10(err_dens), '.-', label=RO_label)
        
        ## For Collection of Results, the time cpu for the SetUp of fields is available
        if IMPORT_FILENAME == __RESULTS_FILENAME:
            xx = np.power(np.array(x_sp), 3)
            y_TcpuSetUp, err_TcpuSetUp_s = [], []
            for mz, vals in mzvals.items():
                y_TcpuSetUp.    append(vals[10])
                err_TcpuSetUp_s.append(vals[11])
            A,B = linear_regression(xx, y_TcpuSetUp)
            ax6.plot(xx, y_TcpuSetUp, 
                     label=RO_label + "  Reg: A[{:4.2e}] B[{:4.2e}]".format(A,B))
            ax6.errorbar(xx, y_TcpuSetUp, yerr=err_TcpuSetUp_s, 
                         fmt = '.', elinewidth=1.5, capsize=7, color='k')
            ax6.set_xlabel('sp dim $^3$')
            ## add the sp-dimensions involved in the calculation (no scaling) -----
            ax62 = ax6.twiny()
            ax62.set_xlim(ax6.get_xlim())
            ax62.set_xticks(xx)
            ax62.set_xticklabels([f"{x_mz[a]}:{a}" for a in x_sp])
            ax62.set_xlabel("MZ and sp dimension")
            ## --------------------------------------------------------------------
        
    ## organize time per iteration by the sp_(MZ)
    for mz in sorted(mz_set):
        x, y = [], []
        for ro_ in sorted(RO_set):
            vals = data_summary[MODE2PRINT].get(ro_)
            if vals:
                vals = vals.get(mz)
            if vals:
                x.append(vals[3])
                y.append(vals[4]) # cpu Time
        A,B = linear_regression(x, y)
        ax2.plot(x, y, '*--', label="Mz={}  Reg: A[{:4.2e}] B[{:4.2e}]".format(mz,A,B))
    
    labels_RadAng_integ = [f"  R,Om:{RO_}" for RO_ in sorted(RO_set)]
    for i in range(len(x)): 
        ax2.text(x[i], y[i], labels_RadAng_integ[i], rotation='vertical')
    
    _label_sppow = {
        'base': 'sp dim $^3$', 'noRea': 'sp dim $^3$', 'full': 'sp dim $^{2.5}$'}
    ax1.legend()
    ax1.set_title (f"CPU time per iteration (s) [{MODE2PRINT} calculation]")
    ax1.set_xlabel(_label_sppow[MODE2PRINT])
    
    ax2.legend()
    ax2.set_title (f"CPU time per iteration (s) [{MODE2PRINT} calculation] ") #+_label_sppow[MODE2PRINT])
    ax2.set_xlabel(" ( RO_dim ) ")
    
    X, Y = [i[0] for i in trends_MZ_RO_list], [i[1] for i in trends_MZ_RO_list]
    A,B = linear_regression(X, Y)
    ax5.plot(X, Y,  'r*--')
    ax5.set_title(f"Trend of CPU time by sp_dim and RO:\n A= {A:4.2e}   B= {B:+4.2e}")
    ax5.set_ylabel(" t_cpu_iter / {} (s) ".format(_label_sppow[MODE2PRINT]))
    ax5.set_xlabel("  RO_dim  ")
    labels_RadAng_integ = [f"  {x}" for x in data_summary[MODE2PRINT].keys()]
    for i in range(len(x)): ax5.text(X[i], Y[i], labels_RadAng_integ[i])
    
    _label_sppow = {
        'base' : " ( sp_dim )$^3$ ", 'full' : " ( RO_dim * sp_dim )$^2$ ", 
        'noRea': " ( RO_dim * sp_dim )$^2$ "}
    ax3.legend()
    ax3.set_title (f"RAM memory usage (MB) [{MODE2PRINT} calculation]")
    ax3.set_xlabel(_label_sppow[MODE2PRINT])
    
    ax4.legend()
    ax4.set_title (f"error = A - integral <dens(r)> [{MODE2PRINT} calculation]")
    ax4.set_xlabel(" sp_dim ")
    
    ax6.legend()
    ax6.set_title("Estimation of Set Up (CPU) Time by ROmega")
    ax6.set_ylabel("Total SetUp Time (s)")
    
    
    fn_names = []
    from pypdf import PdfMerger
    merger = PdfMerger()
    figures_dict = {'tcpuPi_by_spdim':fig1, 'tcpuPi_byRO':fig2, 
                    'ramUsage_by_spdimRO':fig3, 'densErr_byRO':fig4, 
                    'tpItrend_by_spdimAndRO':fig5, 'setUpTime_byspdim':fig6  }
    for title, f_ in figures_dict.items():
        f_.tight_layout()
        if EXPORT_PDF_IMG:
            fn_names.append(f'{title}.pdf')
            f_.savefig(fn_names[-1])
            merger.append(fn_names[-1])
        f_.show()
    if EXPORT_PDF_IMG:
        merger.write(f"results_{IMPORT_FILENAME}_{MODE2PRINT}.pdf")
        
    _ = 0


if __name__ == '__main__' and os.getcwd().startswith('C'):
    
    
    #===========================================================================
    # PLOT RESULTS  
    #===========================================================================
    # Summary results is a non  conventional export_results file, so there is
    # the plotter specific for the CPU - RAM - dimensions here.
    
    ## Files to plot
    ## TODO: set to print 
    MODE2PRINT = 'base'
    # MODE2PRINT = 'full'
    
    DATA_FOLDER = '../DATA_RESULTS/TestLoading/'
    # __plotEfficiencyShitFromFilesInFolder(DATA_FOLDER, __SUMMARY_FILENAME, MODE2PRINT, EXPORT_PDF_IMG=True)
    
    #===========================================================================
    ##  Upgrading the results by using the single results 
    ##  from Data results collections (statistics etc). 
    #===========================================================================
    
    __plotEfficiencyShitFromFilesInFolder(DATA_FOLDER, __RESULTS_FILENAME, MODE2PRINT,
                                          EXPORT_PDF_IMG=True)