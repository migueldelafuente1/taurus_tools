# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:28:25 2022

@author: Miguel
"""

from collections import OrderedDict
from copy import copy, deepcopy
from datetime import datetime
import os
from random import random
import shutil
import subprocess

from _legacy.exe_isotopeChain_taurus import DataTaurus
from _legacy.exe_q20pes_taurus import _executeProgram, _setDDTermInput, _exportResult
from _legacy.exe_q20pes_taurus import template, template_DD_input, Template, TemplateDDInp
import math as mth
import matplotlib.pyplot as plt
import numpy as np


nucleus = [
    # (6, 6), 
    # (6, 8),
    # (6, 10),
    
    (10, 6),
    # (10, 8),
    # (10, 10),
    # (10, 12),
    # (10, 14),
]

template_sho = """  D1S no density . ME evaluated: Brink_Boeker + Coulomb. Shell(SD+S+P+PF)
4
{}
	0 0
2 12.799713114213203
"""
#%% inthernal methods 

def _remove_states_with_sp(sp_clear, rem_sts, filename, hamilf_out):
    """ Removes all matrix elements blocks with a certain sp state. """
    try:
        rem_sts.remove(sp_clear)
    except ValueError:
        pass
    
    # set SHO
    aux  = "	{} {}".format(len(rem_sts), ' '.join(rem_sts)) 
    text = template_sho.format(aux)
    with open(hamilf_out+'.sho', 'w+') as f:
        f.write(text)
    
    # set .2b
    _reset_hamil_files(sp_clear, filename, hamilf_out, False)
    # set .COM
    _reset_hamil_files(sp_clear, filename, hamilf_out, True)



def _reset_hamil_files(sp_clear, filename, hamilf_out, com_file):
    
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
        
        if sp_clear in sts_head:
            continue
        else:
            final_.append(' 0 5 ' + sp_block)
    
    final_ = title + ''.join(final_)
    with open(hamilf_out + extension, 'w+') as f:
        f.write(final_)
    


def test_execution(z, n, interaction, 
                   repetions=3, iterations=50, integrDims=(6,6),
                   files = None):
    """ 
    Process to execute certain stepes N times to get the time per iteration.
    * D1S_t0 + DD + Rea
    * D1S_t0 + DD
    * D1S_t0  (to get a reference for the interaction)
    """
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    
    RDIM_0, OMEG_0  = integrDims[0], integrDims[1]
    output_filename = 'output_filename'
    file_full   = DataTaurus.BU_folder+'/results_FullD1S.txt'
    file_norea  = DataTaurus.BU_folder+'/results_noRea.txt'
    file_b      = DataTaurus.BU_folder+'/results_base.txt'
    if files:
        file_full, file_norea, file_b = files[0], files[1], files[2]
    
    DOFILE = [False, False, True] ## TODO: Change to not do every process
    if False in DOFILE:
        print(" [WARNING!] There are modes that will not be executed in the"
              "iteration :", DOFILE)
    
    b20_base = 0.05
    
    results_full = [None] * repetions
    results_noRea= [None] * repetions
    results      = [None] * repetions
    
    kwargs = {
        Template.interaction : interaction, 
        Template.com : 1,   Template.read_red_hamil : 0, 
        Template.z : z,     Template.n : n,
        Template.seed : 3,
        Template.grad_type : 1,
        Template.grad_tol  : 0.030,
        Template.eta_grad : 0.03,
        Template.iterations : iterations,
        Template.mu_grad  : 0.15,
        Template.b20  : "1 {:5.4}".format(b20_base),
    }
    
    if DOFILE[0]:
        ## D1S_t0 + DD + Rea
        print("   * doing D1S_t0 + DD + Rea")
        _setDDTermInput(True, rdim=RDIM_0, omega=OMEG_0)
        print(HEAD)
        for iter_ in range(repetions):
            b20 = (1 - 2*random()) * b20_base
            kwargs[Template.b20] = "1 {:5.4}".format(b20)
            res = _executeProgram(kwargs, output_filename+'_1', b20, 
                                  save_final_wf=False)
            results_full[iter_] = res
            _exportResult(results_full, file_full)
    if DOFILE[1]:
        ## D1S_t0 + DD
        print("   * doing D1S_t0 + DD")
        _setDDTermInput(True, rdim=RDIM_0, omega=OMEG_0, rea_on=False)
        print(HEAD)
        for iter_ in range(repetions):
            b20 = (1 - 2*random()) * b20_base
            kwargs[Template.b20] = "1 {:5.4}".format(b20)
            res = _executeProgram(kwargs, output_filename+'_2', b20, 
                                  save_final_wf=False)
            results_noRea[iter_] = res
            _exportResult(results_noRea, file_norea)
    if DOFILE[2]:
        ## D1S_t0
        print("   * doing D1S_t0")
        _setDDTermInput(False)
        print(HEAD)
        for iter_ in range(repetions):
            b20 = (1 - 2*random()) * b20_base
            kwargs[Template.b20] = "1 {:5.4}".format(b20)
            res = _executeProgram(kwargs, output_filename+'_3', b20, 
                                  save_final_wf=False)
            results[iter_] = res
            _exportResult(results, file_b)
    
def _get_totalOccupation(rem_sts):
    occ = 0
    for st in rem_sts:
        j = int(st) % 100
        occ += j + 1
    return occ

def _check_total_occupancy_ok(rem_sts, z, n):
    occ = _get_totalOccupation(rem_sts)
    
    if z >= occ or n >= occ:
        print(" z[{}] or n[{}] exceed shell degeneration[{}]"
              .format(z, n, occ))
        return False
    return True

def nPointsLebedevGrid(maxGrid):
    nPoints = [6,14,26,38,50,74,86,110,146,170,194,230,266,
               302,350,434,590,770,974,1202,1454,1730,2030]
    return nPoints[maxGrid - 1]


#%% set up parameters --------------------------------------------------------

z = 6
n = 6

# all_states = "001 101 103 205 10001 203 307 10103 305 10101 409 10205 "\
#                  "407 20001 10203 511 509 10307 10305 20103 20101 613 "\
#                  "10409 20205 611 10407 30001 20203"
# interaction_base  = 'D1S_t0_MZmax6' 

all_states = "001 101 103 205 10001 203 307 10103 305 10101"
interaction_base  = 'D1S_ls0t0_SPSDPF' 

# all_states = "001 101 103 205 10001 203"
# interaction_base  = 'D1S_ls0t0_SPSD'

interaction_cut   = "aux_hamil"

repetitions  = 8
iterations   = 80
integrationDims = (4, 4) # (R, omega)
    
output_filename = 'aux_output'
# DataTaurus.BU_folder = "BU_test_loading/Base_MZ6"
# DataTaurus.BU_folder = "BU_test_loading/R4O4_SPSDPF"
# DataTaurus.BU_folder = "BU_test_loading/R6O6_SPSDPF"
# DataTaurus.BU_folder = "BU_test_loading/R8O10_SPSDPF"
# DataTaurus.BU_folder = "BU_test_loading/R10O14_SPSDPF"

## --------------------------------------------------------------------------
    
if __name__ == "__main__" and os.getcwd().startswith('C:'):
    
    ## statistics for the results in WINDOWS
    all_data_ROmegadimens = {}
    all_data_spbase = {}
    all_data_sppow2 = {}
    all_data_results_time = {}
    all_data_results = {}
    
    folders = (
        "BU_test_loading/R4O4_SPSDPF",
        "BU_test_loading/R6O6_SPSDPF",
        "BU_test_loading/R8O10_SPSDPF",
        "BU_test_loading/R10O14_SPSDPF",
        # "BU_test_loading/R0O0_Base_MZ6",
               )
    

    
    # plt.figure()
    fig1, ax1 = plt.subplots()
    
    linear_slopes_time = {'results_base':{}, 'results_FullD1S':{}, 'results_noRea':{}}
    
    sppow_pow = {'results_base' :   lambda xx: xx**3., 
                  'results_noRea':   lambda xx: xx**3.66666,# * np.log(xx), 
                  'results_FullD1S' :lambda xx: xx** 5.}
    xlabel_pow = {'results_base' :  'sp dimension ^ 3',  
                   'results_noRea':  'sp dimension ^ 3.666',  #Ln(sp dimension)
                   'results_FullD1S':'sp dimension ^ 5'}
    for folder in folders:
        label_ = folder.split('/')[1].split('_')[0]
        Rdim, Omega = label_.replace('R','').replace('O',' ').split()
        Rdim, Omega = int(Rdim), int(Omega)
        Adim  = nPointsLebedevGrid(Omega)
        total = Adim * Rdim
        all_data_ROmegadimens[folder] = [label_, Rdim, Omega, Adim, total]
        print(" *** ", label_,"*** ")
        
        spbase_shell = {'results_base':[], 'results_FullD1S':[], 'results_noRea':[]}
        sppow2_shell = {'results_base':[], 'results_FullD1S':[], 'results_noRea':[]}
        
        results      = {'results_base':[], 'results_FullD1S':[], 'results_noRea':[]}
        
        elems_       = {'mean': [], 'desv': []}
        results_time = {'results_base'    : deepcopy(elems_), 
                        'results_FullD1S' : deepcopy(elems_), 
                        'results_noRea'   : deepcopy(elems_)}
        
        sts_base = all_states.split()
        sts_base.append('')
        rem_sts = copy(sts_base)
        stop_ = False
        for i in range(len(sts_base)-1, -1, -1):
            
            st = sts_base[i]
            print(i, st)
            
            rem_sts.remove(st)
            
            for file_txt in results_time.keys():
                aux_st = '_no'+str(st) if st else st
                file_ = folder+'/'+file_txt+aux_st+'.txt'
                results[file_txt].append([])
                
                if os.path.exists(file_):
                    data = []
                    with open(file_, 'r') as f:
                        data = f.readlines()
                        if len(data) == 0:
                            print(" file ", file_, "is empty, continue")
                            continue
                        
                        dim_ = _get_totalOccupation(rem_sts)
                        spbase_shell[file_txt].append(dim_)
                        sppow2_shell[file_txt].append(sppow_pow[file_txt](dim_))
                    
                    times_ = []
                    for line in data:
                        res = DataTaurus(None, None, None, True)
                        res.setDataFromCSVLine(line)
                        results[file_txt][-1].append(res)
                        times_.append(res.time_per_iter)
                    
                    results_time[file_txt]['mean'].append(np.mean(times_))
                    NoN1 = 1 if len(times_)==0 else len(times_) / (len(times_)-1)
                    desv_ = NoN1**0.5 * np.std(times_)
                    results_time[file_txt]['desv'].append(desv_)
                    
                else:
                    stop_ = True
                    
            if stop_: break
        
        for file_txt in results_time.keys():
            dy = results_time[file_txt]['mean'][0] - results_time[file_txt]['mean'][-1]
            dx = sppow2_shell[file_txt][0] - sppow2_shell[file_txt][-1]
            linear_slopes_time[file_txt][folder] = dy / dx
        
        all_data_spbase[folder]  = deepcopy(spbase_shell)
        all_data_sppow2[folder]  = deepcopy(sppow2_shell)
        all_data_results[folder] = deepcopy(results)
        all_data_results_time[folder] = deepcopy(results_time)
    
        ## ploting times
        
        # file_ = 'results_FullD1S'
        # file_ = 'results_noRea'
        file_ = 'results_base'
        Tot_ = 1.0 # all_data_ROmegadimens[folder][-1]
        
        x  = (all_data_spbase[folder][file_])
        x2 = (all_data_sppow2[folder][file_])
        y    = (np.array(all_data_results_time[folder][file_]['mean']) / Tot_)
        yerr = (all_data_results_time[folder][file_]['desv'])
        
        # plt.semilogy(x, y, 'b--')
        _, = ax1.plot(x2, y, '--', label= "{:7}  A(t_RO/sp^x)={:4.3e}".format(
                                            all_data_ROmegadimens[folder][0], 
                                            linear_slopes_time[file_][folder]))
        ax1.errorbar(x2, y, yerr=yerr, 
                     fmt = '.', elinewidth=1.5, capsize=7, color='k')
        ax1.set_xlabel(xlabel_pow[file_])
        ax1.set_ylabel('time per iteration (s)')
        ax1.set_title(file_)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('Figure_tpspDim_{}.pdf'.format(file_))
    
    #%% ploting time per iterarion compared to ROmega dim
    # file_ = 'results_FullD1S'
    # file_ = 'results_noRea'
    file_ = 'results_base'
    index_ = 3
    
    fig2, ax2 = plt.subplots()
    
    xscale = lambda xx: xx
    xlabel_ = 'Total (Radial x Angular) dimension'
    if file_ == 'results_FullD1S':
        xscale = lambda xx: np.log(xx)#xx**0.5
        xlabel_ = 'Ln (Total (Radial x Angular) dimension)' #  ** (1/2)
    
    x = {}
    y = {}
    yerr = {}
    linear_slopes_intg = {}
    for index_ in (0, 3, 4, 6, 7):
        x[index_] = []
        y[index_] = []
        yerr[index_] = []
        for folder in folders:
            x[index_].append(all_data_ROmegadimens[folder][-1])
            y[index_].append(all_data_results_time[folder][file_]['mean'][index_])
            yerr[index_].append(all_data_results_time[folder][file_]['desv'][index_])
        
        x[index_] = np.array(x[index_])
        x[index_] = xscale(x[index_])
        # x[index_] = np.log(x[index_])
        
        # get slopes
        dy = y[index_][-1] - y[index_][0]
        dx = x[index_][-1] - x[index_][0]
        linear_slopes_intg[index_] = dy /dx
        
        
        _, = ax2.plot(x[index_], y[index_], 
                      label='sp dim= {:2}  A(t_sp/RO^x)={:4.3e}'.format(
                              all_data_spbase[folder][file_][index_], 
                              linear_slopes_intg[index_]))
        ax2.errorbar(x[index_], y[index_], yerr=yerr[index_], 
                     fmt = '.', elinewidth=1.5, capsize=7, color='k')
        
    
    ax2.set_xlabel(xlabel_)
    ax2.set_ylabel('time per iteration (s)')
    ax2.set_title(file_)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('Figure_tpIntegrDim_{}.pdf'.format(file_))
    
    #%% plot of the Time-SP_dim^x slope evolution with ROmega_dim
    plt.figure()
    
    
    def regres(X,Y):
        x, y = np.array(X), np.array(Y)
        n = np.sum((x-np.mean(x)) * (y - np.mean(y)))
        d = np.sum((x - np.mean(x))**2)
        A = n / d
        B = np.mean(y) - (A*np.mean(x))
        return A, B
    
    y = [a for a in linear_slopes_time[file_].values()]
    
    x = [a[4] for a in all_data_ROmegadimens.values()]
    xlabel_ = 'Radial x Angular dimension'
    if file_ == 'results_FullD1S':
        xlabel_ = 'Ln( Radial x Angular dimension)'
        x = [np.log(a) for a in x]
    
    slope = regres(x, y)
    print("Regresion  =", )
    
    plt.plot(x, y, 'o-')
    plt.xlabel(xlabel_)
    plt.ylabel('Linear regresion: t_iteration / N^x')
    plt.title("{}\nRegression = {:5.4e} *f(RO dim) + {:3.2e}".format(file_, *slope))
    plt.tight_layout()
    plt.show()
    plt.savefig('Figure_RegrTimeIntegr_{}.pdf'.format(file_))
    
    
    
else:
    ## execution in LINUX
    
    # Overwrite/create the back up folder
    DataTaurus.setUpFolderBackUp()
    # if os.path.exists(file_):
    #     os.remove(file_)
    
    sts_base      = all_states.split()
    sts_remaining = all_states.split()
    print("[MAIN] The integration dimensions are:", integrationDims)
    print("[MAIN] First execution NumberStates={}\nSTATES:{}\n"
          .format(len(sts_base), all_states))
    _remove_states_with_sp(None, sts_remaining,
                           interaction_base, interaction_cut)
    file_full   = DataTaurus.BU_folder+'/results_FullD1S.txt'
    file_norea  = DataTaurus.BU_folder+'/results_noRea.txt'
    file_b      = DataTaurus.BU_folder+'/results_base.txt'
    files = (file_full, file_norea, file_b)
    ## first the whole space interaction (not removing any state)
    test_execution(z, n, interaction_cut, repetitions, iterations, 
                   integrationDims, files)
    
    ## "001 101 103 205 10001 203 307 10103 305 10101"
    ## "1   2   3   4   5     6   7   8     9   10   "
    sts_base.reverse()
    for sp_clear in sts_base:
        if len(sts_remaining) < 3:
            print("done (stopped at ST: {})".format(sp_clear))
            break
        if len(sts_remaining) < 5:
            repetitions += 3
        elif len(sts_remaining) < 8:
            repetitions += 1
        
        _remove_states_with_sp(sp_clear, sts_remaining,
                               interaction_cut, interaction_cut)
                            # this will reduce the hamiltonian step by step
        
        print("[MAIN] removing {}, lenStatesRemain[{}]\n NEW STATES: {}\n"
                 .format(sp_clear, len(sts_remaining), ' '.join(sts_remaining)))
        if not _check_total_occupancy_ok(sts_remaining, z, n):
            print("[MAIN] Occupancy for z or n exceed shell vacancies: STOP")
            break
        
        file_full   = DataTaurus.BU_folder+'/results_FullD1S_no'+sp_clear+'.txt'
        file_norea  = DataTaurus.BU_folder+'/results_noRea_no'+sp_clear+'.txt'
        file_b      = DataTaurus.BU_folder+'/results_base_no'+sp_clear+'.txt'
        files = (file_full, file_norea, file_b)
        test_execution(z, n, interaction_cut, repetitions, iterations, 
                       integrationDims, files)
        print(" --------------- ")
    
