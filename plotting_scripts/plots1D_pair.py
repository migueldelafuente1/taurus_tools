'''
Created on 13 dic 2024

@author: delafuente
'''
from copy import deepcopy
from pathlib import Path
from tools.inputs import InputTaurus
from tools.data import DataTaurus, DataTaurusPAV, EigenbasisData
from tools.helpers import elementNameByZ, OUTPUT_HEADER_SEPARATOR,\
    getVariableInLatex
from tools.plotter_levels import MATPLOTLIB_INSTALLED

import os
import numpy as np

HEADER_HAMIL = 'HAMIL_DECOMP_'
HAMILT_BENCH = 'HAMIL_DECOMP_bench'

__MARKERS__byPcnstr = {
    InputTaurus.ConstrEnum.P_T00_J10  : ('*', 8),
    InputTaurus.ConstrEnum.P_T10_J00  : ('.', 8),
    InputTaurus.ConstrEnum.P_T1p1_J00 : ('+', 10),
    InputTaurus.ConstrEnum.P_T1m1_J00 : ('_', 10),
}
if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import MaxNLocator
    
    # Enable LaTeX in Matplotlib
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    
def convertPairingShortcuts(constraints):
    if not isinstance(constraints, (list, tuple)): constraints = [constraints, ]
    final = []
    for constraint in constraints:
        assert constraint in InputTaurus.ConstrEnum.members(), "Invalid argument"
        final.append( constraint.replace('_',''))
    return final

def get_old_constrained_value(obj, constr, indx):
    """ old format has no header """
    pair_val = getattr(obj, constr)
    key_ = f"{indx:4}: {pair_val:8.5f}"
    return key_
    
def get_old_index_for_tes(object_data):
    """ get the index of the energy minimum for old format (index-less)"""
    energies = [getattr(o, 'E_HFB') for o in object_data]
    indx_, e_min = 0, 999999
    for i, e in enumerate(energies):
        if e < e_min: indx_, e_min = i, e
    return indx_

def get_cannonical_basis_from_results(constr, fld,  object_data, cannonicalBas, i_0):
    """
    read the folder for cannoncial basis files, select the index according to 
    the state selected in the export file (object data) by reading the 'res_' 
    files in the same folder
    """
    res_files = filter(lambda x: x.startswith('res_'), os.listdir(fld))
    res_files = filter(lambda x: not ('unconv' in x or 'brok' in x), res_files)
    canbas_files = filter(lambda x: x.startswith('eigenbasis_h'), os.listdir(fld))
    canbas_files = filter(lambda x: not ('unconv' in x or 'brok' in x), canbas_files)
    res_files, canbas_files = list(res_files), list(canbas_files)
    
    if len(canbas_files)==0 : 
        print(f" [WARNING] Cannonical basis not found for [{constr}] SKIP")
        return False
    
    zn   = res_files[0].split('_')[1]
    z, n = [int(x) for x in zn[1:].split('n')]
    def_indx = [i - i_0 for i in range(len(object_data[constr]))]
    for i0, i in enumerate(def_indx):
        head = f'res_{zn}_d{i}_'
        ress = list(filter(lambda x: x.startswith(head), res_files))
        res0 = object_data[constr][i0]
        cfn  = None
        for rfn in ress:
            res = DataTaurus(z, n, fld / rfn)
            if abs(res0.E_HFB - res.E_HFB) > 1.0e-5: continue
            cfn = rfn.replace('res', 'eigenbasis_h').replace('OUT', 'dat')
                    
        if not cfn: 
            print(f" [ERR] can-bas file for [{i}] not found")
            continue
        else:
            can = EigenbasisData(fld / cfn)
            cannonicalBas[constr].append(deepcopy(can))
    return True

def plotVAP_pairCoupling_FromFolders(folders_2_import, MAIN_FLD_TEMP, 
                                     constraints_pair, observables2plot, 
                                     hamil_decompos=True):
    """
    Plot the different constraints related to pairing 
    """
    OLD_FMT = False
    global INTER
    global GLOBAL_TAIL_INTER
    
    FOLDER2SAVE = '/'.join(MAIN_FLD_TEMP.split('/'))
    
    ## SHIT TO DO CAUSE ITS REFUSING TO READ RELATIVELY THE EXPORT FILES, F*U Guido
    cwd_ = Path(os.getcwd()) 
    if '..' in MAIN_FLD_TEMP:
        MAIN_FLD_TEMP = Path(cwd_).parent / Path(MAIN_FLD_TEMP.replace('../',''))
    else:
        MAIN_FLD_TEMP = Path(cwd_) / Path(MAIN_FLD_TEMP)
    MAIN_FLD_TEMP = str(MAIN_FLD_TEMP).replace('\\', '/')
    
    
    for folder_args in folders_2_import:
        z, n = folder_args[0]
        _core_0 = 0
        z_real, n_real = z + _core_0, n + _core_0
        nucl = (f"{z_real+n_real}{elementNameByZ[z_real]}", 
                f"$^{{{z_real+n_real}}}${elementNameByZ[z_real]}")
        
        print(f"[{z}, {n}]   ******** ")
        
        folder_templ = folder_args[1] # '{MAIN_FLD}/BU_folder_{constrs}_{INTER}_z{z}n{n}/'
        export_templ = folder_args[2] # 'export_TES2_{}_z{}n{}_{}'
        
        data_cnstr_plot = {}
        deforms = {}
        hamiltonians = {}
        data_hamils = {}
        deforms_hamils = {}
        object_hamils  = {}
        object_data    = {}
        index_0 = {}
        cannonicalBas = {}
        for constr in constraints_pair:
            data_cnstr_plot[constr] = dict([(v, []) for v in observables2plot])
            deforms    [constr]     = []
            data_hamils[constr]     = {}
            deforms_hamils[constr]  = {}
            object_data   [constr]  = []
            object_hamils [constr]  = {}
            hamiltonians  [constr]  = []
            index_0[constr] = 0
            cannonicalBas [constr]  = []
            
            cnst2 = constr.replace('_','')
            fld = folder_templ.format(MAIN_FLD=MAIN_FLD_TEMP, constrs=cnst2, 
                                      INTER=INTER, z=z, n=n)
            fld = Path(fld)
            if OLD_FMT:
                export_fn = export_templ.format(z, n, INTER, cnst2) # old data
            else:
                export_fn = export_templ.format(cnst2, z, n, INTER) # general
            with open(fld / export_fn, 'r') as f:
                lines = f.readlines()
                if not OLD_FMT: 
                    _, cnst2 = lines[0].strip().split(', ')
                    assert constr == cnst2, f"Imported export file inconsistent with constrait [{constr}/{cnst2}]"
                for i, line in enumerate(lines[1:]):
                    if OLD_FMT:
                        csv_ = line
                    else:
                        key_, csv_ = line.split(OUTPUT_HEADER_SEPARATOR)
                    obj = DataTaurus(z, n, None, True)
                    obj.setDataFromCSVLine(csv_)
                    
                    if OLD_FMT: key_ = get_old_constrained_value(obj, constr, i)
                    
                    key_, defs_ = key_.split(': ')
                    if key_.strip() == '0': index_0[constr] = i
                    deforms[constr].append(float(defs_))
                    for var in observables2plot:
                        var2 = var
                        if var == 'pair':
                            if constr[:5] in ('P_T10', 'P_T00'):
                                var2 = 'pair_pn'
                            else: 
                                var2 = 'pair_pp' if constr==InputTaurus.ConstrEnum.P_T1m1_J00 else 'pair_nn'
                        data_cnstr_plot[constr][var].append(getattr(obj, var2))
                    object_data[constr].append(deepcopy(obj))
            
            if OLD_FMT: index_0[constr] = get_old_index_for_tes(object_data[constr])
            
            # canbasFiles = get_cannonical_basis_from_results(constr, fld, object_data, 
            #                                                 cannonicalBas, index_0[constr])
            canbasFiles = None
            
            ## get the data from hamil decomposition
            hamils_ = filter(lambda x: os.path.isdir(fld / x) and x.startswith(HEADER_HAMIL),
                             os.listdir(fld))
            for h_fld in hamils_:
                hamil = h_fld.replace(HEADER_HAMIL, '')
                hamiltonians[constr].append(hamil)
                data_hamils[constr][hamil] = dict([(v, []) for v in observables2plot])
                deforms_hamils[constr][hamil] = []
                object_hamils [constr][hamil] = []
                cnst2 = constr.replace('_','')
                
                export_fn = f"{h_fld}/" + export_templ.format(cnst2, z, n, hamil)
                
                if not os.path.exists(fld / export_fn):
                    print(" ERROR, cannot access export_fn=", cnst2, export_fn)
                    continue
                with open(fld / export_fn, 'r') as f:
                    lines = f.readlines()
                    _, cnst2 = lines[0].strip().split(', ')
                    assert constr == cnst2, f"Imported export file inconsistent with constrait [{constr}/{cnst2}]"
                    for line in lines[1:]:
                        key_, csv_ = line.split(OUTPUT_HEADER_SEPARATOR)
                        obj = DataTaurus(z, n, None, True)
                        obj.setDataFromCSVLine(csv_)
                        
                        key_, defs_ = key_.split(': ')
                        deforms_hamils[constr][hamil].append(float(defs_))
                        for var in observables2plot:
                            data_hamils[constr][hamil][var].append(getattr(obj, var))
                        object_hamils [constr][hamil].append(deepcopy(obj))
            hamiltonians[constr].sort()
            if 'bench' in hamiltonians[constr]:
                del hamiltonians[constr][hamiltonians[constr].index('bench')]
                hamiltonians[constr].append('bench')
            
        
        ## Plot the main constraints
        colors = {
            InputTaurus.ConstrEnum.P_T00_J10: 'blue',
            InputTaurus.ConstrEnum.P_T10_J00: 'red',
            InputTaurus.ConstrEnum.P_T1m1_J00: 'green',
            InputTaurus.ConstrEnum.P_T1p1_J00: 'black',}
        for var in observables2plot:
            fig = plt.figure()
            ax  = fig.add_subplot(111)
            for constr, x in deforms.items():
                kwargs = {'marker':     __MARKERS__byPcnstr[constr][0],
                          'markersize': __MARKERS__byPcnstr[constr][1], }
                ax.plot(x, data_cnstr_plot[constr][var], '--', 
                        label=getVariableInLatex(constr), linewidth=2,  
                        color=colors.get(constr, None), **kwargs)
                ax.scatter(deforms[constr][index_0[constr]], 
                           data_cnstr_plot[constr][var][index_0[constr]], 
                           marker='o', color=colors.get(constr, None),
                           s=70)
            ax.set_xlabel('Pair. Coup. value $\delta^{JT}$', fontsize=20)
            # ax.set_ylabel(getVariableInLatex(var), fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=22)  # For major ticks
            ax.tick_params(axis='both', which='minor', labelsize=22)  # For minor ticks (if applicable)

            # ax.set_title (var+nucl[1])
            if var.startswith('pair'):
                ax.text(0.50, 0.90, nucl[1], size=30, family='sans-serif',
                        ha='left', va='top', transform=ax.transAxes)
            else:
                ax.text(0.50, 0.90, nucl[1], size=30, family='sans-serif',
                        ha='left', va='top', transform=ax.transAxes)
            if ((z, n) == (18, 18)): ax.legend(fontsize=22)
            ax.set_xlim( (-0.05, max(deforms[constr]) + 0.05))
            fig.tight_layout()
            fig.savefig(FOLDER2SAVE+f"/{var}-{nucl[0]}.pdf")
        ## PLOT the hamiltonian contributions for each variable to plot
        if hamil_decompos:
            # hamiltonians['long-range Gaussian'] = 1
            markers = '*.o+P^vs'*7
            for constr in constraints_pair:
                for var in observables2plot:
                    y2com = np.array(data_hamils[constr]['void'][var])
                    
                    fig = plt.figure()
                    ax  = fig.add_subplot(111)
                    for i, hamil in enumerate(hamiltonians[constr]):
                        #if hamil == 'bench': continue
                        x = deforms_hamils[constr][hamil]
                        y = data_hamils[constr][hamil][var]
                        
                        if var.startswith('pair') and not hamil in ('void', 'bench'): 
                            y = np.array(y) - y2com
                        ax.plot(x, y, marker=markers[i], label=hamil, linestyle='--')
                    
                    ax.plot(deforms[constr], data_cnstr_plot[constr][var],
                            label=INTER, color=colors.get(constr, None))
                    ax.set_xlabel('Pair. Coup. value $\delta^{JT}$')
                    ax.set_ylabel(var)
                    ax.set_xticklabels(ax.get_xticks(), fontsize=14)
                    ax.set_yticklabels(ax.get_yticks(), fontsize=14)
                    ax.set_title(f"{constr}-{var} for {INTER} decomposition. {nucl[1]}")
                    ax.legend(
                        )
                    fig.savefig(FOLDER2SAVE+f"/allTerms_{var}_{constr}_{nucl[0]}.pdf")
            
            process_EnergyPairing(constraints_pair, hamiltonians, deforms_hamils, 
                                  object_hamils, FOLDER2SAVE, nucl)
        
        ## PLOT SP-STATES EVOLUTION BY CONSTR
        if canbasFiles:
            plot_eigenbasis_h_states(constraints_pair, deforms, cannonicalBas,
                                     index_0, FOLDER2SAVE, nucl)
    plt.show()

def process_EnergyPairing(constraints_pair, hamils, b20_by_hamil, obj_hamils,
                          FOLDER2SAVE, nucl):
    """
    Process the contributions considering certain issues from the energy
    """
    com2b_pp   = dict([(cntr, []) for cntr in constraints_pair])
    LRgauss_pp = dict([(cntr, []) for cntr in constraints_pair])
    SRgauss_pp = dict([(cntr, []) for cntr in constraints_pair])
    couLS_pp   = dict([(cntr, []) for cntr in constraints_pair])
    total_pp   = dict([(cntr, []) for cntr in constraints_pair])
    bench_pp   = dict([(cntr, []) for cntr in constraints_pair])
    
    e_pair     = [{}, {}, {}, {}]  ## pp, nn, pn, total
    # deltas     = [{}, {}, {}, {}]
    colors = {
        InputTaurus.ConstrEnum.P_T00_J10: 'blue',
        InputTaurus.ConstrEnum.P_T10_J00: 'red',
        InputTaurus.ConstrEnum.P_T1m1_J00: 'green',
        InputTaurus.ConstrEnum.P_T1p1_J00: 'black',}
    markers = '.o*po'
    hamils_2plot = ()
    for constr in constraints_pair:
        for i in range(4): e_pair[i][constr] = {}
        for h in hamils[constr]:
            e_pair[0][constr][h] = [obj.pair_pp for obj in obj_hamils[constr][h]]
            e_pair[1][constr][h] = [obj.pair_nn for obj in obj_hamils[constr][h]]
            e_pair[2][constr][h] = [obj.pair_pn for obj in obj_hamils[constr][h]]
            e_pair[3][constr][h] = [obj.pair    for obj in obj_hamils[constr][h]]
            ## 
            for i in range(4): 
                e_pair[i][constr][h] = deepcopy(np.array(e_pair[i][constr][h]))
        
        com2b_pp  [constr] = e_pair[3][constr]['void']
        SRgauss_pp[constr] = e_pair[3][constr]['gauss1Wigner'] + e_pair[3][constr]['gauss1Majorana']
        LRgauss_pp[constr] = e_pair[3][constr]['gauss2Wigner'] + e_pair[3][constr]['gauss2Majorana']
        couLS_pp  [constr] = e_pair[3][constr]['coulomb']      + e_pair[3][constr]['spinOrbit']
        total_pp  [constr] = e_pair[3][constr]['bench']
        
        SRgauss_pp[constr] = SRgauss_pp[constr] - 2*com2b_pp[constr]
        LRgauss_pp[constr] = LRgauss_pp[constr] - 2*com2b_pp[constr]
        couLS_pp  [constr] = couLS_pp  [constr] - 2*com2b_pp[constr]
        bench_pp  [constr] = SRgauss_pp[constr] + LRgauss_pp[constr] + \
                             couLS_pp  [constr] + com2b_pp[constr]
        
        hamisDict = {
            'LR-gaussian': LRgauss_pp[constr], 'SR-gaussian': SRgauss_pp[constr], 
            'Coul+LS': couLS_pp [constr],      'com-2b':com2b_pp  [constr], 
            'bench': bench_pp  [constr],       'total': total_pp[constr],
        }
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        for i, hamil in enumerate(hamisDict):
            x = b20_by_hamil[constr]['bench']
            y = hamisDict[hamil]
            
            if hamil == 'total':
                ax.plot(x, y, '.-', label=INTER, color=colors.get(constr, None))
            else:
                l, = ax.plot(x, y, marker=markers[i], label=hamil, linestyle='--')
                if hamil == 'bench': l.set_markerfacecolor('none')
            
        ax.set_xlabel('Pair. Coup. value $\delta^{JT}$')
        ax.set_ylabel('$E_{pair}$')
        ax.set_title(f"{constr} for {INTER} decomposition. {nucl[1]}")
        ax.legend()
        fig.savefig(FOLDER2SAVE+f"/terms_pairing_{constr}_{nucl[0]}.pdf")
    plt.show()

def plot_eigenbasis_h_states(constraints_pair, deforms, cannonicalBas, index_0,
                             FOLDER2SAVE, nucl):
    """ Print the evolution of the states 
    TODO: Connect the surfaces."""    
    
    cnstr_idx = {
        InputTaurus.ConstrEnum.P_T00_J10  : (0, 1),
        InputTaurus.ConstrEnum.P_T10_J00  : (1, 0),
        InputTaurus.ConstrEnum.P_T1m1_J00 : (0, 0),
        InputTaurus.ConstrEnum.P_T1p1_J00 : (1, 1),
    }
    
    can : EigenbasisData = None
    # can.
    vars_2_plot = 'h', 'avg_proton', 'avg_jz', 'avg_n'
    for var in vars_2_plot:
        fig, ax = plt.subplots(2, 2, figsize=(6,8))
        for constr in constraints_pair:
            indxPlt   = cnstr_idx[constr]
            x         = deforms[constr]
            canbas_ff = cannonicalBas[constr]
            
            sp_dim = canbas_ff[0].dim
            y_sp = [list() for _ in range(sp_dim)]
            
            
            for can in canbas_ff:
                for i, e in enumerate(getattr(can, var)):
                    y_sp[i].append(e)
            
            ## Plots
            for i in range(sp_dim):
                ax[indxPlt].plot(x, y_sp[i], marker='.', label=f'{i}', linestyle='-')
            
            ax[indxPlt].set_ylabel(getVariableInLatex(constr))
            # ax.legend()
            ax[indxPlt].axvline(deforms[constr][index_0[constr]], color='black')
        ax[1, 0].set_xlabel('Pair. Coup. value $\delta^{JT}$')
        ax[1, 1].set_xlabel('Pair. Coup. value $\delta^{JT}$')
        fig.suptitle(var)
        fig.tight_layout()
        # fig.savefig(FOLDER2SAVE+f"/terms_pairing_{constr}_{nucl[0]}.pdf")
    
def plot_1Bvs2B_deltaOperators(folders_2_import, MAIN_FLD_TEMP, constraints_pair):
    """
    Testing the relation between delta-JT from taurus and my 2B-version
    """
    global INTER, GLOBAL_TAIL_INTER
    for zn, subfolder_temp, export_templ  in folders_2_import:
        z, n = zn
        # subfolder_temp, export_templ = _kwargs
        data = {}
        
        for cons in constraints_pair:
            constrs = cons.replace('_', '')
            fld_  = subfolder_temp.format(MAIN_FLD=MAIN_FLD_TEMP, constrs=constrs, INTER=INTER, z=z, n=n)
            file_ = Path(fld_) / export_templ.format(constrs, z, n, INTER)
            with open(file_, 'r') as f:
                aux = f.readlines()
                _, _imp_cnstr = aux[0].split(', ')
                _imp_cnstr = _imp_cnstr.strip()
                assert _imp_cnstr == cons, "Missmatch between the pairing operators."
                
                data[cons] = [[], [], {},]  ## [index], [x-values], {data values}
                for line in aux[1:]:
                    head, line = line.split(OUTPUT_HEADER_SEPARATOR)
                    
                    int_, val = head.split(': ')
                    int_, val = int(int_), float(val)
                    
                    dat = DataTaurus(z, n, None, empty_data=True)
                    dat.setDataFromCSVLine(line)
                    
                    data[cons][0].append(int_)
                    data[cons][1].append(val)
                    data[cons][2][int_] = dat
        
        # figures
        for cons in constraints_pair:
            indx, x_vals, dat_objs = data[cons]
            
            var_attr = 'var_pn'
            if   cons == InputTaurus.ConstrEnum.P_T1m1_J00:
                var_attr = 'var_p'
            elif cons == InputTaurus.ConstrEnum.P_T1p1_J00:
                var_attr = 'var_n'
            
            attr_2b   = cons.replace('P', 'P2b')
            y2b_vals  = []
            y1b_vals  = []
            yvar_vals = []
            avg_zn = (z + n) / 2
            CASE = 1 ## 2
            for i in indx:
                dat = dat_objs[i]
                
                if CASE == 1:
                    y1b_vals .append( getattr(dat, cons)**2)
                    y2b_vals .append( (avg_zn*getattr(dat, attr_2b)) )
                    yvar_vals.append( getattr(dat, var_attr) )
                elif CASE == 2:
                    y1b_vals .append( getattr(dat, cons) ) 
                    y2b_vals .append( np.sqrt(avg_zn*getattr(dat, attr_2b)) )
                    yvar_vals.append( np.sqrt(getattr(dat, var_attr)) )
            
            fig, ax = plt.subplots(1, 1, figsize=(6,8))
            if CASE == 1:
                ax.plot(x_vals, y1b_vals, '.--', label='(1B-approx)${}^2$', color='black')
                ax.plot(x_vals, y2b_vals, '.--', label='zn * 2B-approx'
                                                        +f'   zn={avg_zn:3.2f}', color='red')
                ax.plot(x_vals, yvar_vals,'.--', label='$\sigma^2$', color='blue')
            elif CASE == 2:
                ax.plot(x_vals, y1b_vals, '.--', label='1B-approx', color='black')
                ax.plot(x_vals, y2b_vals, '.--', label='(zn * 2B-approx)${}^{1/2}$'
                                                        +f'   zn={avg_zn:3.2f}', color='red')
                ax.plot(x_vals, yvar_vals,'.--', label='$\sqrt{\sigma^2}$', color='blue')
            ax.set_title(cons)
            
            ax.legend()
        plt.show()
    
if __name__ == '__main__':
    
    #===========================================================================
    # ## TESTS FOR THE 2B pair-coupling OPERATORS example24 Mg
    
    INTER = 'B1_MZ5' # 'D1S' # 'P2_MZ4' #P2
    GLOBAL_TAIL_INTER = '_B1' # '_D1S' #
    
    MAIN_FLD = f'../DATA_RESULTS/TestsPJTOperators/_BU_hfbCase'
    subfolder_temp = '{MAIN_FLD}/BU_folder_{constrs}_{INTER}_z{z}n{n}/'
    export_templ = 'export_TES_{}_z{}n{}_{}.txt'
    
    constraints_ = [                     ### DO NOT CHANGE THIS ORDER
        InputTaurus.ConstrEnum.P_T00_J10,
        # InputTaurus.ConstrEnum.P_T00_J1m1,
        # InputTaurus.ConstrEnum.P_T00_J1p1,
        InputTaurus.ConstrEnum.P_T10_J00,
        InputTaurus.ConstrEnum.P_T1m1_J00,
        InputTaurus.ConstrEnum.P_T1p1_J00,
    ]
    nuclei = [
        # ( 2, 2),
        # (10,10),
        (20,20),
    ]
    folders_2_import = [
        ((z, n), subfolder_temp, export_templ) for z,n in nuclei
    ]
    
    plot_1Bvs2B_deltaOperators(folders_2_import, MAIN_FLD, constraints_)
    
    0/0
    #===========================================================================    
    
    MAIN_FLD = '../DATA_RESULTS/SD_Odd_pnPairing/{pair_cnstr}'
    
    # nuclei = [(12,11 + 2*i) for i in range(0, 6)] # 6
    nuclei = [
        # (12,12), (12,13), (12,15),
        # (12,12), # (12,16), (6, 6), 
        # (8, 8), (10, 10), 
        # (16,16), (14,14), (18,18), (20,20)
        # (12,18), 
        # (10,11), (12,13), (8,9), #(16,17), 
        # (12,10), (12,12), (12,14), (12,16), (12,18), (12,20), (12,22), (12,24),
        # (12,26), (12,28)
        # (10,16),
        # (12,12+i) for i in range(7)
        # (12,14)
        # ( 8, 8), (14,14)
        # (11,11), (13,13),
    ] 
    
    INTER = 'B1_MZ5' # 'D1S' # 'P2_MZ4' #P2
    GLOBAL_TAIL_INTER = '_B1' # '_D1S' #
    export_templ = 'export_TES_{}_z{}n{}_{}.txt'
    # export_templ = 'export_PSz{}n{}_{}_{}.txt'
    
    constraints_ = [                     ### DO NOT CHANGE THIS ORDER
        InputTaurus.ConstrEnum.P_T00_J10,
        # InputTaurus.ConstrEnum.P_T00_J1m1,
        # InputTaurus.ConstrEnum.P_T00_J1p1,
        InputTaurus.ConstrEnum.P_T10_J00,
        InputTaurus.ConstrEnum.P_T1m1_J00,
        InputTaurus.ConstrEnum.P_T1p1_J00,
    ]
    observables2plot = [
        'pair', 
        # 'pair_pn',
        #  'pair_pp', 'pair_nn',
        'E_HFB',
        # 'hf',
        # 'hf_pn',
        # 'var_n', 'var_p',
        # 'b20_isoscalar',
        # 'b22_isoscalar',
        # 'b30_isoscalar',
        # 'b32_isoscalar',
        # 'b40_isoscalar',
        # 'b42_isoscalar',
        # 'b44_isoscalar',
        # 'gamma_isoscalar',
        # 'parity',
        # 'Jz',
        # InputTaurus.ConstrEnum.P_T00_J10,
        # InputTaurus.ConstrEnum.P_T10_J00,
        # InputTaurus.ConstrEnum.P_T00_J1m1,
        # InputTaurus.ConstrEnum.P_T1p1_J00,
        # InputTaurus.ConstrEnum.P_T1m1_J00,
    ]
    
    MAIN_FLD = f'../DATA_RESULTS/PN_mixing/hamilParts_S0B1VAP_MZ5'
    subfolder_temp = '{MAIN_FLD}/BU_folder_{constrs}_{INTER}_z{z}n{n}/'
    # old D1S results
    # MAIN_FLD = f'../DATA_RESULTS/PN_mixing/Mg_MZ5' #SDnuclei_MZ5' # 
    # subfolder_temp = '{MAIN_FLD}/' #
    # MAIN_FLD = f'../DATA_RESULTS/PN_mixing/hamilParts_S0M3YP2_MZ4'
    # subfolder_temp = '{MAIN_FLD}/BU_folder_{constrs}_{INTER}_z{z}n{n}/'
    
    folders_2_import = [
        ((z, n), subfolder_temp, export_templ) for z,n in nuclei
    ]
    
    plotVAP_pairCoupling_FromFolders(folders_2_import, MAIN_FLD, constraints_,
                                     observables2plot, hamil_decompos=False)

