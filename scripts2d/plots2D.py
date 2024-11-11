'''
Created on 7 nov 2024

@author: delafuente
'''
import os, shutil
from pathlib import Path
import itertools
import numpy as np

from tools.inputs import InputTaurus
from tools.helpers import elementNameByZ
from tools.data import DataTaurus
from tools.helpers import OUTPUT_HEADER_SEPARATOR
from tools.plotter_levels import MATPLOTLIB_INSTALLED
from copy import deepcopy
if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Enable LaTeX in Matplotlib
    # plt.rcParams.update({
    #     "text.usetex": False,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"],
    # })

INTER = ''
GLOBAL_TAIL_INTER = ''

def getVariableInLatex(var):
    _P_vars = {
        InputTaurus.ConstrEnum.P_T00_J10:  '$\delta^{T=0\ M_T= 0}_{J=1\ M=0}$',
        InputTaurus.ConstrEnum.P_T00_J1m1: '$\delta^{T=0\ M_T=-1}_{J=1\ M=0}$',
        InputTaurus.ConstrEnum.P_T00_J1p1: '$\delta^{T=0\ M_T=+1}_{J=1\ M=0}$',
        InputTaurus.ConstrEnum.P_T10_J00:  '$\delta^{T=1\ M_T=0}_{J=0}$',
        InputTaurus.ConstrEnum.P_T1m1_J00: '$\delta^{pp}_{J=0}$',
        InputTaurus.ConstrEnum.P_T1p1_J00: '$\delta^{nn}_{J=0}$',
    }
    aux = var.split('_')
    if   var.startswith('b') or var.startswith('q') or var.startswith('gamma'):
        aux[0] = f"$\\beta_{{{aux[0][1:]}}}" if var.startswith('b') else f"$Q_{{{aux[0][:1]}}}"
        return aux[0] + f"^{{({aux[1]})}}$"
    elif var.startswith('P_'):
        return _P_vars[var]
    elif var.startswith('E_HFB'):
        if len(aux) == 2: aux.append('(total)')
        return f'$E_{{HFB}}^{{{aux[2]}}}$'
    elif var[:2] in ('ki', 'pa', 'hf'):
        if len(aux) == 1: aux.append('(total)')
        return f"$E_{{{aux[0]}}}^{{{aux[1]}}}$"
    elif var[:3] == 'var':
        return f"$\\Delta^{{{aux[1]}}}$"
    elif var.startswith('J'):
        if var.endswith('_var'): return f"$\Delta\ J_{{{var[1]}}}$"
        else: 
            aux.append('')
            return f"$J_{{{var[1]}}}^{{{aux[1]}}}$"
    elif var.startswith('r_'):
        return f"$r^{{{aux[1]}}}$"
    else:raise Exception("Unimplemented:", var)
        
    
def convertPairingShortcuts(constraints):
    if not isinstance(constraints, (list, tuple)): constraints = [constraints, ]
    final = []
    for constraint in constraints:
        assert constraint in InputTaurus.ConstrEnum.members(), "Invalid argument"
        final.append( constraint.replace('_',''))
    return final

def plotVAPPAV_2d_byConstraintsFromFolders(folders_2_import, MAIN_FLD, 
                                           constr_list, observables2plot,
                                           CONTOUR=True):
    """
    Validate the ordering of the 
    """
    global INTER
    global GLOBAL_TAIL_INTER
    
    FOLDER2SAVE = MAIN_FLD
    shuffle_constr = list(itertools.permutations(constr_list))
    
    for folder_args in folders_2_import:
        FLD = 'BU_folder_B1_MZ3_z2n1'
        EXPORT_FN = 'export_TES2_PT00J10_PT1p1J00_z2n1_B1_MZ3.txt'
        
        z, n = folder_args[0]
        _core_0 = 0
        z_real, n_real = z + _core_0, n + _core_0
        nucl = (f"{z_real+n_real}{elementNameByZ[z_real]}", 
                f"$^{{{z_real+n_real}}}${elementNameByZ[z_real]}")
        
        print(f"[{z}, {n}]   ******** ")
        
        export_templ = folder_args[2] # 'export_TES2_{}_z{}n{}_{}'
        
        possible_export_fn = [convertPairingShortcuts(c) for c in shuffle_constr]
        possible_export_fn = ['_'.join(permcnstr) for permcnstr in possible_export_fn]
        
        fld_  = folder_args[1].format(MAIN_FLD=MAIN_FLD, INTER=INTER, z=z, n=n)
        FLD  = Path(fld_)
        
        possible_export_fn = [export_templ.format(exp,z,n,INTER) for exp in possible_export_fn]
        
        for swipe, EXPORT_FN in enumerate(possible_export_fn):
            ## Swipe indicates the order of the constraints is different from the 
            ## order in the given constr_list, (itertools gives original and then 
            ## the swiped element).
            if not os.path.exists(f"{FLD}/{EXPORT_FN}"): continue
            with open(f"{FLD}/{EXPORT_FN}", 'r') as f:
                lines = f.readlines()
                constr_2 = lines[0].strip().split(', ')
                assert all([c in constr_list for c in constr_2[1:]]), "Missing constraints"
                deforms = {}
                data = dict([(o, {}) for o in observables2plot])
                set_values = [set() for _ in range(len(constr_list))]
                for line in lines[1:]:
                    key_, csv_ = line.split(OUTPUT_HEADER_SEPARATOR)
                    obj = DataTaurus(z, n, None, True)
                    obj.setDataFromCSVLine(csv_)
                    
                    key_, defs_ = key_.split(': ')
                    key_  = [ int(k)  for k in key_.split(', ')]
                    key_  = tuple(key_) 
                    defs_ = [float(x) for x in defs_.split(',')]
                    
                    if swipe:
                        key_ =  key_[1],  key_[0]
                        defs_= defs_[1], defs_[0]
                    
                    for var in observables2plot: 
                        data[var][key_] = getattr(obj, var)
                    deforms[key_] = defs_
                    for i in range(len(constr_list)): set_values[i].add(defs_[i])
            break
        
        # Create grid and compute Z valu
        x0 = sorted(list(set([d[0] for d in deforms.values()])))
        y0 = sorted(list(set([d[1] for d in deforms.values()])))
        x = np.array(x0)
        y = np.array(y0)
        X, Y = np.meshgrid(x, y)
        
        k0,k1 = [],[]
        for k, v in deforms.items():
            k0.append(k[0])
            k1.append(k[1])
        k0_lims = min(k0), max(k0)
        k1_lims = min(k1), max(k1)
            
        for var_plot in observables2plot:
            # Define a function for the surface (for example, a Gaussian)
            Z = 0.0*X + 0.0*Y
            for key_, defs_ in data[var_plot].items():
                i = key_[0] - k0_lims[0]
                j = key_[1] - k1_lims[0]
                ## remember, Z indexing is [y, x]
                Z[j, i] = data[var_plot][key_]
            
            # Plotting the surface
            fig = plt.figure()
            if CONTOUR:
                ax = fig.add_subplot(111)
                filled_contours = ax.contourf(X, Y, Z, levels=20, cmap='viridis')  # Fills contours
                
                # Overlay contour lines
                # Overlay contour lines with custom line widths and styles
                contours = ax.contour(
                    X, Y, Z, levels=20, colors='black',
                    linewidths=0.7,       # Set line thickness
                    linestyles='dashed'   # Set line style
                )
                ax.clabel(contours, inline=True, fontsize=6)  # Label the contour lines
                # Add a color bar for reference
                fig.colorbar(filled_contours, ax=ax)#, label="Z Value")
                
                ax.set_title (f"{getVariableInLatex(var_plot)}  Calculation {INTER}. {nucl[1]}")
            else:
                
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
                # Add a color bar for reference
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                ax.view_init(elev=30, azim=180+45)  # Set elevation and azimuth
                
                ax.set_zlabel(getVariableInLatex(var_plot))
                ax.set_title (f"Calculation  {INTER} 2-pair constraints. {nucl[1]}")

            # Labels and title
            ax.set_xlabel(getVariableInLatex(constr_list[0]))
            ax.set_ylabel(getVariableInLatex(constr_list[1]))
            
            constr_list_2 = [x.replace('_', '') for x in constr_list]
            plt.savefig("{}/{}_{}_{}_{}.pdf".format(FOLDER2SAVE, var_plot, 
                                                    *constr_list_2, nucl[0]))
        # Display the plot
        plt.show()

def plotContourVAPPAV_2dT1T0vsTppnnFromFolders(folders_2_import, MAIN_FLD_TEMP, 
                                               constraints_pn, observables2plot):
    """
    THis plot show every contour for the constraints_pn (pp/nn non valid)
    vs pp and nn pair constraints.
    
    It requires every folder asked for to be and have the data.
    
    TESTING:
        put the constraint constraints_pn for every variable X(T=1,|M|=1) and for 
        the Y(Other variables), plots must match their corresponding axis.
    """
    global INTER
    global GLOBAL_TAIL_INTER
    
    FOLDER2SAVE = '/'.join(MAIN_FLD_TEMP.split('/')[:-1])
    constraints_t1 = [InputTaurus.ConstrEnum.P_T1m1_J00,
                      InputTaurus.ConstrEnum.P_T1p1_J00, ] 
    
    for folder_args in folders_2_import:
        FLD = 'BU_folder_B1_MZ3_z2n1'
        EXPORT_FN = 'export_TES2_PT00J10_PT1p1J00_z2n1_B1_MZ3.txt'
        
        z, n = folder_args[0]
        _core_0 = 0
        z_real, n_real = z + _core_0, n + _core_0
        nucl = (f"{z_real+n_real}{elementNameByZ[z_real]}", 
                f"$^{{{z_real+n_real}}}${elementNameByZ[z_real]}")
        
        print(f"[{z}, {n}]   ******** ")
        
        export_templ = folder_args[2] # 'export_TES2_{}_z{}n{}_{}'
        
        constr_2_plot = {}
        for constr in constraints_pn:
            constr_list = [[constr, c] for c in constraints_t1]
            constr_list = [sorted(c) for c in constr_list]
            possible_export_fn = [convertPairingShortcuts(c) for c in constr_list]
            possible_export_fn = ['-'.join(permcnstr) for permcnstr in possible_export_fn]
            
            constr_2_plot[constr] = deepcopy(possible_export_fn)
        
        data    = {}
        deforms = {}
        for cnst_pn in constr_2_plot:
            main_ = MAIN_FLD_TEMP.format(pair_cnstr=cnst_pn)
            
            possible_export_fn = [
                export_templ.format(exp,z,n,INTER) for exp in constr_2_plot[cnst_pn]]
            
            data[cnst_pn]    = dict([(c, {}) for c in constraints_t1])
            deforms[cnst_pn] = dict([(c, {}) for c in constraints_t1])
            for T, contr_T1 in enumerate(constraints_t1):
                
                ctr_str = constr_2_plot[cnst_pn][T].replace('_','-')
                fld_  = folder_args[1].format(MAIN_FLD=main_, constrs=ctr_str,
                                              INTER=INTER, z=z, n=n)
                FLD  = Path(fld_)
                
                EXPORT_FN = possible_export_fn[T]
                _CORRECT_SORTING = [contr_T1, cnst_pn] ## x for the T=1 variable
                if not os.path.exists(FLD / EXPORT_FN): continue
                with open(FLD / EXPORT_FN, 'r') as f:
                    lines = f.readlines()
                    constr_2 = lines[0].strip().split(', ')
                    swipe = constr_2[1:] != _CORRECT_SORTING
                    
                    # assert all([c in constr_2_plot[cnst_pn] for c in constr_2[1:]]), "Missing constraints"
                    deforms[cnst_pn][contr_T1] = {}
                    data[cnst_pn][contr_T1] = dict([(o, {}) for o in observables2plot])
                    for line in lines[1:]:
                        key_, csv_ = line.split(OUTPUT_HEADER_SEPARATOR)
                        obj = DataTaurus(z, n, None, True)
                        obj.setDataFromCSVLine(csv_)
                        
                        key_, defs_ = key_.split(': ')
                        key_  = [ int(k)  for k in key_.split(', ')]
                        key_  = tuple(key_) 
                        defs_ = [float(x) for x in defs_.split(',')]
                        
                        if swipe:
                            key_ =  key_[1],  key_[0]
                            defs_= defs_[1], defs_[0]
                        for var in observables2plot: 
                            data[cnst_pn][contr_T1][var][key_] = getattr(obj, var)
                        deforms[cnst_pn][contr_T1][key_] = defs_
                
        nrows = 0
        for cnst_pn, dicts in data.items():
            if any(map(lambda x: len(x) > 0, dicts.values())): nrows += 1
        # Plotting the surface
        figs_and_axes = {}
        for var_plot in observables2plot:
            figs_and_axes[var_plot] = plt.subplots(nrows, 2)
        
        for i, cnst_pn in enumerate(data.keys()):
            j = -1
            for contr_T1, vals in data[cnst_pn].items():
                j += 1
                if len(vals) == 0: 
                    print(" SKIP. No data for constraints:", cnst_pn, contr_T1)
                    continue
                # Create grid and compute Z valu
                x0 = sorted(list(set([d[0] for d in deforms[cnst_pn][contr_T1].values()])))
                y0 = sorted(list(set([d[1] for d in deforms[cnst_pn][contr_T1].values()])))
                x = np.array(x0)
                y = np.array(y0)
                X, Y = np.meshgrid(x, y)
                
                k0,k1 = [],[]
                for k, v in deforms[cnst_pn][contr_T1].items():
                    k0.append(k[0])
                    k1.append(k[1])
                k0_lims = min(k0), max(k0)
                k1_lims = min(k1), max(k1)
                    
                for var_plot in observables2plot:
                    # Define a function for the surface (for example, a Gaussian)
                    Z = np.zeros(X.shape) # = 0.0*X + 0.0*Y
                    for key_, defs_ in data[cnst_pn][contr_T1][var_plot].items():
                        ii = key_[0] - k0_lims[0]
                        jj = key_[1] - k1_lims[0]
                        ## remember, Z indexing is [y,x]
                        Z[jj, ii] = data[cnst_pn][contr_T1][var_plot][key_]
                    
                    fig, axx = figs_and_axes[var_plot]
                    # print(f"{i}, {j}", cnst_pn, contr_T1, " :", var_plot)
                    ax = axx[i, j] if nrows > 1 else axx[j]
                    filled_contours = ax.contourf(X, Y, Z, levels=20, cmap='viridis')  # Fills contours
                    
                    # Overlay contour lines with custom line widths and styles
                    contours = ax.contour(
                        X, Y, Z, levels=20, colors='black',
                        linewidths=0.7,       # Set line thickness
                        linestyles='dashed'   # Set line style
                    )
                    ax.clabel(contours, inline=True, fontsize=6)  # Label the contour lines
                    # Add a color bar for reference
                    fig.colorbar(filled_contours, ax=ax)#, label="Z Value")
                    
                    fig.suptitle (f"{getVariableInLatex(var_plot)}  Calculation {INTER}. {nucl[1]}")
        
        for var_plot in observables2plot:
            # Labels and title
            fig, ax = figs_and_axes[var_plot]
            if nrows > 1:
                for j in range(nrows):
                    ax[j,0].set_ylabel(getVariableInLatex(constraints_pn[j] ))
                i = nrows - 1
                ax[i,0].set_xlabel(getVariableInLatex(constraints_t1[0]))
                ax[i,1].set_xlabel(getVariableInLatex(constraints_t1[1]))
            else:
                ax[0].set_ylabel(getVariableInLatex(constraints_pn[j] ))
                ax[0].set_xlabel(getVariableInLatex(constraints_t1[0]))
                ax[1].set_xlabel(getVariableInLatex(constraints_t1[1]))
            fig.tight_layout()
            fig.savefig("{}/{}_contoursPT1T0_{}.pdf".format(FOLDER2SAVE, var_plot, nucl[0]))                            
        plt.show()
    

if __name__ == '__main__':
    #===========================================================================
    # # PLOT FROM FOLDERS
    # NOTE_ Use the constraints to plot in this order (see main_2dpair)
    # constraints_ = {
        # InputTaurus.ConstrEnum.P_T10_J00   : 0.0,
        # InputTaurus.ConstrEnum.P_T1p1_J00  : 0.0,
        # InputTaurus.ConstrEnum.P_T1m1_J00  : 0.0,
        # InputTaurus.ConstrEnum.P_T00_J10   : 0.0,
        # InputTaurus.ConstrEnum.P_T00_J1p1  : 0.0,
        # InputTaurus.ConstrEnum.P_T00_J1m1  : 0.0,
    # }
    #===========================================================================
    
    
    SUBFLD_ = '../BU_folder_B1_MZ3_z2n1/PNAMP/'
    
    MAIN_FLD = '..'
    # MAIN_FLD = '../DATA_RESULTS/SD_Odd_pnPairing/F'
    MAIN_FLD = '../DATA_RESULTS/SD_Odd_pnPairing/{pair_cnstr}'
    
    # nuclei = [(12,11 + 2*i) for i in range(0, 6)] # 6
    nuclei = [(12,13),  ] # (17,12),
    
    
    INTER = 'B1_MZ4'
    GLOBAL_TAIL_INTER = '_B1'
    export_templ = 'export_TES2_{}_z{}n{}_{}.txt'
    
    constraints_ = [
        # InputTaurus.ConstrEnum.P_T10_J00,
        
        InputTaurus.ConstrEnum.P_T1p1_J00,
        # InputTaurus.ConstrEnum.P_T1m1_J00,
        
        InputTaurus.ConstrEnum.P_T00_J10,
        # InputTaurus.ConstrEnum.P_T00_J1p1,
        # InputTaurus.ConstrEnum.P_T00_J1m1,
    ]
    observables2plot = [
        'pair', 
        'E_HFB',
        # 'b20_isoscalar',
        InputTaurus.ConstrEnum.P_T00_J10,
        InputTaurus.ConstrEnum.P_T10_J00,
        InputTaurus.ConstrEnum.P_T1p1_J00,
        InputTaurus.ConstrEnum.P_T1m1_J00,
    ]
    
    MAIN_FLD = f'../DATA_RESULTS/SD_Odd_pnPairing/HFB/{constraints_[0]}'
    folders_2_import = [
        ((z, n), '{MAIN_FLD}/BU_folder_{constrs}_{INTER}_z{z}n{n}/', export_templ) for z,n in nuclei
    ]
    
    # plotVAPPAV_2d_byConstraintsFromFolders(folders_2_import, MAIN_FLD, 
    #                                        constraints_, observables2plot,
    #                                        CONTOUR=True)
    
    constraints_pn = [
        InputTaurus.ConstrEnum.P_T00_J10,
        # InputTaurus.ConstrEnum.P_T00_J1m1,
        # InputTaurus.ConstrEnum.P_T00_J1p1,
        InputTaurus.ConstrEnum.P_T10_J00,
    ]
    MAIN_FLD = '../DATA_RESULTS/SD_Odd_pnPairing/HFB/{pair_cnstr}'
    plotContourVAPPAV_2dT1T0vsTppnnFromFolders(folders_2_import, MAIN_FLD, 
                                               constraints_pn, observables2plot)
    
    
