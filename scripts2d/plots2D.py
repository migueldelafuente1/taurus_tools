'''
Created on 7 nov 2024

@author: delafuente
'''
import os, shutil
from pathlib import Path
import itertools
import numpy as np

from tools.inputs import InputTaurus
from tools.helpers import elementNameByZ, getVariableInLatex
from tools.data import DataTaurus
from tools.helpers import OUTPUT_HEADER_SEPARATOR
from tools.plotter_levels import MATPLOTLIB_INSTALLED
from copy import deepcopy
from matplotlib import ticker

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

INTER = ''
GLOBAL_TAIL_INTER = ''
        
    
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
        
        data_main_dirs, defs_main_dirs = {}, {}
        for const in (*constr_2_plot, *constraints_t1):
            data_main_dirs[const] = {}
            defs_main_dirs[const] = []
            for var in observables2plot:
                data_main_dirs[const][var] = []
        
        nrows = 0
        for cnst_pn, dicts in data.items():
            if any(map(lambda x: len(x) > 0, dicts.values())): nrows += 1
        import matplotlib.gridspec as gridspec

        for indivCs in (1, 0):
            if indivCs == 0 and len(data) != 2: break
            # Plotting the surface
            figs_and_axes  = {}
            for var_plot in observables2plot:
                if indivCs: 
                    figs_and_axes[var_plot] = plt.subplots(nrows, 2, figsize=(7,4)) #(6,5)) # 
                    gs = gridspec.GridSpec(1, 3, width_ratios=[0.5, 1.5, 1])  # Allocate 40% less space to the left

                else:       figs_and_axes[var_plot] = plt.subplots(1,2, figsize=(8,4))
            
            pn_list_data = list(data.keys())
                
            for i, cnst_pn in enumerate(pn_list_data):
                if not indivCs and i == 1: break
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
                    
                    defs_main_dirs[cnst_pn]  = y
                    defs_main_dirs[contr_T1] = x
                    
                    k0,k1 = [],[]
                    for k, v in deforms[cnst_pn][contr_T1].items():
                        k0.append(k[0])
                        k1.append(k[1])
                    k0_lims = min(k0), max(k0)
                    k1_lims = min(k1), max(k1)
                        
                    for var_plot in observables2plot:
                        # Define a function for the surface (for example, a Gaussian)
                        Z = np.zeros(X.shape) # = 0.0*X + 0.0*Y
                        v0_min = 0
                        # if var_plot.startswith('E_HFB'):
                        #     v0_min = min(data[cnst_pn][contr_T1][var_plot].values())
                        for key_, defs_ in data[cnst_pn][contr_T1][var_plot].items():
                            ii = key_[0] - k0_lims[0]
                            jj = key_[1] - k1_lims[0]
                            ## remember, Z indexing is [y,x]
                            val = data[cnst_pn][contr_T1][var_plot][key_]
                            if indivCs:
                                Z[jj, ii] = val - v0_min
                                
                                if ii == 0 and j == 0: data_main_dirs[cnst_pn] [var_plot].append(val - v0_min)
                                if jj == 0 and i == 0: data_main_dirs[contr_T1][var_plot].append(val - v0_min)
                            else:
                                cnst_pn2  = pn_list_data[1] 
                                Z[jj, ii] = val - data[cnst_pn2][contr_T1][var_plot][key_]
                        
                        fig, axx = figs_and_axes[var_plot]
                        # print(f"{i}, {j}", cnst_pn, contr_T1, " :", var_plot)
                        ax = axx[i, j] if (nrows > 1 and indivCs) else axx[j]
                        filled_contours = ax.contourf(X, Y, Z, levels=20, cmap='plasma')  # Fills contours
                        
                        # Overlay contour lines with custom line widths and styles
                        contours = ax.contour(
                            X, Y, Z, levels=15, colors='black',
                            linewidths=0.7,       # Set line thickness
                            linestyles='dashed'   # Set line style
                        )
                        ax.clabel(contours, inline=True, fontsize=10)  # Label the contour lines
                        # Add a color bar for reference
                        cbar = fig.colorbar(filled_contours, ax=ax)#, label="Z Value")
                        cbar.locator = ticker.MaxNLocator(nbins=5)
                        cbar.ax.tick_params(labelsize=14)
                        ax.xaxis.set_major_locator(MaxNLocator(3))  # Set maximum 5 ticks on x-axis
                        ax.yaxis.set_major_locator(MaxNLocator(3))  # Set maximum 4 ticks on y-axis
                        ax.set_xlim(0, max(x))
                        ax.set_ylim(0, max(y))
                        ax.tick_params(labelsize=16)
                        
                        kw = {'fontsize': 20} # 15
                        if indivCs: ax.set_aspect('equal', adjustable='box')
                        int_str = INTER
                        int_str = "${}^{{({})}}$".format(*INTER.split('_'))
                        # if indivCs: 
                        #     # fig.suptitle (f"{getVariableInLatex(var_plot)}  {int_str}   {nucl[1]}", **kw)
                        #     fig.suptitle(f"{getVariableInLatex(var_plot)}   {nucl[1]}", **kw)
                        # else:
                        #     args = [getVariableInLatex(x) for x in 
                        #             (var_plot, cnst_pn, var_plot, cnst_pn2)]
                        #     fig.suptitle ("{}({}) - {}({})  ".format(*args) + 
                        #                   f"         ({nucl[1]})", **kw)
                        #     # fig.suptitle(f"{getVariableInLatex(var_plot)}    {nucl[1]}", **kw)
            kw = {'fontsize': 20}
            for var_plot in observables2plot:
                # Labels and title
                fig, ax = figs_and_axes[var_plot]
                if nrows > 1 and indivCs:
                    for j in range(nrows):
                        ax[j,0].set_ylabel(getVariableInLatex(constraints_pn[j] ),**kw)
                    i = nrows - 1
                    ax[i,0].set_xlabel(getVariableInLatex(constraints_t1[0]),**kw)
                    ax[i,1].set_xlabel(getVariableInLatex(constraints_t1[1]),**kw)
                else:
                    # ax[0].set_ylabel(getVariableInLatex(constraints_pn[j-1] ))
                    ax[0].set_ylabel('$\delta^{pn}_{J(M=0)}$', **kw)
                    ax[0].set_xlabel(getVariableInLatex(constraints_t1[0]),**kw)
                    ax[1].set_xlabel(getVariableInLatex(constraints_t1[1]),**kw)
                fig.tight_layout()
                
                if indivCs: 
                    fig.savefig("{}/{}_contoursPT1T0_{}.pdf".format(FOLDER2SAVE, var_plot, nucl[0]))
                else:
                    fig.savefig("{}/{}_diff_contoursPT1T0_{}.pdf".format(FOLDER2SAVE, var_plot, nucl[0]))
        # plt.show()
        ## plots Main delta directions
        kw2 = {
            InputTaurus.ConstrEnum.P_T1m1_J00:{'color':'green', 'marker':'.','markersize':6},
            InputTaurus.ConstrEnum.P_T1p1_J00:{'color':'black', 'marker':'.','markersize':6},
            InputTaurus.ConstrEnum.P_T10_J00:{'color':'red',  'marker':'.','markersize':10},
            InputTaurus.ConstrEnum.P_T00_J10:{'color':'blue', 'marker':'.','markersize':10},                    
        }
        kw = {'fontsize': 18}
        for var in observables2plot:
            fig, ax = plt.subplots(1, 1, figsize=(5,4))
            for const in (*constr_2_plot, *constraints_t1):
                
                ax.plot(defs_main_dirs[const][:-1], data_main_dirs[const][var][:-1], 
                        label=getVariableInLatex(const), **kw2[const])
            ax.set_xlabel('Pair. Coup. value $\delta^{JT}$',**kw)
            ax.set_ylabel(getVariableInLatex(var),**kw)
            ax.set_title(getVariableInLatex(var)+ f'  {int_str}         $Z,N=({z},{n})$',**kw)
            ax.grid(True)
            if var == 'pair':
                lims = {(10,12): (-0.5,0.03), (10,16): (-0.5,0.03), (10,14): (-3.3,-2.5)}
                if (z,n) in lims: ax.set_ylim(lims[(z, n)])
            ax.legend()
            fig.tight_layout()
            fig.savefig("{}/{}_1dim-PTJ_{}.pdf".format(FOLDER2SAVE, var, nucl[0]))
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
    nuclei = [
        # (12,12), (12,13), # (12,15),
        # (10,10), (10,11),
        # (10,12), 
        (10,14), 
        # (10,16),
        # ( 8, 8), (14,14)
        # (11,11), (13,13),
    ] 
    
    
    INTER = 'B1_MZ5'
    GLOBAL_TAIL_INTER = '_B1'
    export_templ = 'export_TES2_{}_z{}n{}_{}.txt'
    
    constraints_ = [
        # InputTaurus.ConstrEnum.P_T00_J10,
        InputTaurus.ConstrEnum.P_T00_J1m1,
        # InputTaurus.ConstrEnum.P_T00_J1p1,
        # InputTaurus.ConstrEnum.P_T10_J00,
        # InputTaurus.ConstrEnum.P_T1m1_J00,
        #InputTaurus.ConstrEnum.P_T1p1_J00,
        
    ]
    observables2plot = [
        'pair', 
        # 'pair_pn', 'pair_pp', 'pair_nn',
        'E_HFB',
        # 'var_n', 'var_p',
        # 'beta_isoscalar',
        # 'gamma_isoscalar',
        # 'Jz',
        # InputTaurus.ConstrEnum.P_T00_J10,
        # InputTaurus.ConstrEnum.P_T10_J00,
        # InputTaurus.ConstrEnum.P_T00_J1m1,
        # InputTaurus.ConstrEnum.P_T1p1_J00,
        # InputTaurus.ConstrEnum.P_T1m1_J00,
    ]
    
    MAIN_FLD = f'../DATA_RESULTS/SD_Odd_pnPairing/HFB_S0MZ5/{constraints_[0]}'
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
    MAIN_FLD = '../DATA_RESULTS/SD_Odd_pnPairing/HFB_S0MZ5/{pair_cnstr}'
    plotContourVAPPAV_2dT1T0vsTppnnFromFolders(folders_2_import, MAIN_FLD, 
                                               constraints_pn, observables2plot)
    
    
