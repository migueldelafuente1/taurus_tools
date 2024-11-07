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
if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Enable LaTeX in Matplotlib
    plt.rcParams.update({
        "text.usetex": False,
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

def plotVAPPAV_2d_byConstraintsFromFolders(folders_2_import, MAIN_FLD, constr_list):
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
                f"$^{{{z_real+n_real}}}${{{elementNameByZ[z_real]}}}")
        
        print(f"[{z}, {n}]   ******** ")
        
        export_templ = folder_args[2] # 'export_TES2_{}_z{}n{}_{}'
        
        possible_export_fn = [convertPairingShortcuts(c) for c in shuffle_constr]
        possible_export_fn = ['_'.join(permcnstr) for permcnstr in possible_export_fn]
        
        fld_  = folder_args[1].format(MAIN_FLD=MAIN_FLD, INTER=INTER, z=z, n=n)
        FLD  = Path(fld_)
        
        possible_export_fn = [export_templ.format(exp,z,n,INTER) for exp in possible_export_fn]
        
        for EXPORT_FN in possible_export_fn:
            if not os.path.exists(f"{FLD}/{EXPORT_FN}"): continue
            with open(f"{FLD}/{EXPORT_FN}", 'r') as f:
                lines = f.readlines()
                constr_2 = lines[0].strip().split(', ')
                assert all([c in constr_list for c in constr_2[1:]]), "Missing constraints"
                data, deforms = {}, {}
                set_values = [set() for _ in range(len(constr_list))]
                for line in lines[1:]:
                    key_, csv_ = line.split(OUTPUT_HEADER_SEPARATOR)
                    obj = DataTaurus(z, n, None, True)
                    obj.setDataFromCSVLine(csv_)
                    
                    key_, defs_ = key_.split(': ')
                    key_  = [ int(k)  for k in key_.split(', ')]
                    key_  = tuple(key_) 
                    defs_ = [float(x) for x in defs_.split(',')]
                                        
                    data[key_] = obj.P_T1m1_J00
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
        
        
        # Define a function for the surface (for example, a Gaussian)
        Z = 0.0*X + 0.0*Y
        for key_, defs_ in data.items():
            i = key_[0] - k0_lims[0]
            j = key_[1] - k1_lims[0]
            Z[i, j] = data[key_]
            
            
        # Plotting the surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        
        # Add a color bar for reference
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Labels and title
        ax.set_xlabel(constr_list[0])
        ax.set_ylabel(constr_list[1])
        ax.set_zlabel("E_hfb")
        ax.set_title("E_hfb")
        
        # Display the plot
        plt.show()
    'export_TES2_PT1p1J00_PT00J10_z12n13_B1_MZ4'
if __name__ == '__main__':
    #===========================================================================
    # # PLOT FROM FOLDERS
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
        InputTaurus.ConstrEnum.P_T00_J10,
        # InputTaurus.ConstrEnum.P_T00_J1m1,
        # InputTaurus.ConstrEnum.P_T00_J1p1,
        # InputTaurus.ConstrEnum.P_T10_J00,
        
        # InputTaurus.ConstrEnum.P_T1m1_J00,
        InputTaurus.ConstrEnum.P_T1p1_J00,
    ]
    
    MAIN_FLD = f'../DATA_RESULTS/SD_Odd_pnPairing/{constraints_[0]}'
    folders_2_import = [
        ((z, n), '{MAIN_FLD}/BU_folder_{INTER}_z{z}n{n}/', export_templ) for z,n in nuclei
    ]
    
    plotVAPPAV_2d_byConstraintsFromFolders(folders_2_import, MAIN_FLD, constraints_)
    
    
