'''
Created on 25 oct 2024

@author: delafuente
'''
import os
from tools.helpers import importAndCompile_taurus, TBME_SUITE, printf, __log_file

from tools.inputs import InputTaurus
from tools.Enums import GognyEnum
from scripts2d.pair_scripts2d import run_pair_surfaces_2d
from tools.base_executors import _Base1DTaurusExecutor, SetUpStoredWFAndHamiltonian

if not (InputTaurus.PROGRAM in os.listdir()): importAndCompile_taurus()

if __name__ == '__main__':
    
    ## MZ4 lengths
    interactions = {
        (10,10): (4, 0, 1.80), (10,12): (4, 0, 1.75),
        (12,12): (4, 0, 1.75), (12,14): (4, 0, 1.75),
    }
    
    interaction_ = 'usdb'
    N = 5
    interactions = {
        # (2, 2) : interaction_,
        #( 8, 8): (N, 0, None),
        (10,10): (N, 0, None),
        (10,11): (N, 0, None),
        (10,12): (N, 0, None),
        (10,13): (N, 0, None),
        (10,14): (N, 0, None),
        (10,15): (N, 0, None),
        (10,16): (N, 0, None),
        #(12,12): (N, 0, None),
        #(12,13): (N, 0, None),
        #(12,15): (N, 0, None),
        #(14,14): (N, 0, None),
        # (14,15): (N, 0, None),
        # (16,16): (N, 0, None),
        # (16,17): (N, 0, None),
        # (18,18): (N, 0, None),
        # (18,19): (N, 0, None),
    }
    if os.getcwd().startswith('C:'):
        interactions = { (10, 11) : 'B1_MZ4', (12, 12) : 'B1_MZ4', }
    
    nucleus = sorted(list(interactions.keys()))    
    
    ## !! DO NOT CHANGE THE ORDER OF THESE CONSTRAINTS.
    PAIR_CONSTRS = {
        # InputTaurus.ConstrEnum.P_T00_J10   : (-0.01, 0.8, 10),
        # InputTaurus.ConstrEnum.P_T00_J1m1  : (-0.01, 0.8, 10),
        # InputTaurus.ConstrEnum.P_T00_J1p1  : (-0.01, 0.8, 10),
        # InputTaurus.ConstrEnum.P_T10_J00   : (-0.8, 0.8, 10),
        InputTaurus.ConstrEnum.P_T1m1_J00  : (-0.05, 0.8, 10),
        InputTaurus.ConstrEnum.P_T1p1_J00  : (-0.01, 0.8, 10),        
    }
    constr_onrun = {
        InputTaurus.ConstrEnum.b10 : (0.0, 0.0),
        InputTaurus.ConstrEnum.b11 : (0.0, 0.0),
        InputTaurus.ConstrEnum.b21 : (0.0, 0.0),
        InputTaurus.ConstrEnum.b31 : (0.0, 0.0),
        InputTaurus.ConstrEnum.b41 : (0.0, 0.0),
        # InputTaurus.ConstrEnum.P_T00_J10   : 0.0,
        # InputTaurus.ConstrEnum.P_T00_J1m1  : 0.0,
        # InputTaurus.ConstrEnum.P_T00_J1p1  : 0.0,
        # InputTaurus.ConstrEnum.P_T10_J00   : 0.0,
        # InputTaurus.ConstrEnum.P_T1m1_J00  : 0.0,
        # InputTaurus.ConstrEnum.P_T1p1_J00  : 0.0,
    }
    
    # SetUpStoredWFAndHamiltonian.setUpMainFolder('SEEDS_VAPS0_ZNK')
    
    run_pair_surfaces_2d(
        nucleus, interactions, PAIR_CONSTRS,
        gogny_interaction=GognyEnum.B1, ROmega=(0,0), 
        convergences=20, seed_base=0, ## put to 0, 1 to start from seeds
        valid_Ks_to_block=[],
        fomenko_points=(7, 7), 
        sym_calc_setup=_Base1DTaurusExecutor.SymmetryOptionsEnum.NO_CORE_CALC,
        **constr_onrun
    )
    printf("I finished!")
    
    if os.getcwd().startswith('/'): 
        ## Print the base results in case of preconvergence calculation
        SetUpStoredWFAndHamiltonian.setUpMainFolder('')
    
    #===========================================================================
    # ## Testing in Windows to plot auxiliary 2D plot from folder
    #===========================================================================
    if os.getcwd().startswith('C:'):
        FLD = 'BU_folder_B1_MZ3_z2n1'
        EXPORT_FN = 'export_TES2_PT00J10_PT1p1J00_z2n1_B1_MZ3.txt'
        
        from tools.data import DataTaurus
        from tools.helpers import OUTPUT_HEADER_SEPARATOR
        with open(f"{FLD}/{EXPORT_FN}", 'r') as f:
            lines = f.readlines()
            _, var1, var2 = lines[0].split(', ')
            data, deforms = {}, {}
            sets12 = [set(), set()]
            for line in lines[1:]:
                key_, csv_ = line.split(OUTPUT_HEADER_SEPARATOR)
                obj = DataTaurus(2, 1, None, True)
                obj.setDataFromCSVLine(csv_)
                
                key_, defs_ = key_.split(': ')
                key_  = [ int(k)  for k in key_.split(', ')]
                key_  = tuple(key_) 
                defs_ = [float(x) for x in defs_.split(',')]
                
                for i in (0,1): sets12[i].add(key_[i])
                
                data[key_] = obj.E_HFB
                deforms[key_] = defs_
        
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
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
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title("2D Surface Plot")
        
        # Display the plot
        plt.show()        
                