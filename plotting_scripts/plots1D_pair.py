'''
Created on 13 dic 2024

@author: delafuente
'''
from copy import deepcopy
from pathlib import Path
from tools.inputs import InputTaurus
from tools.data import DataTaurus, DataTaurusPAV
from tools.helpers import elementNameByZ, OUTPUT_HEADER_SEPARATOR
from tools.plotter_levels import MATPLOTLIB_INSTALLED

import os

HEADER_HAMIL = 'HAMIL_DECOMP_'
HAMILT_BENCH = 'HAMIL_DECOMP_bench'

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

def plotVAP_pairCoupling_FromFolders(folders_2_import, MAIN_FLD_TEMP, 
                                     constraints_pair, observables2plot, 
                                     hamil_decompos=True):
    """
    Plot the different constraints related to pairing 
    """
    global INTER
    global GLOBAL_TAIL_INTER
    
    FOLDER2SAVE = '/'.join(MAIN_FLD_TEMP.split('/')[:-1])
    
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
        for constr in constraints_pair:
            data_cnstr_plot[constr] = dict([(v, []) for v in observables2plot])
            deforms[constr]         = []
            data_hamils[constr]     = {}
            deforms_hamils[constr]  = {}
            hamiltonians[constr]    = []
            
            cnst2 = constr.replace('_','')
            fld = folder_templ.format(MAIN_FLD=MAIN_FLD_TEMP, constrs=cnst2, 
                                      INTER=INTER, z=z, n=n)
            fld = Path(fld)
            export_fn = export_templ.format(cnst2, z, n, INTER)
            
            with open(fld / export_fn, 'r') as f:
                lines = f.readlines()
                _, cnst2 = lines[0].strip().split(', ')
                assert constr == cnst2, f"Imported export file inconsistent with constrait [{constr}/{cnst2}]"
                for line in lines[1:]:
                    key_, csv_ = line.split(OUTPUT_HEADER_SEPARATOR)
                    obj = DataTaurus(z, n, None, True)
                    obj.setDataFromCSVLine(csv_)
                    
                    key_, defs_ = key_.split(': ')
                    deforms[constr].append(float(defs_))
                    for var in observables2plot:
                        data_cnstr_plot[constr][var].append(getattr(obj, var))
                    
            ## get the data from hamil decomposition
            hamils_ = filter(lambda x: os.path.isdir(fld / x) and x.startswith(HEADER_HAMIL),
                             os.listdir(fld))
            for h_fld in hamils_:
                hamil = h_fld.replace(HEADER_HAMIL, '')
                hamiltonians[constr].append(hamil)
                data_hamils[constr][hamil] = dict([(v, []) for v in observables2plot])
                deforms_hamils[constr][hamil] = []
                cnst2 = constr.replace('_','')
                
                export_fn = f"{h_fld}/" + export_templ.format(cnst2, z, n, hamil)
                
                
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
                
                ax.plot(x, data_cnstr_plot[constr][var], '.-',
                        label=constr, color=colors.get(constr, None))
            ax.set_xlabel('Pair. Coup. value $\delta^{JT}$')
            ax.set_ylabel(var)
            ax.set_title (var)
            ax.legend()
        
        ## PLOT the hamiltonian contributions for each variable to plot
        if hamil_decompos:
            markers = '*.o+P^vs'
            for constr in constraints_pair:
                for var in observables2plot:
                    fig = plt.figure()
                    ax  = fig.add_subplot(111)
                    for i, hamil in enumerate(hamiltonians[constr]):
                        #if hamil == 'bench': continue
                        x = deforms_hamils[constr][hamil]
                        y = data_hamils[constr][hamil][var]
                        
                        ax.plot(x, y, marker=markers[i], label=hamil, linestyle='--')
                    
                    ax.plot(deforms[constr], data_cnstr_plot[constr][var],
                            label=INTER, color=colors.get(constr, None))
                    ax.set_xlabel('Pair. Coup. value $\delta^{JT}$')
                    ax.set_ylabel(var)
                    ax.set_title(f"{constr}-{var} for {INTER} decomposition.")
                    ax.legend()
        plt.show()
        

if __name__ == '__main__':
    
    MAIN_FLD = '../DATA_RESULTS/SD_Odd_pnPairing/{pair_cnstr}'
    
    # nuclei = [(12,11 + 2*i) for i in range(0, 6)] # 6
    nuclei = [
        # (12,12), (12,13), (12,15),
        # (10,10), (10,11),
        # (10,12), 
        (10,10), 
        # (10,16),
        # ( 8, 8), (14,14)
        # (11,11), (13,13),
    ] 
    
    INTER = 'B1_MZ5'
    GLOBAL_TAIL_INTER = '_B1'
    export_templ = 'export_TES_{}_z{}n{}_{}.txt'
    
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
        # 'pair_pn', 'pair_pp', 'pair_nn',
        # 'E_HFB',
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
    
    MAIN_FLD = f'../DATA_RESULTS/PN_mixing/hamilParts_S0B1_MZ5'
    subfolder_temp = '{MAIN_FLD}/BU_folder_{constrs}_{INTER}_z{z}n{n}/'
    folders_2_import = [
        ((z, n), subfolder_temp, export_templ) for z,n in nuclei
    ]
    
    plotVAP_pairCoupling_FromFolders(folders_2_import, MAIN_FLD, constraints_,
                                     observables2plot, hamil_decompos=True)
    