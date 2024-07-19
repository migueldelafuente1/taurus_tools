'''
Created on 21 mar 2024

@author: delafuente
'''
import os
import shutil
import numpy as np
from pathlib import Path

from tools.plotter_1d import _Plotter1D, Plotter1D_Taurus,\
    Plotter1D_CanonicalBasis
from tools.helpers import elementNameByZ, printf, OUTPUT_HEADER_SEPARATOR
# from _legacy.exe_isotopeChain_taurus import DataTaurus
from copy import deepcopy
from tools.data import DataTaurus, DataTaurusPAV, DataTaurusMIX
from tools.plotter_levels import EnergyLevelGraph, BaseLevelContainer

__K_COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black']
# __K_LSTYLE = ['-', '--', ':', '-.', ' ', '']*2
# __K_LSTYLE = ['solid', 'dashed', 'dashdot', 'dotted', 
#               (0, (1,10)), (0, (5, 5)), (5, (10,3)), (0, (3,5,1,5)), (0, (3,5,1,5,1,5))]
__K_LSTYLE = ['solid', (0, (5,1)), (5, (10,3)), 'dashed', 'dashdot', 
            (0, (3,1,1,1)), (0, (3,5,1,5)), (0, (5,10)), 'dotted',
            (0, (3,10,1,10)), (0, (3,10,1,10)), (0, (1,10))]
__J_MARKER = ['.',  'd', 'v', '^','+',"1", "2", "3", "x"]

def _getPAVdiagonalFromList(list_dat_file):
    """
    from a list of PAV elements or folders: 1,2,3,4,5,6 get the n elements (3) 
    and the values for the diagonal wf overlap: 1, 4, 6
    """
    if not isinstance(list_dat_file, (list, tuple)): list_dat_file = list(list_dat_file)
    if isinstance(list_dat_file[0], str): list_dat_file = [int(x) for x in list_dat_file]
    
    list_dat_file.sort()
    n = int((-1 + (1 + 8*len(list_dat_file))**.5)//2)
    list_dat_file, k = [], 0
    for i in range(n):
        for j in range(i, n):
            k += 1
            if i == j: list_dat_file.append(str(k))
    return list_dat_file


def _plotscript1_OEK_withNoProjection(nuclei, K_MAX, MZMAX, folders2import=None,
                                      main_folder='../DATA_RESULTS/K_OddEven/'):
    """
        Using for results stored as 'Interaction'/'Z-nucleus as str'/'A'/'HFB'/
    """
    MODE_ = 0 if not folders2import else 1
    for z, n in nuclei:
        NUC = elementNameByZ[z]
        if MODE_ == 0:
            SUBFLD_ = f'B1/{NUC}/{z+n}/HFB/'
        elif MODE_ == 1:
            SUBFLD_ = f'{folders2import[(z, n)]}/'
        
        _Plotter1D.setFolderPath2Import(main_folder+SUBFLD_)
        _Plotter1D.EXPORT_PDF_AND_MERGE = True
        FLD_ = _Plotter1D.FOLDER_PATH
        printf(FLD_)
        ## Set the files and the labels to plot
        aa = z+n
        files_, labels_by_files, kwargs = [], [], {}
        for K in range(1, K_MAX+1, 2):
            _fld = f"{K}_0_VAP".replace('-', '_')
            files_.append(f"{_fld}/export_TESb20_K{K}_z{z}n{n}_B1_MZ{MZMAX}.txt")
            kwargs[files_[-1]] = dict(linestyle='--', #'--',
                                      marker = '.' if (K>0) else 'o',
                                      markerfacecolor ='None' if (K<0) else None,
                                      color  =__K_COLORS[abs(K)//2], )
            labels_by_files.append(f"block K={K}/2")
        files_.append(f'export_TESb20_z{z}n{n}_B1_MZ{MZMAX}.txt')
        labels_by_files.append(f"HFB false O-E ")
        
        labels_by_files = dict(zip(files_, labels_by_files))
        
        plt_obj = Plotter1D_Taurus(files_)
        # plt_obj.modifyValuesForResults(files_[:-1],
        #                                E_HFB = lambda r: r.E_HFB - (r.E_zero/2))
        plt_obj.setTitle(f"$Quadrupole\ TES,\ ^{{{z+n}}}{NUC}\ B1$")
        plt_obj.LATEX_FORMAT = True
        plt_obj.setPyplotKWARGS(dict([(f, kwargs[f]) for f in files_[:-1]]))
        plt_obj.setPyplotKWARGS(
            {#files_[0]:  dict(color='red', marker='.'),
             files_[-1]: dict(color='black', linestyle='-', marker='.')})
        
        attr2plot_list = [] # [ 'E_HFB', 'hf',]
        for attr2plot in attr2plot_list:
            plt_obj.setXlabel(r"$\beta_{20}$")
            # plt_obj.defaultPlot(attr2plot, show_plot=attr2plot==attr2plot_list[-1])
            plt_obj.setLabelsForLegendByFileData(labels_by_files)
            plt_obj.shift2topValue_plot(attr2plot, show_errors=True,
                                        show_plot=False )#attr2plot==attr2plot_list[-1])
        
        attr2plot_list = ['E_HFB', 'hf', 'pair', 'r_isoscalar', 'parity', 
                          'b30_isoscalar', 'Jz', 'Jz_2']  # DataTaurus
        for attr2plot in attr2plot_list:
            plt_obj.setXlabel(r"$\beta_{20}$")
            plt_obj.setLabelsForLegendByFileData(labels_by_files)
            plt_obj.defaultPlot(attr2plot, show_errors=True,
                                show_plot=attr2plot==attr2plot_list[-1])
            
        plt_obj.mergeFiguresInto1PDF(f'{NUC}{z+n}_MZ{MZMAX}_Kblocking.pdf')


def particlePlusRotor_SCL_spectra(data_j, b20_deform, J2_vals, K_val,
                                  data_5o2=None):
    """
    Get the interpolation curves for two set of data, the first J values must be
    in order in J_vals. 
    
    The formula is for the strong coupling limit: E = A + B (J(J+1) - K**2)
    
    For K = 1/2, coriolis decoupling term can be evaluated, In that case, resultant
    B value differs from the first formula.
    """
    from scipy.interpolate import interp1d
    do_coriolis = K_val == 1 and data_5o2 != None
    K_val /= 2
    J_vals = [j/2 for j in J2_vals]
    data_1o2, data_3o2 = data_j[J2_vals[0]], data_j[J2_vals[1]] 
    
    x_new = np.linspace(min(b20_deform), max(b20_deform), 101, endpoint=True)
    y01 = np.array(data_1o2)
    y02 = np.array(data_3o2)
    if data_5o2: y03 = np.array(data_5o2)
    
    A, B, C = np.zeros(len(y01)), np.zeros(len(y01)), np.zeros(len(y01))
    if data_5o2: y03 = np.array(data_5o2)
    if do_coriolis:
        for i in range(len(y01)):
            a, b = [], np.array([y01[i], y02[i], y03[i]])
            for ii in range(3):
                a.append([1, J_vals[ii]*(J_vals[ii]+1), 
                          (J_vals[ii]+0.5)*((-1)**(J_vals[ii]+0.5))])
            x = np.linalg.solve(np.array(a), b)
            A[i], B[i], C[i] = x[0], x[1], x[2]
    else:
        B = (y01 - y02) / (J_vals[0]*(J_vals[0] + 1) - J_vals[1]*(J_vals[1] + 1))
        A = y01 - B*(J_vals[0]*(J_vals[0] + 1) - K_val**2)
        C = np.zeros(len(y01))
    y0 = [y01, y02,]
    model_diff = {}
    for k, J in enumerate(J_vals[2:]):
        _2J = int(2*J)
        y0.append(A + B*(J*(J+1) - K_val**2))
        if do_coriolis: 
            y0[2+k] += ((-1)**(J+0.5)) * C * (J+0.5)
        model_diff[_2J] = np.array(data_j[_2J]) - y0[-1]
    
    data_interpolated = {}
    for i, y_j in enumerate(y0):
        J = J_vals[i]
        # Create the interpolation function
        # try:
        #     interp_func = interp1d(b20_deform, y_j, kind='cubic') # 'quadratic'
        # except ValueError as e:
        #     _=0
        interp_func = interp1d(b20_deform, y_j, kind='cubic') # 'quadratic'
        
        # Generate points for the interpolated curve
        y_new = interp_func(x_new)
        
        data_interpolated[J2_vals[i]] = (x_new, y_new)
    
    C = np.array([C[i]/B[i] for i in range(len(C))])
    return data_interpolated, A, B, C, model_diff

def _plotPAVresultsFromFolders(folders_2_import, MAIN_FLD_TMP, K_val, parity=0,
                               plot_SCL_interpolation=False):
    """
    folders_2_import: <list> of ((z, n), folder path, Datatype, )
            folder path must have keyword=MAIN_FLD
    MAIN_FLD_TMP: must have keyword K_val
    """
    assert parity in (-1, 0), "Error parity value"
    PAR = parity % 2
    for folder_args in folders_2_import:
        FLD_, list_dat_file = None, None
        
        for i, arg in enumerate(folder_args):
            if i == 0:
                z, n = arg
                nucl = f"{z+n}{elementNameByZ[z]}"
            elif i == 1:
                MAIN_FLD_TMP = MAIN_FLD_TMP.format(K_val=K_val)
                arg = arg.format(MAIN_FLD=MAIN_FLD_TMP, z=z, n=n)
                assert os.path.exists(arg), "Unfound Folder"
                FLD_ = Path(arg)
                FLD_PNPAMP = FLD_ 
            elif i == 2:
                export_VAP_file = arg.format(z, n)
            elif i == 3:
                assert os.path.exists(FLD_ / arg), "unfound list.dat file"
                list_dat_file = arg
        
        FLD_HWG = FLD_ / 'HWG'
        if not os.path.exists(FLD_HWG): FLD_HWG = FLD_ / Path('PNPAMP_HWG/HWG/')
        if not os.path.exists(FLD_HWG): FLD_HWG = None
        
        for fld2 in ('PNAMP', 'outputs_PAV', 'PNPAMP_HWG/outputs_PAV', 'PNPAMP_HWG'):
            FLD_PNPAMP = FLD_ / fld2
            if (FLD_PNPAMP.exists()) and os.listdir(FLD_PNPAMP) != []: 
                break
        if os.listdir(FLD_PNPAMP)==0: 
            print("[ERROR - SKIP] Not found PAV results for plotting in ", FLD_)
            continue
        else:
            K = K_val
            if FLD_PNPAMP.name == 'outputs_PAV':
                if 'gcm_diag' in os.listdir(FLD_PNPAMP):
                    with open(FLD_PNPAMP / 'gcm_diag') as f:
                        list_dat_file = [ff.strip() for ff in f.readlines()]
                else:
                    list_dat_file = filter(lambda x: x.name.startswith('OUT_'), FLD_PNPAMP.iterdir())
                    list_dat_file = map(lambda x: str(x.name), list_dat_file)
                    list_dat_file = [int(x.replace('OUT_', '')) for x in list_dat_file]
                    list_dat_file = _getPAVdiagonalFromList(list_dat_file)
                    list_dat_file = [f"OUT_{x}" for x in list_dat_file]
                    
            elif FLD_PNPAMP.name == 'PNPAMP_HWG':
                list_dat_file = filter(lambda x: x.name.isdigit(), FLD_PNPAMP.iterdir())
                list_dat_file = map(lambda x: str(x.name), list_dat_file)
                list_dat_file = [int(x) for x in list_dat_file]
                list_dat_file = _getPAVdiagonalFromList(list_dat_file)
                list_dat_file = [f"{x}/OUT" for x in list_dat_file]
            
            if FLD_PNPAMP.name != 'PNAMP':
                # asume from the vap-
                b20_deform = filter(lambda x: x.startswith('def'), os.listdir(FLD_))
                b20_deform = [x.replace('def','').replace('_PAV','') for x in b20_deform]
                b20_deform = [int(ff.replace('_', '-')) for ff in b20_deform]            
        ## 
        
        if not list_dat_file:
            list_dat_file = filter(lambda x: x.endswith('.dat'), os.listdir(FLD_PNPAMP))
            list_dat_file = list(filter(lambda x: x.startswith('list'), list_dat_file))
            if len(list_dat_file) == 1: 
                list_dat_file = list_dat_file[0]
            else: raise Exception("unfound unique list.dat file for data order.")
        
        
        if isinstance(list_dat_file, str):
            with open(FLD_PNPAMP / list_dat_file, 'r') as f:
                K = int(list_dat_file.replace('_pav.dat', '').replace('list_k', ''))
                list_dat_file = [ff.strip() for ff in f.readlines()]
                b20_deform = [ff.replace('d','').replace('.OUT','') for ff in list_dat_file]
                b20_deform = [int(ff.split('_')[1]) for ff in b20_deform]
        
        data = [DataTaurusPAV(z, n, FLD_PNPAMP / f) for f in list_dat_file]
        data_hwg = []
        if FLD_HWG != None:
            list_dat = filter(lambda x: x.endswith('.dat'), os.listdir(FLD_HWG))
            list_dat = list(list_dat)
            data_hwg = [DataTaurusMIX(FLD_HWG / f) for f in list_dat]
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1)
        energy_by_J = {}
        for id_, dat in enumerate(data):
            kval = [(i, k) for i, k in enumerate(dat.KJ)]
            kval = list(filter(lambda x: x[1]==K, enumerate(dat.KJ)))
            
            ener, jval = [], []
            for i, _ in kval:
                if abs(dat.proj_norm[i]) < 1.0e-8:
                    printf("[Warning] PAV norm = 0 state found. i_b20_def=",
                          id_, "\n", data[i])
                    continue
                else:
                    if abs(dat.proj_energy[i]) < 1.0e-8:
                        printf("[Warning] PAV norm/= 0 and E_pav=0: i_b20_def=",
                              id_, "\n", data[i])
                        continue
                    
                ener.append(dat.proj_energy[i])
                jval.append(dat.J[i])
            
            for i, J in enumerate(jval):
                if J not in energy_by_J: energy_by_J[J] = []

                energy_by_J[J].append(ener[i])
        
        E_vap, b20_vap = [], []
        ## Import the values form HFB.
        with open(FLD_ / export_VAP_file, 'r') as f:
            dataVAP = f.readlines()
            Dty_, _ = dataVAP[0].split(',')
            dataVAP = [l.split(OUTPUT_HEADER_SEPARATOR) for l in dataVAP[1:]] 
            
            for h, line in dataVAP:
                i, b20 = h.split(':')
                i, b20 = int(i), float(b20)
                
                b20_vap.append(b20)
                if i in b20_deform:
                    ## this consideration is to avoid i=-1, b20 = -1.00 refixing
                    if isinstance(b20_deform[b20_deform.index(i)], int) :
                        b20_deform[b20_deform.index(i)] = b20 + 0.001
                
                res = DataTaurus(z, n, None, empty_data=True)
                res.setDataFromCSVLine(line)
                E_vap.append(res.E_HFB)
        
        b20_kvap, E_Kvap = [], []
        export_VAP_K_file = export_VAP_file.replace('TESb20_', f'TESb20_K{K_val}_')
        with open(FLD_ / Path(f'{K_val}_{PAR}_VAP/') / export_VAP_K_file, 'r') as f:
            dataVAP = f.readlines()
            Dty_, _ = dataVAP[0].split(',')
            dataVAP = [l.split(OUTPUT_HEADER_SEPARATOR) for l in dataVAP[1:]] 
            
            for h, line in dataVAP:
                i, b20 = h.split(':')
                i, b20 = int(i), float(b20)
                
                b20_kvap.append(b20)
                if i in b20_deform:
                    ## this consideration is to avoid i=-1, b20 = -1.00 refixing
                    if isinstance(b20_deform[b20_deform.index(i)], int) :
                        b20_deform[b20_deform.index(i)] = b20 + 0.001
                
                res = DataTaurus(z, n, None, empty_data=True)
                res.setDataFromCSVLine(line)
                E_Kvap.append(res.E_HFB)
        
        # b20_deform = [i-(len(energy_by_J[1])//2) for i in range(len(energy_by_J[1]))]
        # b20_deform = [i for i in range(len(energy_by_J[1]))]
        par_str = '+' if PAR==0 else '-'
        J_vals = [i for i in energy_by_J.keys()]
        for J, ener in energy_by_J.items():
            if b20_deform.__len__() != ener.__len__():
                printf("   >>> ", J, ener)
                continue
            ax.plot(b20_deform, ener, '.-', label=f'J={J}/2 {par_str}', 
                                            color=__K_COLORS[J_vals.index(J)])
        
        if len(energy_by_J) > 1:
            interp = particlePlusRotor_SCL_spectra(energy_by_J,
                                                   b20_deform, J_vals, K_val)
            interp, sp_e_inter, _1o2IM_inter, _cor_coupl, model_diff = interp
            if plot_SCL_interpolation:
                for J in energy_by_J.keys():
                    ax.plot(*interp[J], '--', label=f'PPRM-SCL J={J}/2 {par_str}',
                                              color=__K_COLORS[J_vals.index(J)])
        ax.plot(b20_vap,  E_vap, "ko-", label='HFB')
        ax.plot(b20_kvap, E_Kvap, "*-", color='orange', label=f'HFB-VAP K={K_val}/2 {par_str}')
        
        plt.title(f"{nucl} PAV projections for K=+{K_val}/2 and PPRM interpolation")
        plt.legend()
        # plt.ylim([-130, -90])
        # plt.xlim([-0.7, 1.2])
        plt.xlabel('b_20')
        plt.savefig(FLD_ / f"plot_pav_pprm_{nucl}.pdf")
        #plt.show()
        
        if len(energy_by_J) > 1:
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(b20_deform, sp_e_inter)
            ax[1].plot(b20_deform, 0.5/_1o2IM_inter)
            ax[0].set_title("sp energies")
            ax[1].set_title("1/2I Inertia Mom.")
            plt.suptitle(f"{nucl} PPRM derived quantities for K=+{K_val}/2 from interpolation")
            plt.savefig(FLD_ / f"derived_pprm_parameters_{nucl}.pdf")
            
            if data_hwg == [] :plt.show()
        
        ## preparing the HWG level scheme
        if data_hwg:
            level_str = ''.join([x.getSpectrumLines() for x in data_hwg])
            levels_1 = EnergyLevelGraph(title=f'K={K_val}{par_str}')
            levels_1.setData(level_str, program='taurus_hwg')
            BaseLevelContainer.RELATIVE_PLOT = True
            BaseLevelContainer.ONLY_PAIRED_STATES = False
            BaseLevelContainer.MAX_NUM_OF_SIGMAS  = 3
            
            _graph = BaseLevelContainer()
            _graph.global_title = "Comparison HWG from different K-blocks"
            _graph.add_LevelGraph(levels_1)
            _graph.plot()


def _generate_images_hfb_vapK_pav_hwg(b20_hfb, E_hfb, b20_vap_K, E_vap_K, b20_pav_K, 
                                      b20_by_K_and_J, energy_by_J, Jmax_2_plot, 
                                      data_hwg_K,
                                      plot_PAV=True, plot_SCL_interpolation=True, 
                                      FOLDER2SAVE=None, nucl='***'):
    """
    Auxiliary method to print VAP-PAV-HWG merging results from folders in the 
    same or different folders, methods to organize the arguments must be 
    set outside.
    """
    PLOT_PPRC = False
    import matplotlib.pyplot as plt
    # Enable LaTeX in Matplotlib
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # b20_deform = [i-(len(energy_by_J[1])//2) for i in range(len(energy_by_J[1]))]
    # b20_deform = [i for i in range(len(energy_by_J[1]))]
    sp_e_inter_by_K, _1o2IM_inter_by_K, coriolis_coupl_by_K = {}, {}, {}
    model_diff_by_K = {}
    _lims = []
    for K in K_vals:
        if plot_PAV:
            J_vals = sorted(energy_by_J[K].keys())
            for J in J_vals:
                if Jmax_2_plot and (J > Jmax_2_plot): continue 
                ener = energy_by_J[K][J]
                # if b20_deform[K].__len__() != ener.__len__():
                #     printf("   >>> ", J, ener)
                #     continue
                ax.plot(b20_by_K_and_J[K][J], ener, label=f'J={J}/2 K={K}/2', 
                        linestyle=__K_LSTYLE[8],#[J_vals.index(J)+1],
                        marker=f"${J}/2$", markersize=11,
                        color=__K_COLORS[K//2])
        
        if len(energy_by_J.get(K, [])) > 1 and plot_SCL_interpolation:
            data_5o2 = energy_by_J[K][J_vals[2]] if K == 1 else None
            _array_b20 = b20_hfb[K] if isinstance(b20_hfb, dict) else b20_hfb
            _array_b20 = b20_pav_K[K]
            interp = particlePlusRotor_SCL_spectra(energy_by_J[K], 
                                                   _array_b20, J_vals, K,
                                                   data_5o2=data_5o2)
            interp, sp_e_inter, _1o2IM_inter, _cor_coupl, model_diff = interp
            sp_e_inter_by_K  [K]    = sp_e_inter
            _1o2IM_inter_by_K[K]    = _1o2IM_inter
            coriolis_coupl_by_K[K]  = _cor_coupl
            model_diff_by_K  [K]    = model_diff
            
            for J in J_vals:
                if (not PLOT_PPRC) or (Jmax_2_plot and (J > Jmax_2_plot)): 
                    continue 
                ax.plot(*interp[J], '--', label=f'PPRM-SCL J,K={J}/2 {K}/2',
                                          color=__K_COLORS[K//2])
        # ax.plot(b20_hfb [K], E_hfb [K], "o-", label=f'HFB(K={K}/2)')
        ax.plot(b20_vap_K[K], E_vap_K[K], "o-", 
                color=__K_COLORS[K//2], label=f'HFB-VAP K={K}/2',
                linewidth=3)
        _lims = _lims + b20_vap_K[K]
        #plt.pause(2)
    
    _lims = [min(_lims), max(_lims)]
    _rngs = _lims[1] - _lims[0]
    ax.set_xlim( [_lims[0] * 1.1, _lims[1] * 1.5 ] )
    # plt.ylim([-130, -90])
    # plt.xlim([-0.7, 1.2])
    ax.set_xlabel(r"$\beta_{20}$")
    ax.set_ylabel(r"$E\ (MeV)$")
    _txt = f"{nucl[1]} PAV energies for all K-blockings"
    if PLOT_PPRC and plot_SCL_interpolation: _txt += " and PPRM interpolation"
    plt.title(_txt)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    plt.savefig(FOLDER2SAVE / f"plot_pav_pprm_{nucl[0]}.pdf")
    if not plot_PAV:
        plt.show()
        return
    
    fig, ax = plt.subplots(1, 3 if len(coriolis_coupl_by_K)>0 else 2,
                           figsize=(10, 4))
    any_ = False
    for K in K_vals:
        if len(energy_by_J[K]) > 1 and plot_SCL_interpolation:
            ax[0].plot(b20_pav_K[K], sp_e_inter_by_K[K])
            ax[1].plot(b20_pav_K[K], 0.5/_1o2IM_inter_by_K[K], label=f"{K}/2+")
            if len(coriolis_coupl_by_K)>0:
                ax[2].plot(b20_pav_K[K], coriolis_coupl_by_K[K], label=f"{K}/2+")
            any_ = True
    if any_:
        ax[0].set_title("sp energies")
        ax[1].set_title("1/2$\mathcal{I}$ Inertia Mom.")
        if len(coriolis_coupl_by_K)>0:
            ax[2].set_title("Decoupling factor")
        plt.suptitle(f"{nucl[1]} PPRM derived quantities for all K from interpolation")
        plt.savefig(FOLDER2SAVE / f"derived_pprm_parameters_{nucl[0]}.pdf")
        plt.legend()
    else:
        plt.show()
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
    for K in K_vals:
        i = 0
        for J, model_diff in model_diff_by_K[K].items():
            ax2.semilogy(b20_by_K_and_J[K][J], [abs(x) for x in model_diff], 
                     marker=__J_MARKER[i], color=__K_COLORS[K//2],
                     label=f'$K={K}/2^+$ $J={J}/2^+$')
            i += 1 
    ax2.set_title(f"Correspondence PAV - PPR model SCL ({nucl[1]})")
    ax2.set_ylabel(r"$\delta(E^{PAV} - E^{SCL})_{K, J} [MeV]$")
    ax2.set_xlabel(r"$\beta_{20}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER2SAVE / f"model_validity_{nucl[0]}.pdf")
    
    if sum([len(x) for x in data_hwg_K.values()]):
        par_str = '+' #if  parity
        BaseLevelContainer.RELATIVE_PLOT = False #True
        BaseLevelContainer.ONLY_PAIRED_STATES = False
        BaseLevelContainer.MAX_NUM_OF_SIGMAS  = 3
        
        _graph = BaseLevelContainer()
        _graph.global_title = f"Comparison HWG from different K-blocks, {nucl[1]}"
        EnergyLevelGraph.resetClassAttributes()
        for K in K_vals + [-111, ]:
            ## preparing the HWG level scheme
            if data_hwg_K.get(K, None):
                level_str = ''.join([x.getSpectrumLines() for x in data_hwg_K[K]])
                title = f'K={K}/2 {par_str}' if K !=-111 else 'K-mix'
                levels_1  = EnergyLevelGraph(title)
                levels_1.setData(level_str, program='taurus_hwg')
                
                _graph.add_LevelGraph(levels_1)
        plt.tight_layout()
        _graph.plot(FOLDER2SAVE / f'hwg_comparison_{nucl[0]}.pdf',
                    figaspect=(6*(2/3)*len(K_vals), 5))
    ####
    plt.show()

def _plotPAVresultsFromFolders_mulipleK(folders_2_import, MAIN_FLD_TEMP, K_vals,
                                        plot_SCL_interpolation=True, plot_PAV=True,
                                        Jmax_2_plot=None):
    """
    :folders_2_import <nucleus> 
    """
    PLOT_PPRC = False
    FOLDER2SAVE= Path(MAIN_FLD_TEMP).parent
    for folder_args in folders_2_import:
        b20_deform, data, data_hwg, FLD_, FLD_PNPAMP, FLD_HWG = {}, {}, {}, {}, {}, {}
        for K_val in K_vals:
            list_dat_file = None
            
            for i, arg in enumerate(folder_args):
                if i == 0:
                    z, n = arg
                    nucl = (f"{z+n}{elementNameByZ[z]}", 
                            rf"$^{{{z+n}}}${{{elementNameByZ[z]}}}")
                elif i == 1:
                    MAIN_FLD = MAIN_FLD_TEMP.format(K_val=K_val)
                    _aux_path_multi = arg.format(MAIN_FLD=MAIN_FLD_TEMP.format(K_val='mix'), z=z, n=n)
                    _aux_path_multi = Path(_aux_path_multi)
                    arg = arg.format(MAIN_FLD=MAIN_FLD, z=z, n=n)
                    assert os.path.exists(arg), f"Unfound Folder: [{arg}]"
                    FLD_[K_val] = Path(arg)
                    FLD_PNPAMP[K_val] = FLD_[K_val]
                elif i == 2:
                    export_VAP_file = arg.format(z, n)
                elif i == 3:
                    assert os.path.exists(FLD_[K_val]+arg), "unfound list.dat file"
                    list_dat_file = arg
            
            FLD_HWG[K_val] = FLD_[K_val] / 'HWG'
            if not os.path.exists(FLD_HWG[K_val]): 
                FLD_HWG[K_val] = FLD_[K_val] / Path('PNPAMP_HWG/HWG')
            if not os.path.exists(FLD_HWG[K_val]): FLD_HWG[K_val] = None
            
            ## process to resolve PAV data
            K = K_val
            for fld2 in ('PNPAMP_HWG/outputs_PAV', 'outputs_PAV', 'PNPAMP_HWG', 'PNAMP',):
                FLD_PNPAMP[K] = FLD_[K] / fld2
                if (FLD_PNPAMP[K].exists()) and os.listdir(FLD_PNPAMP[K]) != []: 
                    break
            if os.listdir(FLD_PNPAMP[K])==0: 
                print("[ERROR - SKIP] Not found PAV results for plotting in ", FLD_[K])
                continue
            else:
                if FLD_PNPAMP[K].name == 'outputs_PAV':
                    if 'gcm_diag' in os.listdir(FLD_PNPAMP[K]):
                        with (FLD_PNPAMP[K] / 'gcm_diag').open(mode='r') as f:
                            list_dat_file = [ff.strip() for ff in f.readlines()]
                    else:
                        list_dat_file = filter(lambda x: x.name.startswith('OUT_'), FLD_PNPAMP[K].iterdir())
                        list_dat_file = map(lambda x: str(x.name), list_dat_file)
                        list_dat_file = [int(x.replace('OUT_', '')) for x in list_dat_file]
                        list_dat_file = _getPAVdiagonalFromList(list_dat_file)
                        list_dat_file = [f"OUT_{x}" for x in list_dat_file]
                        
                elif FLD_PNPAMP[K].name == 'PNPAMP_HWG':
                    list_dat_file = filter(lambda x: x.name.isdigit(), FLD_PNPAMP[K].iterdir())
                    list_dat_file = map(lambda x: str(x.name), list_dat_file)
                    list_dat_file = [int(x) for x in list_dat_file]
                    list_dat_file = _getPAVdiagonalFromList(list_dat_file)
                    list_dat_file = [f"{x}/OUT" for x in list_dat_file]
                
                if FLD_PNPAMP[K].name != 'PNAMP':
                    export_VAP_K_file = export_VAP_file.replace('TESb20_', f'TESb20_K{K}_')
                    
                    with open(FLD_[K]/Path(f'{K}_0_VAP')/export_VAP_K_file, 'r') as f:
                        dataVAP = f.readlines()
                        Dty_, _ = dataVAP[0].split(',')
                        dataVAP = [l.split(OUTPUT_HEADER_SEPARATOR) for l in dataVAP[1:]] 
                        
                        aux = []
                        for h, line in dataVAP:
                            i, b20 = h.split(':')
                            i, b20 = int(i), float(b20)
                            dat = DataTaurus(z, n, None, empty_data=True)
                            dat.setDataFromCSVLine(line)
                            if dat.properly_finished: aux.append(i)
                    b20_deform[K] = sorted(aux)
            ## 
            
            if not list_dat_file:
                list_dat_file = filter(lambda x: x.endswith('.dat'), 
                                       os.listdir(FLD_PNPAMP[K_val]))
                list_dat_file = list(filter(lambda x: x.startswith('list'), list_dat_file))
                if len(list_dat_file)== 1: 
                    list_dat_file = list_dat_file[0]
                else: Exception("unfound unique list.dat file for data order.")
            
            if isinstance(list_dat_file, str):
                with open(FLD_PNPAMP[K_val] / list_dat_file, 'r') as f:
                    K = int(list_dat_file.replace('_pav.dat', '').replace('list_k', ''))
                    assert K == K_val, "K not in expected K-blocked values."
                    list_dat_file = [ff.strip() for ff in f.readlines()]
                    b20_deform[K] = [ff.replace('d','').replace('.OUT','') 
                                     for ff in list_dat_file]
                    b20_deform[K] = [int(ff.split('_')[1]) for ff in b20_deform[K]]
            elif isinstance(list_dat_file, list):
                pass
            else: raise Exception("Invalid list_dat object", list_dat_file)
            
            data[K] = [DataTaurusPAV(z, n, FLD_PNPAMP[K_val] / f) for f in list_dat_file]
            data_hwg[K] = []
            if FLD_HWG[K] != None:
                list_dat = filter(lambda x: x.endswith('.dat'), os.listdir(FLD_HWG[K]))
                list_dat = list(list_dat)
                data_hwg[K] = [DataTaurusMIX(FLD_HWG[K] / f) for f in list_dat]
            if os.path.exists(_aux_path_multi) and not -111 in data_hwg:
                _aux_path_multi = _aux_path_multi / 'HWG'
                list_dat = filter(lambda x: x.endswith('.dat'), os.listdir(_aux_path_multi))
                list_dat = list(list_dat)
                data_hwg[-111] = [DataTaurusMIX(_aux_path_multi / f) for f in list_dat]
        
        b20_by_K_and_J = {}
        energy_by_J    = {}
        for K in K_vals:
            energy_by_J[K] = {}
            b20_by_K_and_J[K] = {}
            for id_, dat in enumerate(data[K]):
                kval = [(i, k) for i, k in enumerate(dat.KJ)]
                kval = list(filter(lambda x: x[1]==K, enumerate(dat.KJ)))
                
                ener, jval, b20J = [], [], []
                for i, _ in kval:
                    if abs(dat.proj_norm[i]) < 1.0e-8:
                        printf("[Warning] PAV norm = 0 state found. i_b20_def=",
                              id_, "\n", data[K][id_])
                        continue
                    else:
                        if abs(dat.proj_energy[i]) < 1.0e-8:
                            printf("[Warning] PAV norm/= 0 and E_pav=0: i_b20_def=",
                                  id_, "\n", data[K][id_])
                            continue
                    ener.append(dat.proj_energy[i])
                    jval.append(dat.J[i])
                    if jval[-1] in b20_by_K_and_J[K]:
                        b20_by_K_and_J[K][jval[-1]].append(b20_deform[K][id_])
                    else:
                        b20_by_K_and_J[K][jval[-1]] = [b20_deform[K][id_], ]
                
                for i, J in enumerate(jval):
                    if J not in energy_by_J[K]: energy_by_J[K][J] = []
                    energy_by_J[K][J].append(deepcopy(ener[i]))
        
        E_vap, b20_vap = {}, {}
        for K in K_vals:
            E_vap  [K] = []
            b20_vap[K] = []
            ## Import the values form HFB.
            with (FLD_[K] / export_VAP_file).open(mode='r') as f:
                dataVAP = f.readlines()
                Dty_, _ = dataVAP[0].split(',')
                dataVAP = [l.split(OUTPUT_HEADER_SEPARATOR) for l in dataVAP[1:]] 
                
                for h, line in dataVAP:
                    i, b20 = h.split(':')
                    i, b20 = int(i), float(b20)
                    
                    b20_vap[K].append(b20)
                    if i in b20_deform[K]:
                        ## this consideration is to avoid i=-1, b20 = -1.00 refixing
                        ii = b20_deform[K].index(i)
                        if isinstance(b20_deform[K][ii], int):
                            b20_deform[K][ii] = b20 + 0.001
                    
                    res = DataTaurus(z, n, None, empty_data=True)
                    res.setDataFromCSVLine(line)
                    E_vap[K].append(res.E_HFB)
        
        b20_kvap, E_Kvap = {}, {}
        for K in K_vals:
            b20_kvap[K] = []
            E_Kvap  [K] = []
            export_VAP_K_file = export_VAP_file.replace('TESb20_', f'TESb20_K{K}_')
            with open(FLD_[K]/Path(f'{K}_0_VAP')/export_VAP_K_file, 'r') as f:
                dataVAP = f.readlines()
                Dty_, _ = dataVAP[0].split(',')
                dataVAP = [l.split(OUTPUT_HEADER_SEPARATOR) for l in dataVAP[1:]] 
                
                for h, line in dataVAP:
                    i, b20 = h.split(':')
                    i, b20 = int(i), float(b20)
                    for J in b20_by_K_and_J[K].keys():
                        for indx_, ii in enumerate(b20_by_K_and_J[K][J]):
                            if isinstance(ii, int) and ii == i:
                                b20_by_K_and_J[K][J][indx_] = b20 + 0.001
                                        
                    b20_kvap[K].append(b20)
                    if i in b20_deform[K]:
                        ## this consideration is to avoid i=-1, b20 = -1.00 refixing
                        ii = b20_deform[K].index(i)
                        if isinstance(b20_deform[K][ii], int):
                            # if any(filter(lambda x: abs(x - b20)< 2e-3, b20_deform[K])):
                            #     continue
                            b20_deform[K][ii] = b20 + 0.001
                    
                    res = DataTaurus(z, n, None, empty_data=True)
                    res.setDataFromCSVLine(line)
                    E_Kvap[K].append(res.E_HFB)
        
        #-------------------------------------------------------------------- ##
        ##    Figures.
        #-------------------------------------------------------------------- ##
        import matplotlib.pyplot as plt
        # Enable LaTeX in Matplotlib
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # b20_deform = [i-(len(energy_by_J[1])//2) for i in range(len(energy_by_J[1]))]
        # b20_deform = [i for i in range(len(energy_by_J[1]))]
        sp_e_inter_by_K, _1o2IM_inter_by_K, coriolis_coupl_by_K = {}, {}, {}
        model_diff_by_K = {}
        _lims = []
        for K in K_vals:
            
            J_vals = sorted(energy_by_J[K].keys())
            if plot_PAV:
                for J in J_vals:
                    if Jmax_2_plot and (J > Jmax_2_plot): continue 
                    ener = energy_by_J[K][J]
                    # if b20_deform[K].__len__() != ener.__len__():
                    #     printf("   >>> ", J, ener)
                    #     continue
                    ax.plot(b20_by_K_and_J[K][J], ener, label=f'J={J}/2 K={K}/2', 
                            linestyle=__K_LSTYLE[8],#[J_vals.index(J)+1],
                            marker=f"${J}/2$", markersize=11,
                            color=__K_COLORS[K//2])
        
            if len(energy_by_J[K]) > 1 and plot_SCL_interpolation:
                data_5o2 = energy_by_J[K][J_vals[2]] if K == 1 else None
                _b20_values = b20_by_K_and_J[K][J_vals[0]] # b20_deform [K]
                interp = particlePlusRotor_SCL_spectra(energy_by_J[K],
                                                       _b20_values, J_vals, K,
                                                       data_5o2=data_5o2)
                interp, sp_e_inter, _1o2IM_inter, _cor_coupl, model_diff = interp
                sp_e_inter_by_K  [K]    = sp_e_inter
                _1o2IM_inter_by_K[K]    = _1o2IM_inter
                coriolis_coupl_by_K[K]  = _cor_coupl
                model_diff_by_K  [K]    = model_diff
                
                for J in J_vals:
                    if (not PLOT_PPRC) or (Jmax_2_plot and (J > Jmax_2_plot)): 
                        continue 
                    ax.plot(*interp[J], '--', label=f'PPRM-SCL J,K={J}/2 {K}/2',
                                              color=__K_COLORS[K//2])
            # ax.plot(b20_vap [K], E_vap [K], "o-", label=f'HFB(K={K}/2)')
            ax.plot(b20_kvap[K], E_Kvap[K], "o-", 
                    color=__K_COLORS[K//2], label=f'HFB-VAP K={K}/2',
                    linewidth=3)
            _lims = _lims + b20_kvap[K]
        
            #plt.pause(2)
        ax.set_xlabel(r"$\beta_{20}$")
        ax.set_ylabel(r"$E\ (MeV)$")
        _txt = f"{nucl[1]} PAV energies for all K-blockings"
        if PLOT_PPRC and plot_SCL_interpolation: _txt += " and PPRM interpolation"
        _lims = [min(_lims), max(_lims)]
        _rngs = _lims[1] - _lims[0]
        ax.set_xlim( [_lims[0] * 1.1, _lims[1] * 1.5 ] )
        # plt.ylim([-130, -90])
        # plt.xlim([-0.7, 1.2])
        plt.title(f"{nucl[1]} PAV projections for all K and PPRM interpolation")
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.savefig(FOLDER2SAVE / f"plot_pav_pprm_{nucl[0]}.pdf")
        #plt.show()
        
        fig, ax = plt.subplots(1, 3 if len(coriolis_coupl_by_K)>0 else 2,
                               figsize=(10, 4))
        any_ = False
        for K in K_vals:
            if len(energy_by_J[K]) > 1 and plot_SCL_interpolation:
                ax[0].plot(b20_by_K_and_J[K][J_vals[0]], sp_e_inter_by_K[K])
                ax[1].plot(b20_by_K_and_J[K][J_vals[0]], 0.5/_1o2IM_inter_by_K[K], label=f"{K}/2+")
                if len(coriolis_coupl_by_K)>0:
                    ax[2].plot(b20_by_K_and_J[K][J_vals[0]],
                               coriolis_coupl_by_K[K], label=f"{K}/2+")
                any_ = True
        if any_:
            ax[0].set_title("sp energies")
            ax[1].set_title(r"1/2$\mathcal{I}$ Inertia Mom.")
            if len(coriolis_coupl_by_K)>0:
                ax[2].set_title("Decoupling factor")
            plt.suptitle(f"{nucl[1]} PPRM derived quantities for all K from interpolation")
            plt.legend()
            plt.savefig(FOLDER2SAVE / f"derived_pprm_parameters_{nucl[0]}.pdf")
            # plt.show()
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        for K in K_vals:
            i = 0
            for J, model_diff in model_diff_by_K[K].items():
                ax2.semilogy(b20_by_K_and_J[K][J], [abs(x) for x in model_diff],  
                         marker=__J_MARKER[i], color=__K_COLORS[K//2],
                         label=f'$K={K}/2^+$ $J={J}/2^+$')
                i += 1 
        ax2.set_title(f"Correspondence PAV - PPR model SCL ({nucl[1]})")
        ax2.set_ylabel(r"$\delta(E^{PAV} - E^{SCL})_{K, J} [MeV]$")
        ax2.set_xlabel(r"$\beta_{20}$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FOLDER2SAVE / f"model_validity_{nucl[0]}.pdf")
        
        
        if data_hwg:
            par_str = '+' #if  parity
            BaseLevelContainer.RELATIVE_PLOT = False #True
            BaseLevelContainer.ONLY_PAIRED_STATES = False
            BaseLevelContainer.MAX_NUM_OF_SIGMAS  = 3
        
            _graph = BaseLevelContainer()
            _graph.global_title = f"Comparison HWG from different K-blocks {nucl[1]}"
            EnergyLevelGraph.resetClassAttributes()
            for K in K_vals + [-111, ]:
                ## preparing the HWG level scheme
                if data_hwg.get(K, None):
                    level_str = ''.join([x.getSpectrumLines() for x in data_hwg[K]])
                    title = f'K={K}/2 {par_str}' if K !=-111 else 'K-mix'
                    levels_1  = EnergyLevelGraph(title)
                    levels_1.setData(level_str, program='taurus_hwg')
        
                    _graph.add_LevelGraph(levels_1)
            _graph.plot(FOLDER2SAVE / f'hwg_comparison_{nucl[0]}.pdf',
                        figaspect=(6*(2/3)*len(K_vals), 5))

        # _generate_images_hfb_vapK_pav_hwg(
        #     b20_deform, E_vap, b20_kvap, E_Kvap, 
        #     b20_kvap, b20_by_K_and_J, energy_by_J, Jmax_2_plot, 
        #     data_hwg,
        #     plot_PAV=True, plot_SCL_interpolation=True, FOLDER2SAVE=None, nucl=nucl
        # )
        

def _plotPAVresultsSameFolder_mulipleK(folders_2_import, MAIN_FLD, K_vals,
                                       plot_SCL_interpolation=True, plot_PAV=True,
                                       Jmax_2_plot=None):
    """
    :folders_2_import <nucleus>  ((z,n), Folder_BU, export_filename.txt) - templates
    
    This script is meant for multi-K calculations, that employ all the K blocked
    surfaces from the same VAP-false odd-even calculation. The results therefore
    share the same deformations.
    """
    FOLDER2SAVE= Path(MAIN_FLD)#.parent
    for folder_args in folders_2_import:
        index_b20 = {}
        b20_hfb, b20_vap_K, b20_pav_K = [], {}, {}
        data_hfb, data_vap_K, data_pav_K, data_hwg_K = [], {}, {}, {}
        E_hfb, E_vap_K, energy_by_J = [], {}, {}
        
        z, n = folder_args[0]
        nucl = (f"{z+n}{elementNameByZ[z]}", 
                rf"$^{{{z+n}}}${{{elementNameByZ[z]}}}")
        print(f"[{z}, {n}]   ******** ")
        FLD  = folder_args[1].format(MAIN_FLD=MAIN_FLD, z=z, n=n)
        FLD  = Path(FLD)
        
        FLD_KVAP = [(k, FLD / f"{k}_0_VAP") for k in K_vals]
        FLD_KPAV = [(k, FLD / f"{k}_0_PNPAMP_HWG") for k in K_vals]
        FLD_KVAP = dict(FLD_KVAP)
        FLD_KPAV = dict(FLD_KPAV)
        
        if not FLD.exists():
            print(" [ERROR Imprt] Folder not found, skipping!")
            continue
        export_fn = folder_args[2].format(z, n)
        
        ## Read VAP mean field file
        # with open(export_fn, 'r') as f:
        with open(FLD / export_fn, "r") as f:
            for line in f.readlines()[1:]:
                head_, line = line.split(OUTPUT_HEADER_SEPARATOR)
                i, b20 = head_.strip().split()
                index_b20[i] = float(b20)
                b20_hfb.append(float(b20))
                
                obj_ = DataTaurus(z, n, None, empty_data=True)
                obj_.setDataFromCSVLine(line)
                data_hfb.append(obj_)
                E_hfb.append(obj_.E_HFB)
        
        b20_by_K_and_J = {}        
        ## Read VAP-K blocked state - along wiht the PAV folder(diagonal) if exists.
        index_K_elements_log = {}
        for K in K_vals:
            print(" * Importing K =", K)
            b20_vap_K[K], data_vap_K[K] = [], []
            E_vap_K[K] = []
            # with open(export_fn.replace('TESb20_', f'TESb20_K{K}'), 'r') as f:
            exp_fn_k = export_fn.replace('TESb20_', f'TESb20_K{K}_')
            with open(FLD_KVAP[K] / exp_fn_k, "r") as f:
                for line in f.readlines()[1:]:
                    head_, line = line.split(OUTPUT_HEADER_SEPARATOR)
                    i, b20 = head_.strip().split()
                    if i not in index_b20: 
                        print(" [Warning] deformation index", i, "not from VAP.")
                    b20_vap_K[K].append(float(b20))
                    
                    obj_ = DataTaurus(z, n, None, empty_data=True)
                    obj_.setDataFromCSVLine(line)
                    data_vap_K[K].append(obj_)
                    E_vap_K[K].append(obj_.E_HFB)
            
            if FLD_KPAV[K].exists():
                index_K_elements_log[K] = []
                b20_pav_K[K], data_pav_K[K] = [], []
                if 'gcm' in os.listdir(FLD_KPAV[K]):
                    with open(FLD_KPAV[K] / 'gcm', "r") as f:
                        i = 0
                        for line in f.readlines():
                            i += 1
                            wf1, wf2, i1, i2 = line.strip().split()
                            if wf1 == wf2: # diagonal
                                b20 = wf1.replace('.bin','').replace('_', '-')
                                b20 = float(b20)
                                if not any([abs(b - b20) > 0.001 for b in index_b20.values()]):
                                    print(" [Error] deformation not registered:", 
                                          b20,'in',sorted(list(index_b20.values())) )
                                else:
                                    b20_pav_K[K].append(b20)
                                    obj_ = DataTaurusPAV(z, n, FLD_KPAV[K] / f'outputs_PAV/OUT_{i}')
                                    index_K_elements_log[K].append(f"K:{K}, b20:{b20:4.3f}, OUT_{i}")
                                    data_pav_K[K].append(obj_)
                else:
                    print(" ERROR: TODO: define a method to get the diagonal elements without gcm file.")
                
                ## HWG folder in the K-Folder
                if 'HWG' in os.listdir(FLD_KPAV[K]):
                    _pth_hwg = FLD_KPAV[K] / 'HWG'
                    list_dat = list(filter(lambda x: x.endswith('.dat'), 
                                           os.listdir(_pth_hwg)) )
                    data_hwg_K[K] = [DataTaurusMIX(_pth_hwg / f) for f in list_dat]
                
                ## Set the J levels and its deformations for the PAV
                energy_by_J[K] = {}
                b20_by_K_and_J[K] = {}
                for id_, dat in enumerate(data_pav_K[K]):
                    kval = [(i, k) for i, k in enumerate(dat.KJ)]
                    kval = list(filter(lambda x: x[1]==K, enumerate(dat.KJ)))
                    
                    ener, jval, b20J = [], [], []
                    for i, _ in kval:
                        if abs(dat.proj_norm[i]) < 1.0e-8:
                            printf("[Warning] PAV norm = 0 state found. i_b20_def=",
                                  id_, "\n  ***", index_K_elements_log[K][id_])#, data_pav_K[K][id_])
                            continue
                        else:
                            if abs(dat.proj_energy[i]) < 1.0e-8:
                                printf("[Warning] PAV norm/= 0 and E_pav=0: i_b20_def=",
                                      id_, "\n  ***", index_K_elements_log[K][id_])#, data_pav_K[K][id_])
                                continue
                        ener.append(dat.proj_energy[i])
                        jval.append(dat.J[i])
                        if jval[-1] in b20_by_K_and_J[K]:
                            b20_by_K_and_J[K][jval[-1]].append(b20_pav_K[K][id_])
                        else:
                            b20_by_K_and_J[K][jval[-1]] = [b20_pav_K[K][id_], ]
                    
                    for i, J in enumerate(jval):
                        if J not in energy_by_J[K]: energy_by_J[K][J] = []
                        energy_by_J[K][J].append(deepcopy(ener[i]))
            else:
                plot_PAV = False
                print(" [Warning] PAV-results are not present")
            
            ## K-mixing hwg:
            _aux_path_multi = FLD / 'kmix_PNPAMP/HWG' #'PNAMP_HWG/HWG'
            if _aux_path_multi.exists():
                list_dat = filter(lambda x: x.endswith('.dat'), os.listdir(_aux_path_multi))
                list_dat = list(list_dat)
                data_hwg_K[-111] = [DataTaurusMIX(_aux_path_multi / f) for f in list_dat]
        
        _generate_images_hfb_vapK_pav_hwg(
            b20_hfb, E_hfb, b20_vap_K, E_vap_K, 
            b20_pav_K, b20_by_K_and_J, energy_by_J, Jmax_2_plot, 
            data_hwg_K,
            plot_PAV=True, plot_SCL_interpolation=True, FOLDER2SAVE=FOLDER2SAVE,
            nucl=nucl
        )
        

def _plotMultiK_vertical(folders_2_import, MAIN_FLD, K_vals, 
                         Jmax2plot=None, onlyPrintJ=None):
    """
    plotting deformations _pav results and HWG in case of done.
    plotting also the intrinsic-bloqued K surfaces
    """
    def _skipThisJ(jval, onlyPrintJ):
        if onlyPrintJ == None: 
            return False
        elif isinstance(onlyPrintJ, (list, tuple)): 
            return not jval in onlyPrintJ
        else: return jval != onlyPrintJ
    
    FOLDER2SAVE= Path(MAIN_FLD) #.parent
    if not Jmax2plot: Jmax2plot = K_vals[-1]
    for folder_args in folders_2_import:
        b20_deform, b20_index, data_int, data_pav, data_hwg = {}, {}, {}, {}, {}
        FLD_K, FLD_PNPAMP, FLD_HWG = {}, {}, {}
        BU_fld = None
        for K_val in K_vals:
            list_dat_file = None
            
            for i, arg in enumerate(folder_args):
                if i == 0:
                    z, n = arg
                    nucl = (f"{z+n}{elementNameByZ[z]}", 
                            rf"$^{{{z+n}}}${{{elementNameByZ[z]}}}")
                elif i == 1:
                    arg = arg.format(MAIN_FLD=MAIN_FLD, z=z, n=n)
                    BU_fld = Path(arg)
                    FLD_K[K_val] = Path(arg) / f"{K_val}_0_VAP" 
                elif i == 2:
                    if arg.count('{') == 2: arg = arg.replace('_TESb20', '_TESb20_K{}')
                    export_VAP_file = arg.format(K_val, z, n)
                    FLD_K[K_val] = FLD_K[K_val] / export_VAP_file
                
            # if not list_dat_file:
            #     list_dat_file = FLD_K[K_val].parent / Path('list.dat')
            #     with open(list_dat_file, 'r') as f:
            #         _bdat = [_b.strip().replace('.bin', '') for _b in f.readlines()]
            #         b20_deform[K_val] = [float(_b.replace('_', '')) for _b in _bdat]
            with open(FLD_K[K_val], 'r') as f:
                aux = f.readlines()
                constr_ = aux[0].split(',')[1].strip()
                b20_deform[K_val] = []
                aux_dat = []
                for line in aux[1:]:
                    head_, args = line.split(OUTPUT_HEADER_SEPARATOR)
                    i, def_ = head_.split(':')
                    i, def_ = int(i), float(def_)
                    if i in b20_index:
                        if b20_index[i] != def_: 
                            print(" [WRN!] deformation does not match", b20_index, i, def_)
                    else: b20_index[i] = def_
                    
                    obj = DataTaurus(z, n, None, empty_data=True)
                    obj.setDataFromCSVLine(args)
                    aux_dat.append(obj)
                    b20_deform[K_val].append(def_)
                data_int[K_val] = aux_dat
        
        b20_dat_pav, data_pav = {}, {}
        for i_def, b20 in b20_index.items():
            ## read the deformations for the diagonal PAVs
            b20_dat_pav[i_def] = b20
            i_def_s   = str(i_def).replace('-', '_')
            fld_i_pav = BU_fld / Path(f'def{i_def_s}_PAV')
            data_pav[i_def] = {}
            with open(fld_i_pav / 'gcm_diag', 'r') as f:
                aux_diag = [p.strip() for p in f.readlines()]
                assert len(K_vals)==len(aux_diag), f"K missing in def={i_def} {b20}"
                for i, K_val in enumerate(K_vals):
                    obj = DataTaurusPAV(z, n, fld_i_pav / aux_diag[i] / 'OUT')
                    data_pav[i_def][K_val] = obj
            
            ## read the HWG for each deformation
            fld_hwg = fld_i_pav / 'HWG'
            if fld_hwg.exists:
                list_Jp = filter(lambda x: x.endswith('.dat'), os.listdir(fld_hwg))
                data_hwg[i_def] = {}
                for jf in list_Jp:
                    j = int(jf.replace('.dat', ''))
                    data_hwg[i_def][j] = DataTaurusMIX(fld_hwg / jf)
                # it asumes the order of the folders for the order of increasing K
            
        ## Intrinsic states for all K and diagonal PAV
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        # fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        for ik, K_val in enumerate(K_vals):
            y = [x.E_HFB for x in data_int[K_val]]
            kwargs = {'label': f'VAP K={K_val}', 'markerfacecolor':'none'}
            # ax[0].plot(b20_deform[K_val], y, 'o-', **kwargs)
            ax[0].plot(b20_deform[K_val], y, 'o-', **kwargs)
            
            print("K=", K_val)
            
            x_j, y_j = {}, {}
            for i in sorted(b20_dat_pav.keys()):
                for jval in range(K_val, Jmax2plot+1, 2):
                    if not jval in x_j: x_j[jval] = []
                    if not jval in y_j: y_j[jval] = []
                    
                    e_j = data_pav[i][K_val].J.index(jval)
                    e_j = data_pav[i][K_val].proj_energy[e_j]
                    
                    x_j[jval].append(b20_dat_pav[i])
                    y_j[jval].append(e_j)
            
            for jval in range(K_val, Jmax2plot+1, 2):
                if _skipThisJ(jval, onlyPrintJ): continue
                
                # if _skipThisJ(K_val,onlyPrintJ): continue
                ax[0].plot(x_j[jval], y_j[jval],
                           linestyle = __K_LSTYLE[ik],
                           marker =f"${jval}/2$", markersize=11, #__J_MARKER[ik],
                           color  = __K_COLORS[jval//2], label=f"J=${jval}^+$ (K={K_val})")
                if jval >= K_val:
                    if _skipThisJ(jval, onlyPrintJ): continue
                    # if _skipThisJ(K_val,onlyPrintJ): continue
                    ax[1].plot(x_j[jval], y_j[jval],
                               linestyle = __K_LSTYLE[ik],
                               marker=f"${jval}/2$", markersize=11, # = __J_MARKER[ik],
                               color  ='black', label=f"J=${jval}^+$ (K={K_val})")
            
        if fld_hwg:
            MAX_spectra = 3
            x_sig, y_sig = {}, {}
            e_tops = [9999, -9999]
            for jval in range(1, Jmax2plot+1, 2):
                indx_attr = DataTaurusMIX.ArgsEnum.i
                ener_attr = DataTaurusMIX.ArgsEnum.energy
                x_sig[jval] = [list() for sig in range(MAX_spectra)]
                y_sig[jval] = [list() for sig in range(MAX_spectra)]
                
                id_fr0 = 0
                for id_, obj_dict in data_hwg.items():
                    indx_sigma = obj_dict[jval].energy_spectrum.get(indx_attr, [])
                    eners_ = obj_dict[jval].energy_spectrum.get(ener_attr, [])
                    for sig in range(MAX_spectra):
                        x_sig[jval][sig].append(b20_dat_pav[id_])
                        if sig < len(eners_):
                            y_sig[jval][sig].append(eners_[sig])
                        else:
                            x_sig[jval][sig].pop()
                            # if sig > 0 and len(y_sig[jval][sig-1]) > 0:
                            #     y_sig[jval][sig].append(y_sig[jval][0][id_fr0] + 30)
                    id_fr0 += 1
                for sig in range(MAX_spectra-1, -1, -1):
                    if y_sig[jval][sig].__len__() == 0:
                        y_sig[jval].pop()
                
                for sig in range(len(y_sig[jval])):
                    e_tops[0] = min(e_tops[0], min(y_sig[jval][sig]))
                    e_tops[1] = max(e_tops[1], max(y_sig[jval][sig]))
                    if len(x_sig[jval][sig]) != len(y_sig[jval][sig]):
                        print("[ERR] does not match j, sig=", jval, sig)
                        continue
                    if _skipThisJ(jval, onlyPrintJ): continue
                    print("j", jval, "  sig=", sig, len(x_sig[jval][sig]), 
                          len(y_sig[jval][sig]))
                    ax[1].plot(x_sig[jval][sig], y_sig[jval][sig],
                               # linestyle = __K_LSTYLE[jval//2],
                               marker = __J_MARKER[jval//2],
                               color  =__K_COLORS[sig],
                               label=f'J=${jval}^+({sig})$')
        
        print("DONE")
        MARGIN_ = 2.0
        e_tops = (e_tops[0] - MARGIN_, e_tops[1] + MARGIN_)
        for i in range(2): 
            ax[i].set_ylim( *e_tops )
            ax[i].set_xlabel(r"$\beta_{20}$")
            ax[i].legend()
        ax[0].set_ylabel(r"E (MeV)")
        plt.suptitle(f"{nucl[1]} vertical mixing")
        if onlyPrintJ: plt.suptitle(f"{nucl[1]} vertical mixing/ Only J={onlyPrintJ}")
        plt.tight_layout()
        
        fn_ = f'vertical_gcmAll_{nucl[0]}.pdf'
        if onlyPrintJ: fn_ = f'vertical_gcm_J{onlyPrintJ}_{nucl[0]}.pdf'
        plt.savefig(FOLDER2SAVE / Path(fn_))
        if not onlyPrintJ: plt.show()
        


if __name__ == '__main__':
    #===========================================================================
    # # PLOT FROM FOLDERS
    #===========================================================================
    if True:
        SUBFLD_ = '../BU_folder_B1_MZ3_z2n1/PNAMP/'
        
        K_val  = 11
        
        list_dat_file = f'list_k{K_val}_pav.dat'
        
        MAIN_FLD = '..'
        MAIN_FLD = '../DATA_RESULTS/SD_Kblocking/K_blocking_noPAV'
        MAIN_FLD = '../DATA_RESULTS/SD_Kblocking/K{K_val}_block_PAV'
        MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_fewDefs/K{K_val}_block_PAV'
        # MAIN_FLD = '../DATA_RESULTS/example_singleJ/K{K_val}_block_PAV'
        MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Cl'
        # MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Mg1ststateSwap_multiK'
        
        nuclei = [( 7, 8 + 2*i)  for i in range(0, 1)] # 7
        # nuclei = [( 9, 8 + 2*i) for i in range(0, 1)] # 7
        # nuclei = [(11, 8 + 2*i) for i in range(0, 1)] # 7
        nuclei = [(12,11 + 2*i) for i in range(0, 6)] # 6
        # nuclei = [(15,  8 + 2*i) for i in range(0, 6)]
        nuclei = [(17,10 + 2*i) for i in range(5, 6)]
        # nuclei = [(9, 20), ]
        
        folders_2_import = [
            # ((2, 1), '../BU_folder_B1_MZ3_z2n1/PNAMP/', DataTaurusPAV),
            #((2, 3), '../BU_folder_B1_MZ3_z2n3/PNAMP/', DataTaurusPAV),
            # ((8, 11), f'{MAIN_FLD}/BU_folder_B1_MZ3_z8n11/', 'export_TESb20_z{}n{}_B1_MZ3.txt'),
            
            ((z, n), '{MAIN_FLD}/BU_folder_B1_MZ4_z{z}n{n}/', 'export_TESb20_z{}n{}_B1_MZ4.txt') for z,n in nuclei
            # ((z, n), 
            #  '{MAIN_FLD}/BU_folder_B1_h11o2_z{z}n{n}/', 
            #  'export_TESb20_z{}n{}_B1_h11o2.txt') for z,n in nuclei
        ]
        # K_val = 3
        # _plotPAVresultsFromFolders(folders_2_import, MAIN_FLD, K_val, parity=0,
        #                            plot_SCL_interpolation=False)
        # raise Exception("STOP HERE")
        
        K_vals  = [1, 3, 5, 7]
        # _plotPAVresultsFromFolders_mulipleK(folders_2_import, MAIN_FLD, K_vals,
        #                                     plot_SCL_interpolation=1,
        #                                     Jmax_2_plot=9)
        _plotPAVresultsSameFolder_mulipleK(folders_2_import, MAIN_FLD, K_vals,
                                            plot_SCL_interpolation=1,
                                            Jmax_2_plot=9)
        raise Exception("STOP HERE")
        
        # K_vals  = [1, 3, 5, 7]
        # for j in range(3, 12, 2):
        #     _plotMultiK_vertical(folders_2_import, MAIN_FLD, K_vals, 
        #                          Jmax2plot=9, onlyPrintJ=j)
        # _plotMultiK_vertical(folders_2_import, MAIN_FLD, K_vals, Jmax2plot=7)
        # raise Exception("STOP HERE")
        
        K_MAX = 11
        MZMAX = 4
        nuclei = [
            #(0, 3), (2, 3)
            # (8, 9), (8, 11), (8, 13), 
            #(9, 8), (9, 10), (9, 12), (9, 14),
            # (10, 9), (10, 11), (10, 13), (10, 15),
            # (11, 8), (11, 10), (11, 12), (11, 14),# 
            (12, 11), (12, 13), (12, 15), #(12, 17), # (12, 9),
            # (13, 8), (13, 10), (13, 12), (13, 14), (13, 16), (13, 18),
            # (14, 9), (14, 11), (14, 13), (14, 15), (14, 17), (14, 19),
        ]
        folders2import = dict([
            ((z, n), f'BU_folder_B1_MZ{MZMAX}_z{z}n{n}/')
            for z, n in nuclei])
        
        _plotscript1_OEK_withNoProjection(nuclei, K_MAX, MZMAX, 
                                          folders2import=folders2import, 
                                          main_folder='../DATA_RESULTS/SD_Kblocking_results/K1_block_PAV/')
        
    raise Exception("STOP HERE")
    #===========================================================================
    # PLOT OF DEFORMATION SURFACES
    #===========================================================================
    
    SUBFLD_ = 'Mg_GDD_test/34/HFB/' 
    # SUBFLD_ = 'Mg_B1/22/VAP9/' 
    SUBFLD_ = 'B1/O/17/HFB/' 
    
    nuclei = [
        # (12, 8),
        # (12, 10),
        # (12, 12), 
        # (12, 14), 
        # (12, 16), 
        # (11, 18), 
        # (12, 19), 
        # (13, 18), 
        # (15, 16), 
        # (12, 21),
        (8, 9),
        # (12, 22),
        ]
    K_MAX = 5
    MZMAX = 2
    _plotscript1_OEK_withNoProjection(nuclei, K_MAX, MZMAX)
    # -------------------------------------------------------------------------
    
    for z, n in nuclei:
        pass
        
        # """
        # Script to export for the json by 
        # """
        # export_ = {}
        # for file_ in files_:
        #
        #     gdd_ = file_.replace('.txt', '').split("_")[-1]
        #     if not gdd_.isdigit(): gdd_ = 'edf'
        #     export_[gdd_] = {}
        #
        #     if not file_ in plt_obj._x_values:
        #         del export_[gdd_]
        #         continue
        #     ii = 0 
        #     for k_b, b20 in plt_obj._x_values[file_].items():
        #         r : DataTaurus = plt_obj._results[file_][ii]
        #         aux = {
        #             'B10': [r.b10_p, r.b10_n, r.b10_isoscalar, r.b10_isovector],
        #             'B20': [r.b20_p, r.b20_n, r.b20_isoscalar, r.b20_isovector],
        #             'B22': [r.b22_p, r.b22_n, r.b22_isoscalar, r.b22_isovector],
        #             'B30': [r.b30_p, r.b30_n, r.b30_isoscalar, r.b30_isovector],
        #             'B30': [r.b32_p, r.b32_n, r.b32_isoscalar, r.b32_isovector],
        #             'B40': [r.b40_p, r.b40_n, r.b40_isoscalar, r.b40_isovector],
        #             # 'B42': [r.b42_p, r.b42_n, r.b42_isoscalar, r.b42_isovector],
        #             # 'B44': [r.b44_p, r.b44_n, r.b44_isoscalar, r.b44_isovector],
        #             'E_1b':[r.kin_p, r.kin_n, r.kin,],
        #             'E_hf':[r.hf_pp, r.hf_pp, r.hf_pn, r.hf],
        #             'E_pp':[r.pair_pp, r.pair_nn, r.pair_pn, r.pair],
        #             'E_hfb':[r.E_HFB_pp, r.E_HFB_nn, r.E_HFB_pn, r.E_HFB],
        #             'Jx' : [r.Jx, r.Jx_2, r.Jx_var],
        #             'Jy' : [r.Jy, r.Jy_2, r.Jy_var],
        #             'Jz' : [r.Jz, r.Jz_2, r.Jz_var],
        #             'r': [r.r_p, r.r_n, r.r_isoscalar, r.r_isovector, r.r_charge],
        #             'Parity': [1.0], 
        #         }
        #         export_[gdd_][f"{b20:3.3f}"] = deepcopy(aux)
        #         ii += 1
        # import json
        # with open(FLD_+f'results_gdd_b20_z{z}n{n}.json', 'w+') as fp:
        #     json.dump(export_, fp)
    
    
    # 0/0
    # SUBFLD_ = 'BU_folder_hamil_gdd_000_z12n12/'
    # # # SUBFLD_ = 'SDnuclei_MZ5/'
    # Plotter1D_CanonicalBasis.setFolderPath2Import('../DATA_RESULTS/Beta20/Mg_GDD_test/24/HFB/')
    # files_ = ["export_TESq20_z12n12_hamil_gdd_000.txt", ]
    # plt_obj2 = Plotter1D_CanonicalBasis(files_, iter_procedure_sweep=True)
    # plt_obj2.setXlabel("b20 value")
    # attr2plot_list = [
    #         'E_HFB', 
    #         # 'E_HFB_pp', 'E_HFB_nn', 'E_HFB_pn', 
    #         # 'hf', 'hf_pp', 'hf_nn', 'hf_pn',
    #         # 'pair_pp', 'pair_nn', 'pair_pn', 
    #         # *pair_constr,
    #         # 'beta_isoscalar', 'gamma_isoscalar',
    #         # 'b30_isoscalar',  'b32_isoscalar',
    #         # 'Jz', 'var_n', 'var_p'
    #         ]
    #
    # attr2plot = [
    #     Plotter1D_CanonicalBasis.AttrEigenbasisEnum.h,
    #     # Plotter1D_CanonicalBasis.AttrEigenbasisEnum.avg_neutron,
    #     # Plotter1D_CanonicalBasis.AttrEigenbasisEnum.v2,
    #     # Plotter1D_CanonicalBasis.AttrEigenbasisEnum.fermi_energ_prot,
    #     ]
    # for ind_, attr_ in enumerate(attr2plot):
    #     plt_obj2.defaultPlot(attr_, index_2_print=None,
    #                         attr2plotExport=attr2plot_list,
    #                         show_plot=ind_==len(attr2plot)-1)
    # _=0