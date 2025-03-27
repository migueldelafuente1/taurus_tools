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
from tools.helpers import elementNameByZ, printf, OUTPUT_HEADER_SEPARATOR,\
    readAntoine, QN_1body_jj, prettyPrintDictionary

from copy import deepcopy
from tools.data import DataTaurus, DataTaurusPAV, DataTaurusMIX, \
    CollectiveWFData, OccupationsHWGData, OccupationNumberData
from tools.plotter_levels import EnergyLevelGraph, BaseLevelContainer,\
    getAllLevelsAsString, MATPLOTLIB_INSTALLED
from plotting_scripts.plots1DLevels import _EnergyLevelSimpleGraph

if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    # Enable LaTeX in Matplotlib
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

__K_COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black']
__KJ_COLORS = [
    []
    ]
# __K_LSTYLE = ['-', '--', ':', '-.', ' ', '']*2
# __K_LSTYLE = ['solid', 'dashed', 'dashdot', 'dotted', 
#               (0, (1,10)), (0, (5, 5)), (5, (10,3)), (0, (3,5,1,5)), (0, (3,5,1,5,1,5))]
__K_LSTYLE = ['solid', (0, (5,1)), (5, (10,3)), 'dashed', 'dashdot', 
            (0, (3,1,1,1)), (0, (3,5,1,5)), (0, (5,10)), 'dotted',
            (0, (3,10,1,10)), (0, (3,10,1,10)), (0, (1,10))]
# __J_MARKER = ['.',  'd', 's', 'h', 'v', '^','+',"1", "2", "3", "x"]
__J_MARKER = ['.',  'P', '*', 'X', 's', 'd', 'p', 'h', 'v', '^','+', "x"]

GLOBAL_TAIL_INTER = ''
INTERACTION_TITLE = ''
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
        valid_kps = []
        for P in (0, 1):
            for K in range(1, K_MAX+1, 2):
                valid_kps.append( (K, P)) 
        for K, P in valid_kps:
            _fld = f"{K}_{P}_VAP".replace('-', '_')
            if os.path.exists(_fld): continue
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

def particlePlusRotor_NCL_spectra(data_j, b20_deform, J2_vals, K):
    """
    Fitting of the first states of the PAV- spectra to the 
    Weak-coupling-Limit formula for single-j case.
    
    K : fixes the limit for the PAV - J states to count.
    """
    from scipy.interpolate import interp1d
    J_vals = [j/2 for j in J2_vals]
    j_sp   = [1, 3, 5] # intrinsic j values
    if K > max(j_sp): 
        return    
    data_Esorted_by_def = []
    y01, y02, yj1, yj2 = [],[],[],[]
    J2_vals_2 = filter(lambda J: len(data_j[J]) == len(b20_deform), J2_vals)
    J2_vals_2 = list(J2_vals_2)
    for i in range(len(data_j[J2_vals[0]])):
        srtd_list = [(J, data_j[J][i]) for J in J2_vals_2]
        srtd_list = sorted(srtd_list, key=lambda x: x[1])
        
        data_Esorted_by_def.append(deepcopy(srtd_list))
        y01.append(srtd_list[0][1])
        y02.append(srtd_list[1][1])
        yj1.append(srtd_list[0][0] / 2.)
        yj2.append(srtd_list[1][0] / 2.)
        
    dim_ = data_Esorted_by_def.__len__()        
    
    x_new = np.linspace(min(b20_deform), max(b20_deform), 101, endpoint=True)
    y01, yj1 = np.array(y01), np.array(yj1)
    y02, yj2 = np.array(y02), np.array(yj2)
    
    model_diff, model_ener = {}, {}
    E_intr, _2I_imom = {}, {}
    for _2j in j_sp:
        _j = _2j / 2
        
        A, B = np.zeros(len(y01)), np.zeros(len(y01))
        
        for i in range(dim_):
            b  = (y02[i] - y01[i])
            b /= (yj2[i] - yj1[i]) * (yj2[i] + yj1[i] - _2j - 1)
            a = y01[i] - b * ((yj1[i]**2 - (_2j + 1)*yj1[i] - _j*(_j - 1)) )
            
            A[i] = a
            B[i] = b
        
        E_intr  [_2j] = A
        _2I_imom[_2j] = B
        
        model_diff[_2j] = {}
        model_ener[_2j] = {}
        for i in range(dim_):
            e_data = dict(data_Esorted_by_def[i])
            for J in J_vals:
                _2J = int(2*J) 
                if _2J < K: continue
                e = E_intr[_2j][i] + _2I_imom[_2j][i]*(J - _j)*(J - _j + 1)
                
                e_d = e_data[_2J]
                
                if i == 0:
                    model_ener[_2j][_2J] = [e, ]
                    model_diff[_2j][_2J] = [e - e_d, ]
                else:
                    model_ener[_2j][_2J].append(e)
                    model_diff[_2j][_2J].append(e - e_d)
    
    data_interpolated = {}
    ## SKIP plot
    return  data_interpolated, A, B, model_diff
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    _MARKERS = '.o*^v+*'
    _COLORS  = ['red', 'blue', 'green', 'magenta', 'cyan', ]
    for _2J in J2_vals:
        ## data - from PAV
        y0 = np.array(data_j[_2J]) - 0.05 * np.ones(dim_) 
        ax.plot(b20_deform, y0, color='black',
                linestyle='--', marker=_MARKERS[(_2J-1)//2],
                label=f'data J = {_2J}/2')
    
    for _2j in j_sp:
        data_interpolated[_2j] = {}
        for _2J, y_j in model_ener[_2j].items():
            
            ax.plot(b20_deform, y_j, color=_COLORS[(_2j-1)//2],
                    linestyle='--', #marker=_MARKERS[(_2J-1)//2],
                    marker=f"${_2J}/2$", markersize=12,
                    label=f'NCL J = {_2J}/2')
            
            # J = J_vals[i]
            # interp_func = interp1d(b20_deform, y_j, kind='cubic') # 'quadratic'
            # # Generate points for the interpolated curve
            # y_new = interp_func(x_new)
            # data_interpolated[J2_vals[i]] = (x_new, y_new)
    ax.set_title(f"NCL for PPR model from K={K}, and 2j in {j_sp}")
    ax.legend()
    plt.show()
    
    return data_interpolated, A, B, model_diff

def particlePlusRotor_RAL_spectra(data_k12J):
    """
    Proposal to evaluate the Coriolis spectra from K-mixing data.
    """
    return
    J_max   = 13
    
    I2_vals = [i for i in range(1, J_max+1, 2)]
    K_vals  = list(data_k12J[len(data_k12J) // 2].keys())
    K_vals.remove('b20') 
    K_vals  = list(set(map   (lambda x: x[0], K_vals)))
    K_vals.sort()
    
    b20_deforms = [data_k12J[i]['b20'] for i in range(len(data_k12J))]
    
    waveFunct_by_Ij = {}
    Hmatrix_by_Ij   = {}
    paramsRAL_by_j  = {}
    for j_sp in K_vals:
        Hmatrix_by_Ij  [j_sp] = np.zeros( (len(K_vals), len(K_vals)) )
        waveFunct_by_Ij[j_sp] = np.zeros( len(K_vals) )
        paramsRAL_by_j [j_sp] = [0, 0, {}, {}] # Inertia, E_intr, dict{<Om|j+-|Om'>: vals}
    
    enerb20_by_k1k2J = {}
    for i, dict_bkk in data_k12J.items():
        if [] in dict_bkk.values(): continue ## States with <1|2> = 0
        
        k_tuples = list(dict_bkk.keys())
        k_tuples.remove('b20')
        k_min = sorted(list(filter(lambda x: x[0]==x[1], k_tuples)))
        k_min = k_min[0][0]
        
        I1, I2 = k_min, k_min+2
        
        # Diagonal values
        for j_sp in K_vals[K_vals.index(k_min):]:
            k = k_min
            cte_1 = (j_sp/2*(j_sp/2 + 1) - (k/2)**2) \
                    + (I1/2*(I1/2 + 1) - k/2*(k/2 + 1))
            cte_2 = (j_sp/2*(j_sp/2 + 1) - (k/2)**2) \
                    + (I2/2*(I2/2 + 1) - k/2*(k/2 + 1))
                    
            e_jk1 = dict_bkk[(k, k)][I1][1]
            e_jk2 = dict_bkk[(k, k)][I2][1]
            
            inertia = (e_jk1 - e_jk2) / (cte_1 - cte_2)
            e_intr  = e_jk1 - inertia * cte_1
            
            paramsRAL_by_j[j_sp][0] = inertia
            paramsRAL_by_j[j_sp][1] = e_intr
        
        I = I1 if I1 > 1 else I2
        
        # Coriolis        
        k2fit = [k for k in range(1, I+1, 2)]
        k2fit = [(k, k+2) for k in k2fit[:-1]]
        
        for k1, k2 in k2fit:
            n = dict_bkk[(k1, k2)].__len__()
            JbyK = dict([(dict_bkk[(k1, k2)][i][0], i) for i in range(n)])
            
            e_jk12 = dict_bkk[(k1, k2)][JbyK[I]][1]
            # cte  = (I/2*(I/2 + 1) - k1/2*(k1/2 + 1))
            cte = np.sqrt((I/2 - k1/2) - (I/2 + k1/2 + 1))
            
            for j_sp in K_vals:
                inertia = paramsRAL_by_j[j_sp][0]
                paramsRAL_by_j[j_sp][2][(k1, k2)] = e_jk12 / (cte * inertia)
            
        
        ## Coriolis formula for other Is
        b20 = dict_bkk['b20']
        for k1k2, jvals in dict_bkk.items():
            if k1k2 == 'b20': continue
            if not k1k2 in enerb20_by_k1k2J:
                enerb20_by_k1k2J[k1k2] = dict([(x[0],([b20,], [x[1],], [abs(x[2]),])) 
                                               for x in jvals])
            else:
                for x in jvals:
                    if not x[0] in enerb20_by_k1k2J[k1k2]:
                        enerb20_by_k1k2J[k1k2][x[0]] = ([], [], [])
                    enerb20_by_k1k2J[k1k2][x[0]][0].append(b20)
                    enerb20_by_k1k2J[k1k2][x[0]][1].append(x[1])
                    enerb20_by_k1k2J[k1k2][x[0]][2].append(abs(x[2]))
        
    _ = 0
    
    _COLORS = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow']
    _MARKRS, list_mark_12 = '.v*^do+x'*2, {}
    fig, ax   = plt.subplots(1, len(I2_vals), figsize=(6+len(I2_vals)//2, 6))
    fig2, ax2 = plt.subplots(1, len(I2_vals), figsize=(6+len(I2_vals)//2, 6))
    for k1k2, jvals in enerb20_by_k1k2J.items():
        k1, k2 = k1k2
        for j, prt_vals in jvals.items():
            if (k1==k2): 
                ax[j//2].plot(prt_vals[0], prt_vals[1], marker=_MARKRS[k1//2], 
                              color=_COLORS[k1//2], label=f'J:{j}(2Om={k1}) diag')
                ax2[j//2].semilogy(prt_vals[0], prt_vals[2], marker=_MARKRS[k1//2], 
                               color=_COLORS[k1//2], label=f'J:{j}(2Om={k1}) diag')
            else:
                if not k1k2 in list_mark_12: list_mark_12[k1k2] = len(list_mark_12)
                
                ax[j//2].plot(prt_vals[0], prt_vals[1], 
                              marker=_MARKRS[list_mark_12[k1k2]], 
                              color='black', label=f'J:{j}(2Om={k1},{k2})')
                ax2[j//2].semilogy(prt_vals[0], prt_vals[2], 
                              marker=_MARKRS[list_mark_12[k1k2]], 
                              color='black', label=f'J:{j}(2Om={k1},{k2})')
    
    for j in I2_vals:
        ax[j//2].set_title(f'I={j}/2')
        ax[j//2].legend()
        
        ax2[j//2].set_title(f'I={j}/2')
        ax2[j//2].legend()
        
        ax2[j//2].set_ylim((0.001, 0.6))
    fig.suptitle("$E^{PAV}$")
    fig2.suptitle("Norm overlap")
    plt.tight_layout()
    plt.show()
    _ =0
    
    ## plot
    
    
    

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


def _get_dataKmix_byDef(PVAP_FLD):
    """
    from file gcm (wf by deform and K) 
    :returns:
    {'def_index': 
        {'b20': <float> value
         (k1, k2): 
            [(<int> J, <float> E projected, <float> Norm projected),
             ...
            ]
        ...
        }, ...
    }
    """
    gcm_file = PVAP_FLD[-135].parent / 'gcm'
    pav_fldr = PVAP_FLD[-135].parent / 'outputs_PAV'
    b20_data, data_K1K2 = {}, {}
    with open(gcm_file, 'r') as f:
        data = f.readlines()
        for k, line in enumerate(data):
            b1, b2, i1, i2 = line.split()
            b1_str = b1.replace('.bin', '')[:-2]
            b1 = b1.replace('.bin', '').split('_')
            b2 = b2.replace('.bin', '').split('_')
            b1, k1 = ((-1)**(b1[0]=='')) * float(b1[-2]), int(b1[-1])
            b2, k2 = ((-1)**(b2[0]=='')) * float(b2[-2]), int(b2[-1])
            b1, b2 = round(b1, 3), round(b2, 3)
            if abs(b1 - b2) > 0.0001: continue
            else:
                if not b1_str in b20_data:
                    i = len(b20_data)
                    b20_data [b1_str] = i
                    data_K1K2[i] = {'b20': b1, (k1, k2): None}
                else:
                    i = b20_data[b1_str]
                    if not (k1, k2) in data_K1K2[i]:
                        data_K1K2[i][(k1,k2)] = None
                pth = pav_fldr / f'OUT_{k+1}'
                obj = DataTaurusPAV(0,0,pth)
                
                jEn = [(obj.J[i], obj.proj_energy[i], obj.proj_norm[i]) 
                       for i in range(obj.dim)]
                jEn = filter(lambda x: abs(x[2]) > 1.0e-5, jEn)
                data_K1K2[i][(k1,k2)] = list(jEn)
    
    return data_K1K2

from scipy.interpolate import CubicSpline
def _smooth_curve_splines(x, y):
    x, y = np.array(x), np.array(y)
    # Create the cubic spline interpolation
    cs = CubicSpline(x, y)
    # Generate a smooth curve
    x_new = np.linspace(x.min(), x.max(), 500)
    y_new = cs(x_new)
    return x_new, y_new

def _generate_images_hfb_vapK_pav_hwg(b20_hfb, data_hfb, b20_vap_K, E_vap_K, data_vap_K,
                                      b20_pav_K, b20_by_K_and_J, energy_by_J, 
                                      Jmax_2_plot, data_hwg_K,
                                      parity=0, plot_PAV=True, plot_PPR_interpolation=True, 
                                      FOLDER2SAVE=None, nucl='***',
                                      **kwargs):
    """
    Auxiliary method to print VAP-PAV-HWG merging results from folders in the 
    same or different folders, methods to organize the arguments must be 
    set outside.
    """
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15

    global GLOBAL_TAIL_INTER
    PLOT_PPRC   = False
    Jmax_2_plot = 33 if not Jmax_2_plot else Jmax_2_plot
    _hIntKJ = all([k%2 == 1 for k in b20_by_K_and_J.keys()])
    _frac2 = '/2' if _hIntKJ else ''
    _ALL_Jvals = [2*i + 1 for i in range(10)]
    
    ## Verify the VAP-K surfaces
    if not plot_PAV:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        E_hfb = [dat.E_HFB for dat in data_hfb[0]]
        ax.plot(b20_hfb[0], E_hfb, '.', label='false o-e', linestyle='None',)
        for K in K_vals:
            ax.plot(b20_vap_K[K], E_vap_K[K], '.', 
                    color=__K_COLORS[K//2], label=f'k={K}',linestyle='None',)
            x_spl, y_spl = _smooth_curve_splines(b20_vap_K[K], E_vap_K[K])
            ax.plot(x_spl, y_spl, color=__K_COLORS[K//2], ) #[8], #)
        ax.legend()
        ax.grid()
        ax.set_title("V-VAP K blocked preliminary TES")
        plt.show()
    
    fig, ax   = plt.subplots(1, 1, figsize=(5, 5))
    fig0, ax0 = plt.subplots(1, 1, figsize=(5, 5))
    # b20_deform = [i-(len(energy_by_J[1])//2) for i in range(len(energy_by_J[1]))]
    # b20_deform = [i for i in range(len(energy_by_J[1]))]
    sp_e_inter_by_K, _1o2IM_inter_by_K, coriolis_coupl_by_K = {}, {}, {}
    model_diff_by_K = {}
    _lims, _y_lims = [], [+999999, -999999]
    for K in K_vals:
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        if plot_PAV and (K in energy_by_J):
            J_vals = filter(lambda x: x <= Jmax_2_plot, energy_by_J[K].keys())
            J_vals = sorted(J_vals)
            for J in J_vals:
                ener = energy_by_J[K][J]
                _y_lims = min(_y_lims[0], min(ener)), max(_y_lims[1], max(ener))
                
                # if b20_deform[K].__len__() != ener.__len__():
                #     printf("   >>> ", J, ener)
                #     continue
                if False:
                    ax.plot(b20_by_K_and_J[K][J], ener, label=f'J={J}{_frac2} K={K}{_frac2}', 
                            linestyle=__K_LSTYLE[8],#[J_vals.index(J)+1],
                            marker=f"${J}{_frac2}$", markersize=11,
                            color=__K_COLORS[K//2])
                
                x = list(filter(lambda x: x < 0, b20_by_K_and_J[K][J]))
                y = [ener[i] for i in  range(len(x))]
                ax2.plot(x, y, label=f'J={J}{_frac2} K={K}{_frac2}', 
                         marker=__J_MARKER[_ALL_Jvals.index(J)],
                         linestyle='None', #__K_LSTYLE[(_ALL_Jvals.index(J)+1)],
                         # marker=f"${J}$", markersize=11,
                         color=__K_COLORS[K//2], alpha=(1 - (J-K+1)/15))
                x_spl, y_spl = _smooth_curve_splines(x, y)
                ax2.plot(x_spl, y_spl, 
                         color=__K_COLORS[K//2], alpha=(1 - (J-K+1)/15),
                         linestyle=__K_LSTYLE[(_ALL_Jvals.index(J)+1)],) #[8], #)
                
                x = [b20_by_K_and_J[K][J][i] for i in range(len(x), len(ener))]
                y = [ener[i] for i in  range(len(y), len(ener))]
                ax2.plot(x, y, 
                         linestyle='None', #__K_LSTYLE[(_ALL_Jvals.index(J)+1)],
                         marker=__J_MARKER[_ALL_Jvals.index(J)],
                         # marker=f"${J}$", markersize=11
                         color=__K_COLORS[K//2], alpha=(1 - (J-K+1)/15))
                x_spl, y_spl = _smooth_curve_splines(x, y)
                ax2.plot(x_spl, y_spl, 
                         color=__K_COLORS[K//2], alpha=(1 - (J-K+1)/15),
                         linestyle=__K_LSTYLE[(_ALL_Jvals.index(J)+1)],) #[8], #)
        
        if len(energy_by_J.get(K, [])) > 1 and plot_PPR_interpolation:
            data_5o2 = energy_by_J[K][J_vals[2]] if K == 1 else None
            _array_b20 = b20_hfb[0] if isinstance(b20_hfb, dict) else b20_hfb
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
                ax.plot(*interp[J], '--', label=f'PPRM-SCL J,K={J}{_frac2} {K}{_frac2}',
                                          color=__K_COLORS[K//2], alpha=1- (J-K)/10)
                ax2.plot(*interp[J], '--', label=f'PPRM-SCL J,K={J}{_frac2} {K}{_frac2}',
                                          color=__K_COLORS[K//2], alpha=1- (J-K)/10)
                
        # ax.plot(b20_hfb [K], E_hfb [K], "o-", label=f'HFB(K={K}/2)')
        ax.plot(b20_vap_K[K], E_vap_K[K],  [".", "P","d",'o','X'][K//2], 
                color=__K_COLORS[K//2], linestyle='None')
        x_spl, y_spl = _smooth_curve_splines(b20_vap_K[K], E_vap_K[K])
        ax.plot(x_spl, y_spl, '-', #label=f'PNP-VAP K={K}{_frac2}', 
                color=__K_COLORS[K//2], alpha=0.98, linewidth=2)
        ax.plot(x_spl[:2], y_spl[:2], marker=[".", "P","d",'o','X'][K//2], 
                     color=__K_COLORS[K//2],  alpha=0.98, linewidth=2, 
                     label=f'PNP-VAP K={K}{_frac2}',)
        
        ## if HFB-k for the curve
        if K in data_hfb:
            E_hfbK = [dat.E_HFB for dat in data_hfb[K]]
            pairK  = [dat.pair  for dat in data_hfb[K]]
            pairKvap = [dat.pair for dat in data_vap_K[K]]
            
            ax.plot(b20_hfb[K], E_hfbK,  [".", "P","d",'o','X'][K//2], 
                    color=__K_COLORS[K//2], linestyle='None')
            x_spl, y_spl = _smooth_curve_splines(b20_hfb[K], E_hfbK)
            ax.plot(x_spl, y_spl, '-', #label=f'HFB K={K}{_frac2}', 
                    color=__K_COLORS[K//2], alpha=0.98, linewidth=2)
            ax.plot(x_spl[:2], y_spl[:2], marker=[".", "P","d",'o','X'][K//2], 
                     color=__K_COLORS[K//2],  alpha=0.98, linewidth=2, 
                     label=f'HFB K={K}{_frac2}',)
            
            # pairing plots
            ax0.plot(b20_vap_K[K], pairKvap,  [".", "P","d",'o','X'][K//2], 
                     color=__K_COLORS[K//2], linestyle='None')
            x_spl, y_spl = _smooth_curve_splines(b20_vap_K[K], pairKvap)
            ax0.plot(x_spl, y_spl, '--', label=f'PNP-VAP K={K}{_frac2}', 
                     color=__K_COLORS[K//2], alpha=0.98, linewidth=2)
            
            ax0.plot(b20_hfb[K], pairK,  [".", "P","d",'o','X'][K//2], 
                     color=__K_COLORS[K//2], linestyle='None')
            x_spl, y_spl = _smooth_curve_splines(b20_hfb[K], pairK)
            y_spl = [min(_y, 0) for _y in y_spl]
            ax0.plot(x_spl, y_spl, '-', #label=f'HFB K={K}{_frac2}', 
                     color=__K_COLORS[K//2], alpha=0.98, linewidth=2)
            ax0.plot(x_spl[:2], y_spl[:2], marker=[".", "P","d",'o','X'][K//2], 
                     color=__K_COLORS[K//2],  alpha=0.98, linewidth=2, 
                     label=f'HFB K={K}{_frac2}',)
        
        ax2.plot(b20_vap_K[K], E_vap_K[K], '.',
                 color=__K_COLORS[K//2], linestyle='None')
        x_spl, y_spl = _smooth_curve_splines(b20_vap_K[K], E_vap_K[K])
        ax2.plot(x_spl, y_spl, '-', label=f'PNP-VAP K={K}{_frac2}',
                 color=__K_COLORS[K//2])
        
        _lims = _lims + b20_vap_K[K]
        
        if len(energy_by_J.get(K, [])) > 1 and plot_PPR_interpolation:
            interp = particlePlusRotor_NCL_spectra(energy_by_J[K], _array_b20, J_vals, K)
        #plt.pause(2)
        
        ## Plot nicely the different K blocked surfaces:
        # ax2.set_ylim([-143, -115]) # 
        # ax2.set_xlim([-0.7,  0.8])
        # x = min(b20_vap_K[K]), max(b20_vap_K[K])
        # _rngs = x[1] - x[0]
        # ax2.set_xlim( [x[0] - .1*_rngs, x[1] + .2*_rngs ])
        # x1 = [(min(energy_by_J[K][J]), max(energy_by_J[K][J])) for J in J_vals]
        # x1, x = zip(*x1)
        # x = min(x1), max(x)
        # _rngs = x[1] - x[0]
        # ax2.set_ylim( [x[0] - .05*_rngs, x[1] + .05*_rngs ] )
        ax2.set_xlabel(r"$\beta_{20}$", fontsize=16)
        ax2.set_ylabel(r"$E\ (MeV)$",   fontsize=14)
        _txt = f"{nucl[1]} PAV energies for $K={K}/2^+$ blocking."
        _txt = f"{nucl[1]} $K={K}/2^+$ blocking"
        ax2.set_title(_txt, fontsize=18)
        ax2.legend()
        # ax2.legend(bbox_to_anchor=(1.1, 1.05))
        plt.tight_layout()
        _tail = f'_k{K}'
        fig2.savefig(FOLDER2SAVE / f"plot_pav_pprm_{nucl[0]}{_tail}{GLOBAL_TAIL_INTER}.pdf")
    
        
    _lims = [min(_lims), max(_lims)]
    _rngs = _lims[1] - _lims[0]
    if _y_lims and max([max(E_vap_K[K]) for K in K_vals]) > _y_lims[1]:
        ax.set_xlim( [_lims[0] - .1*_rngs, _lims[1] + .2*_rngs ] )
        
        _rngs = _y_lims[1] - _y_lims[0]
        ax.set_ylim( [_y_lims[0] - .05*_rngs, _y_lims[1] + .05*_rngs ] )
    
    E_hfb, pair = [dat.E_HFB for dat in data_hfb[0]], [dat.pair for dat in data_hfb[0]]
    
    ax.set_ylim([-140.05, -111.3])
    ax.set_xlim([-0.7, 0.8])
    ax.set_xlabel(r"$\beta_{20}$", fontsize=16)
    ax.set_ylabel(r"$E_{HFB}\ (MeV)$",   fontsize=13)
    ax.plot(b20_hfb[0], E_hfb, 'ko', linestyle='None', markerfacecolor='None', )
    x_spl, y_spl = _smooth_curve_splines(b20_hfb[0], E_hfb)
    ax.plot(x_spl, y_spl, '-', #label='HFB false o-e', 
            color='k', alpha=0.98, linewidth=2)
    ax.plot(x_spl[:2], y_spl[:2], 'ko-', markerfacecolor='None', alpha=0.98, linewidth=2, 
            label='HFB false o-e',)
    _txt = f"{nucl[1]} PNP-VAP TES"
    if PLOT_PPRC and plot_PPR_interpolation: _txt += " and PPRM interpolation"
    # ax.set_title(_txt, fontsize=18)
    ax.annotate(f"{nucl[1]}", (-0.55, -137.5), fontsize=22)
    ax.legend() #bbox_to_anchor=(1.1, 1.05))
    fig.tight_layout()
    _tail = '_allK'
    fig.savefig(FOLDER2SAVE / f"plot_pav_pprm_{nucl[0]}{_tail}{GLOBAL_TAIL_INTER}.pdf")
    
    ax0.set_ylim([-12.1, 0.3])
    ax0.set_xlim([-0.7,  0.8])
    ax0.set_xlabel(r"$\beta_{20}$", fontsize=16)
    ax0.set_ylabel(r"$E_{pair}\ (MeV)$",   fontsize=13)
    ax0.plot(b20_hfb[0], pair, 'ko', linestyle='None', markerfacecolor='None', )
    x_spl, y_spl = _smooth_curve_splines(b20_hfb[0], pair)
    ax0.plot(x_spl, y_spl, '-', #label='HFB false o-e', 
             color='k', alpha=0.98, linewidth=2)
    ax0.plot(x_spl[:2], y_spl[:2], 'ko-', markerfacecolor='None', alpha=0.98, linewidth=2, 
             label='HFB false o-e',)
    ax0.legend() #bbox_to_anchor=(1.1, 1.05))
    fig0.tight_layout()
    fig0.savefig(FOLDER2SAVE / f"plot_pav_pprm_pair_{nucl[0]}{_tail}{GLOBAL_TAIL_INTER}.pdf")
    if not plot_PAV:
        plt.show()
        return
    
    plt.show()
    0/0
    #===========================================================================
    if plot_PPR_interpolation and False:
        fig, ax = plt.subplots(1, 3, figsize=(10, 6))
        # b20_deform = [i-(len(energy_by_J[1])//2) for i in range(len(energy_by_J[1]))]
        # b20_deform = [i for i in range(len(energy_by_J[1]))]
        e_inter_ncl_by_K, _1o2IM_ncl_by_K = {}, {}
        model_ncl_diff_by_K = {}
        for K in K_vals:
            if plot_PAV:
                J_vals = sorted(energy_by_J[K].keys())
                
            _array_b20 = b20_hfb[K] if isinstance(b20_hfb, dict) else b20_hfb
            _array_b20 = b20_pav_K[K]
            interp = particlePlusRotor_NCL_spectra(energy_by_J[K], 
                                                   _array_b20, J_vals, K)
            if not interp: continue
            interp, sp_e_inter, _1o2IM_inter, model_diff = interp
            e_inter_ncl_by_K [K]    = sp_e_inter
            _1o2IM_ncl_by_K  [K]    = _1o2IM_inter
            model_ncl_diff_by_K [K] = model_diff
            
            # for J in J_vals:
            #     if (not PLOT_PPRC) or (Jmax_2_plot and (J > Jmax_2_plot)): 
            #         continue 
            #     ax.plot(*interp[J], '--', label=f'PPRM-NCL J,K={J}{_frac2} {K}{_frac2}',
            #                               color=__K_COLORS[K//2])
        
        if 'FLD_KPAV' in kwargs and -135 in kwargs['FLD_KPAV']:
            data_pavKmix = _get_dataKmix_byDef(kwargs['FLD_KPAV'])
            
            interp = particlePlusRotor_RAL_spectra(data_pavKmix)
    
    #===========================================================================
    
    
    fig, ax = plt.subplots(1, 3 if len(coriolis_coupl_by_K)>0 else 2,
                           figsize=(7, 5))
    any_ = False
    for i, K in enumerate(K_vals):
        _M = '.*d^'
        if not K in energy_by_J: continue
        if len(energy_by_J[K]) > 1 and plot_PPR_interpolation:
            _kwargs = {'marker': _M[i], 'label': f"K={K}{_frac2}", 'markersize':5}
            ax[0].plot(b20_pav_K[K], sp_e_inter_by_K[K], **_kwargs)
            ax[1].plot(b20_pav_K[K], 0.5/_1o2IM_inter_by_K[K], **_kwargs)
            if len(coriolis_coupl_by_K)>0:
                ax[2].plot(b20_pav_K[K], coriolis_coupl_by_K[K], **_kwargs)
            any_ = True
    if any_:
        ax[0].set_title("sp energies (MeV)", fontsize=15)
        ax[1].axhline(0, linestyle='--')
        ax[1].set_title("$\hbar^2/2\mathcal{I}$ Inertia Mom. (MeV)", fontsize=15)
        if len(coriolis_coupl_by_K)>0:
            ax[2].set_title("Decoupling factor", fontsize=15)
        plt.suptitle(f"{nucl[1]} PPRM derived quantities from interpolation", 
                     fontsize=15)
        fig.tight_layout()
        ax[2].legend()
        plt.savefig(FOLDER2SAVE / f"derived_pprm_parameters_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf")
    else:
        plt.show()
    
    if plot_PPR_interpolation:
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
        for K in K_vals:
            i = 0
            if not K in energy_by_J: continue
            for J, model_diff in model_diff_by_K[K].items():
                ax2.semilogy(b20_by_K_and_J[K][J], [abs(x) for x in model_diff], 
                             marker=__J_MARKER[i], color=__K_COLORS[K//2], 
                             alpha= 1 - (J-4-K)/8,
                             label=f'$K={K}{_frac2}^+$ $J={J}{_frac2}^+$')
                i += 1 
        ax2.set_title(f"Correspondence PAV - PPR model SCL ({nucl[1]})", fontsize=18)
        ax2.set_ylabel(r"$\delta(E^{PAV} - E^{SCL})_{K, J}\quad (MeV)$", fontsize=15)
        ax2.set_xlabel(r"$\beta_{20}$", fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FOLDER2SAVE / f"model_validity_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf")
    
    if sum([len(x) for x in data_hwg_K.values()]):
        if 'FLD_KPAV' in kwargs: 
            _simpleGraphPlot(FOLDER2SAVE, kwargs['FLD_KPAV'], nucl)
        
        par_str = '+' if  parity==0 else '-'
        BaseLevelContainer.RELATIVE_PLOT = False #True
        BaseLevelContainer.ONLY_PAIRED_STATES = False
        BaseLevelContainer.MAX_NUM_OF_SIGMAS  = 3
        # BaseLevelContainer.RELATIVE_ENERGY_RANGE = 5.0 # MeV
        
        _graph = BaseLevelContainer()
        _graph.global_title = f"Comparison HWG from different K-blocks, {nucl[1]}"
        EnergyLevelGraph.resetClassAttributes()
        for K in K_vals + [-135, ]:
            if not K in energy_by_J: continue
            ## preparing the HWG level scheme
            if data_hwg_K.get(K, None):
                if Jmax_2_plot:
                    for i in range(len(data_hwg_K[K])-1, -1,-1): 
                        if data_hwg_K[K][i].J > Jmax_2_plot: data_hwg_K[K].pop(i)
                ## 
                level_str = ''.join([x.getSpectrumLines() for x in data_hwg_K[K]])
                title = f'PVC({K}{_frac2})' if K !=-135 else 'GCM  K-mix'
                levels_1  = EnergyLevelGraph(title)
                levels_1.setData(level_str, program='taurus_hwg')
                
                _graph.add_LevelGraph(levels_1)
        plt.tight_layout()
        _graph.plot(FOLDER2SAVE / f'hwg_comparison_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf',
                    figaspect=(6*(2/3)*len(K_vals), 5))
    ####
    plt.show()

def _plot_occupationNumbers_vap_by_b20(occnumb_vap_K, FOLDER2SAVE=None, nucl='***'):
    """
    print occupation of the levels from each K and the deformations.
    """
    all_sh_states = {}
    UNPROJ = 0          # 0 for unprojected, 1 for projected PN 
    JMAX = 5
    for K, vals_K in occnumb_vap_K.items():
        b20_sorted = dict([(float(b), b) for b in vals_K.keys()])
        x, yobj = [], []
        states = {-1: {}, 1: {}}
        for b in sorted(list(b20_sorted.keys())):
            x.append(b)
            obj_ : OccupationNumberData =  vals_K[b20_sorted[b]]
            yobj.append(obj_)
            if UNPROJ == 1: 
                assert obj_.hasProjectedOccupations, "Non projected VAP results"
            
            for sh_, qqnn in obj_.get_numbers.items():
                if not sh_ in all_sh_states: all_sh_states[sh_] = qqnn
                if not sh_ in states[-1]:
                    states[-1][sh_], states[ 1][sh_] = [], []
                for t in (-1, 1):
                    val = obj_.get_occupations[UNPROJ][t][sh_]
                    states[t][sh_].append(val)
        _ = 0
        
        fig, ax = plt.subplots(2, 1, figsize=(5, 4.5))
        for t in (-1, 1):
            i_ax = (1 + t) // 2
            
            js = []
            kwargs = {'color':'black', 'linestyle':'--', 'linewidth': 1}
            for sh_ in all_sh_states:
                j = all_sh_states[sh_][2]
                if j in js or j >= JMAX: continue
                else: js.append(j)
                ax[i_ax].axhline(j + 1, **kwargs)
            ax[i_ax].axhline(0, **kwargs)
            
            i=0
            _MARKERS = 'oPX*ds^v><_+xP1234'
            _COLORS_ = ['black', 'blue', 'red', 'green', 'magenta', 'cyan', 'orange']
            for sh_, vals in states[t].items():
                n,l,j = all_sh_states[sh_]
                N = 2*n + l
                _L = 'spdfghij'
                sh_ = f"${n}{_L[l]}_{{{j}/2}}$"
                label_ = None if t == +1 else f'{sh_}'
                ax[i_ax].plot(x, vals, marker=_MARKERS[i], linestyle='--',
                              markersize=5,
                              color=_COLORS_[N], label=label_)
                i += 1
                
        ax[0].set_ylabel('Occ. protons')
        ax[1].set_ylabel('Occ. neutrons')
        ax[1].set_xlabel('$\\beta_{20}$')
        
        fig.suptitle(f'MF SHO-orbital occupations evolution ({nucl[1]}). Blocked to K={K}/2 ')
        # Create a single legend for both subplots, placed to the right
        # handles, labels = ax[0].get_legend_handles_labels()
        # ax[0].legend(handles, labels, loc='center left', bbox_to_anchor=(0.0, 0.0))
        fig.legend(loc='center right', framealpha=1.0)
        plt.subplots_adjust(left=0.1,  right=0.85)
            
        fig.savefig(FOLDER2SAVE / f'occupanciesVAP-k{K}_{nucl[0]}.pdf')

    print("[Done] Plotting the occupancies VAP.")
        
        
        

def __tryFitPPR_SCL_ForFirsHWGStates(graph, k):
    """Auxiliary function to verfy if the HWG get the results from the SCL """
    if k == -135: return
    data_j, b20_deform, J2_vals, data_5o2 = {}, [-.1, 0, .1, .2], [], None
    for i, jp in enumerate(graph._JP[-1]):
        j = int(jp.split('/')[0])
        if not j in J2_vals:
            data_j[j] = [graph._E[-1][i]+0.0001*ii for ii in range(4) ]
            J2_vals.append(j)
            if j == 1: data_5o2 = data_j[j]
    
    J2_vals.sort()
    # data_interpolated, A, B, C, model_diff 
    if len(J2_vals) <= 2 + int(k==1):
        print(" ERROR_fitPPRSCL try:, not enough states for Jvals from _EnergyObjct. SKIP")
        return
    arg = particlePlusRotor_SCL_spectra(data_j, b20_deform, J2_vals, k, data_5o2)
    print(" ** Testing PPRM-SCL with HWG: K =", k, "/2")
    for j, err in arg[4].items() : print(f"   j:{j}/2  diff= {err[0]:5.4f} MeV")
    print()

def _simpleGraphPlot(FOLDER2SAVE, FLD_KPAV, nucleus=('', '')):
    """
    Include the simple plot without the complex spectra plotter/EnergyLevelGraph
    """
    _EnergyLevelSimpleGraph.RELATIVE_PLOT      = False
    _EnergyLevelSimpleGraph.MAX_NUM_OF_SIGMAS  = 2
    _EnergyLevelSimpleGraph.MAX_ENERGY_DISPLAY = 5.5
    _graph = _EnergyLevelSimpleGraph("") #"HWG spectra from each K blocking")
    
    sorting_order = [k for k in range(20)] + [-135]    
    for k in sorting_order:
        fld = FLD_KPAV.get(k, None)
        if not fld: 
            continue
        if (fld.parts[-1] != 'HWG'): fld /= 'HWG'
        example_levels = getAllLevelsAsString(fld)
        
        if not example_levels:
            print(f" [WARNING - Skip] Not found hwg files in [{fld}]")
            continue
        
        if k == -135: _Kstr = 'mix'
        elif k % 2:   _Kstr = f'K={k}/2'
        else:         _Kstr = f'K={k}'    
        _graph.setData(example_levels, f'{nucleus[1]} {_Kstr}')
        
        __tryFitPPR_SCL_ForFirsHWGStates(_graph, k)
    global GLOBAL_TAIL_INTER
    _graph.plot(fld2saveWithFilename=FOLDER2SAVE / f'hwg-spectra_{nucleus[0]}{GLOBAL_TAIL_INTER}.pdf')

def __getKcontributionsForState(labels_by_K, labjp, g2_jp):
    
    valid_K = [k for k in labels_by_K.keys()]
    kp_contributions = dict([(k, 0.0) for k in valid_K])
    kp_found_K = {1: [], 3: [], 5: [],}
    labels_by_K2 = {}
    for k in valid_K:
        labels_by_K2[k] = [str(l) for l in labels_by_K[k]]
    for i, lab in enumerate(labjp):
        g2 = g2_jp[i]
        for k in valid_K:
            if lab in labels_by_K[k]: 
                kp_contributions[k] += g2
                kp_found_K[k].append(str(lab))
                notFound = False
                break
            notFound = True
        if notFound: print(f"  Label not found! lab[{i}] = [{lab}]")
    prettyPrintDictionary(labels_by_K2)
    prettyPrintDictionary(kp_found_K)
    return kp_contributions

_all_labels = {}
def _recoverLabels(dat_obj, coll_hwg_K, K=None, b20=None):
    ## Old calculations have not set label-attribute, recover from the E_HFB and 
    ## the labels stored in the collective functions
    global _all_labels
    if len(_all_labels) == 0:
        for k,  dictK in coll_hwg_K.items():
            _all_labels[k] = []
            _labels_set = []
            for _jp, cllobj in dictK.items():
                for s, labels in cllobj.labels.items():
                    _labels_set = _labels_set + labels
            _all_labels[k] = list(set(_labels_set))
            _all_labels[k] = [str(l) for l in _all_labels[k]]
    assert dat_obj.label_state == None, "Method can only be used if label_state = None"
    
    e_str = f"{abs(dat_obj.E_HFB):9.5f}".replace('.', '')
    if K in (None, -135):
        for k, set_ in _all_labels.items():
            for e2 in set_:
                if e2[:5] == e_str[:5]: ## 5 include already the eV rounding
                    _ = 0
                    dat_obj.label_state = int(e2)
                    return dat_obj
    else:
        for e2 in _all_labels[K]:
            if e2[:5] == e_str[:5]:     ## 5 include already the eV rounding
                _ = 0
                dat_obj.label_state = int(e2)
                return dat_obj
    
    print(f"Not found label K={K}: E={dat_obj.E_HFB:7.6f}!  b20=[{b20}]")
    return dat_obj
        
def _plot_images_collective_wf_hwg(b20_hfb, b20_vap_K, 
                                   data_vap_K, data_hwg_K, 
                                   coll_hwg_K, occn_hwg_K,
                                   FOLDER2SAVE=None, 
                                   Jmax_2_plot=100, nucl='****', MAX_SIGMA=5):
    PLOT_TABLE_AS_LATEX = True
    _hIntKJ = all([k%2 == 1 for k in data_hwg_K.keys()])
    _frac2 = '/2' if _hIntKJ else ''
    print("\n\n\n")
    Kb20_by_label, labels_by_K = {}, {}
    
    ## Create a dictionary label-deform-K
    for K, list_ in data_vap_K.items():
        labels_by_K[K] = []
        for i, dat in enumerate(list_):
            b20 = b20_vap_K[K][i]
            
            if not dat.label_state:
                dat = _recoverLabels(dat, coll_hwg_K, K=K, b20=b20)
            lab = dat.label_state
            
            if lab in Kb20_by_label:
                print(f"  Error, label {lab} K[{K}] b20[{b20}] already found "
                      f" [K:{K},b20:{b20}]")
            else:
                Kb20_by_label[lab] = K, b20
                labels_by_K[K].append(lab)
                
    
    xy_sigmas_byJP, occ_xy_sig_byJP, Kmix_by_JP = {}, {}, {-135: {}}
    error_print = []
    for K, jp_vals in coll_hwg_K.items():
        xy_sigmas_byJP [K] = {}
        occ_xy_sig_byJP[K] = {}
        for jp, obj_ in jp_vals.items():
            if jp[0] > Jmax_2_plot: continue
            if jp not in xy_sigmas_byJP:
                xy_sigmas_byJP [K][jp] = []
                occ_xy_sig_byJP[K][jp] = []
                if K == -135: Kmix_by_JP[K][jp] = []
            
            print()
            for sig in obj_.sigmas:
                # xy_sigmas_byJP[jp].append([])
                g2_jp = obj_.g2values[sig]
                labjp = obj_.labels  [sig]
                occjp = occn_hwg_K[K][jp]
                orbitals = [f"{x:<5}" for x in occjp.labels]
                if sig == 1:
                    print("   K={:>3} ".format(K if K != -135 else "mix"))
                    print("J  \sigma  {x} (p)   {x} (n)".format(x="  ".join(orbitals)))
                
                occsp = [100*x for x in occjp.relative_occ_protons [sig]]
                occsn = [100*x for x in occjp.relative_occ_neutrons[sig]]
                occ_xy_sig_byJP[K][jp].append([deepcopy(occsp), deepcopy(occsn)])
                if sig <= MAX_SIGMA:
                    occsp = [f"{x:5.2f}" for x in occsp]
                    occsn = [f"{x:5.2f}" for x in occsn]
                    print("{:>2}/2 {:>2}    {} (p)   {} (n)".format(jp[0], sig,
                                                                    "  ".join(occsp), 
                                                                    "  ".join(occsn)))
                else: continue
                
                if K == -135:
                    aux = __getKcontributionsForState(labels_by_K, labjp, g2_jp)
                    Kmix_by_JP[K][jp].append(deepcopy(aux))
                
                b20_x, labs2rm = [], []
                for i, lab in enumerate(labjp):
                    if lab not in Kb20_by_label:
                        error_print.append(f"  [Error] {lab} not saved jp,sig={jp},{sig}, prev b20-K={b20},{K}")
                        labs2rm.append(i)
                        continue
                    if K != -135: b20_x.append(Kb20_by_label[lab][1])
                    else:         b20_x.append(Kb20_by_label[lab])
                    
                    K1, b20 = Kb20_by_label[lab]
                    if (K1 != K and K != -135):
                        print(f" >> What, K does not match: {K1}, {K} in s={sig},{jp},  {lab}")
                
                if labs2rm:
                    for i in reversed(labs2rm): g2_jp.pop(i)
                xy_sigmas_byJP [K][jp].append([deepcopy(b20_x), deepcopy(g2_jp)])
        _=0
        
    print("\n".join(error_print))
    
    individual_figs = []
    orbitals   = [str(i) for i in orbitals]
    x_orbitals = [i for i in range(len(orbitals))]
    tickargs   = {'fontsize': 'x-small', 'rotation': 0}
    for K in xy_sigmas_byJP:
        sorted_jp = list(xy_sigmas_byJP[K].keys())
        sorted_jp.sort(key=lambda x: x[0])
        for jp in sorted_jp:
            par_str = '+' if jp[1]==0 else '-'
            _jstr = jp[0] if _hIntKJ else jp[0] // 2
            title_ = f" Coll-wf and orbital occupation $J^\pi={_jstr}{_frac2} ^{{{par_str}}}$ for each $\sigma<${MAX_SIGMA}"
            
            fig, axs = plt.subplots(MAX_SIGMA, 2, figsize=(9, int(2*MAX_SIGMA)), 
                                    gridspec_kw={'width_ratios': [1, 2]})
            if (K != -135):
                fig.suptitle(title_ +f'\n K={K}{_frac2}')
                
                for sig, dat in enumerate(xy_sigmas_byJP[K][jp]):
                    if sig==MAX_SIGMA: break
                    axs[sig,0].plot(*dat, '.-')
                    axs[sig,0].set_ylabel(f"sig={sig+1}")
                    axs[sig,1].set_ylabel("%")
                    axs[sig,1].plot(occ_xy_sig_byJP[K][jp][sig][0], 'r.-', label='p')
                    axs[sig,1].plot(occ_xy_sig_byJP[K][jp][sig][1], 'b.-', label='n')
                    axs[sig,1].set_xticks(x_orbitals, orbitals, **tickargs)
            else:
                fig.suptitle(title_ +'\n K-Mixing')
            
                for sig, dat in enumerate(xy_sigmas_byJP[K][jp]):
                    if (sig == MAX_SIGMA): break
                    for K1 in sorted(list(data_vap_K.keys())):                        
                        dat2 = []
                        for i, kb2 in enumerate(dat[0]):
                            if kb2[0] != K1: continue
                            dat2.append((kb2[1], dat[1][i]))
                        dat2   = list(zip(*dat2))
                        prc_k1 = f": {Kmix_by_JP[K][jp][sig][K1]:3.2f}"
                        axs[sig,0].plot(*dat2, '.-', label=f'K={K1}{_frac2} {prc_k1}')
                        axs[sig,0].set_ylabel(f"sig={sig+1}")
                        axs[sig,1].set_ylabel("%")
                        axs[sig,1].plot(occ_xy_sig_byJP[K][jp][sig][0], 'r.-', 
                                        label='p' if sig==0 else None)
                        axs[sig,1].plot(occ_xy_sig_byJP[K][jp][sig][1], 'b.-',
                                        label='n' if sig==0 else None)
                        axs[sig,1].set_xticks(x_orbitals, orbitals, **tickargs)
                        
                        axs[sig,0].legend()
            axs[-1, 0].set_xlabel(r"$\beta_{{20}}$")
            axs[ 0, 1].legend()
            # axs[0,1].legend()
            fig.tight_layout()
            if not os.path.exists(FOLDER2SAVE / 'temp_'): os.mkdir(FOLDER2SAVE / 'temp_')
            individual_figs.append(FOLDER2SAVE / 'temp_/k{}_j{}p{}_coll_wf.pdf'.format(K, *jp))
            fig.savefig(individual_figs[-1])
    
    global GLOBAL_TAIL_INTER
    ## combine pdf
    from PyPDF2 import PdfMerger
    merger = PdfMerger()
    for pdf in individual_figs:
        merger.append(pdf)
    merger.write(FOLDER2SAVE / f"collective_wf_byK_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf")
    merger.close()
    # for pdf in individual_figs:
    #     os.remove(pdf)
    
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
            if os.path.exists(_aux_path_multi) and not -135 in data_hwg:
                _aux_path_multi = _aux_path_multi / 'HWG'
                list_dat = filter(lambda x: x.endswith('.dat'), os.listdir(_aux_path_multi))
                list_dat = list(list_dat)
                data_hwg[-135] = [DataTaurusMIX(_aux_path_multi / f) for f in list_dat]
        
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
        plt.legend(loc='upper right')
        if PLOT_PPRC and plot_SCL_interpolation: _txt += " and PPRM interpolation"
        _lims = [min(_lims), max(_lims)]
        _rngs = _lims[1] - _lims[0]
        ax.set_xlim( [_lims[0] * 1.1, _lims[1] * 1.5 ] )
        # plt.ylim([-130, -90])
        # plt.xlim([-0.7, 1.2])
        plt.title(f"{nucl[1]} PAV projections for all K and PPRM interpolation")
        # plt.legend(bbox_to_anchor=(1.1, 1.05))
        
        tail = '' if len(K_vals)>1 else f'_k{K_vals[0]}'
        fig.tight_layout()
        plt.savefig(FOLDER2SAVE / f"plot_pav_pprm_{nucl[0]}{tail}.pdf")
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
            for K in K_vals + [-135, ]:
                ## preparing the HWG level scheme
                if data_hwg.get(K, None):
                    level_str = ''.join([x.getSpectrumLines() for x in data_hwg[K]])
                    title = f'K={K}/2 {par_str}' if K !=-135 else 'K-mix'
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
        

def _plotPAVresultsSameFolder_mulipleK(folders_2_import, MAIN_FLD, K_vals, parity=0,
                                       plot_PPR_interpolation=True, plot_PAV=True,
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
        b20_hfb, b20_vap_K, b20_pav_K = {}, {}, {}
        data_hfb, data_vap_K, data_pav_K, data_hwg_K = {}, {}, {}, {}
        coll_hwg_K, occn_hwg_K = {}, {}
        E_vap_K, occnumb_vap_K, energy_by_J = {}, {}, {}
        
        z, n = folder_args[0]
        _core_0 = 0
        z_real, n_real = z + _core_0, n + _core_0
        nucl = (f"{z_real+n_real}{elementNameByZ[z_real]}", 
                rf"$^{{{z_real+n_real}}}${{{elementNameByZ[z_real]}}}")
        
        if (z%2, n%2) == (0, 0): 
            K_vals, plot_PPR_interpolation = [0, ], False
        print(f"[{z}, {n}]   ******** ")
        FLD  = folder_args[1].format(MAIN_FLD=MAIN_FLD, z=z, n=n)
        FLD  = Path(FLD)
        
        FLD_KHFB = [(k, FLD / f"{k}_{parity}_HFB") for k in K_vals]
        FLD_KVAP = [(k, FLD / f"{k}_{parity}_VAP") for k in K_vals]
        FLD_KPAV = [(k, FLD / f"{k}_{parity}_PNPAMP_HWG") for k in K_vals]
        FLD_KHFB = dict(FLD_KHFB)
        FLD_KVAP = dict(FLD_KVAP)
        FLD_KPAV = dict(FLD_KPAV)
        
        if not FLD.exists():
            print(f" [ERROR Imprt] Folder [{FLD}] not found, skipping!")
            continue
        export_fn = folder_args[2].format(z, n)
        
        ## Read VAP mean field file
        # with open(export_fn, 'r') as f:
        with open(FLD / export_fn, "r") as f:
            b20_hfb[0], data_hfb[0] = [], []
            for line in f.readlines()[1:]:
                head_, line = line.split(OUTPUT_HEADER_SEPARATOR)
                i, b20 = head_.strip().split(':')
                index_b20[i] = float(b20)
                b20_hfb[0].append(float(b20))
                
                obj_ = DataTaurus(z, n, None, empty_data=True)
                obj_.setDataFromCSVLine(line)
                data_hfb[0].append(deepcopy(obj_))
        
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
                    i, b20 = head_.strip().split(':')
                    if i not in index_b20: 
                        print(" [Warning] deformation index", i, "not from VAP.")
                    b20_vap_K[K].append(float(b20))
                    
                    obj_ = DataTaurus(z, n, None, empty_data=True)
                    obj_.setDataFromCSVLine(line)
                    data_vap_K[K].append(deepcopy(obj_))
                    E_vap_K[K].append(obj_.E_HFB)
            ## if there are HFB-K surfaces import them  ----- complementary ---
            if os.path.exists(FLD_KHFB[K] / exp_fn_k):
                with open(FLD_KHFB[K] / exp_fn_k, "r") as f:
                    b20_hfb[K], data_hfb[K] = [], []
                    for line in f.readlines()[1:]:
                        head_, line = line.split(OUTPUT_HEADER_SEPARATOR)
                        i, b20 = head_.strip().split(':')
                        if i not in index_b20: 
                            print(" [Warning] deformation index", i, "not from VAP.")
                        b20_hfb[K].append(float(b20))
                        
                        obj_ = DataTaurus(z, n, None, empty_data=True)
                        obj_.setDataFromCSVLine(line)
                        data_hfb[K].append(deepcopy(obj_))
            
            ## Store occupation numbers if exists
            occ_numb = os.listdir(FLD_KVAP[K])
            occ_numb = list(filter(lambda x: x.endswith('_occupation_numbers.dat'), occ_numb))
            if any(occ_numb):
                occnumb_vap_K[K] = {}
                for fn_ in occ_numb:
                    b20 = fn_.split('_occupation_numbers.dat')[0].replace('_', '-')
                    obj_ = OccupationNumberData(FLD_KVAP[K] / fn_)
                    occnumb_vap_K[K][b20] = obj_
            
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
                    
                    list_dat = list(filter(lambda x: x.startswith('collective_wavefunction_'), 
                                                os.listdir(_pth_hwg)) )
                    coll_hwg_K[K], occn_hwg_K[K] = {}, {}
                    for f in list_dat:
                        ff = f[:-4].replace('collective_wavefunction_', '')
                        p = 0 if 'p' == ff[-1] else 1
                        J = int(ff[:-1])
                        coll_hwg_K[K][(J,p)] = CollectiveWFData  (_pth_hwg / f)
                        ff = f.replace('collective_wavefunction_', 'occupation_numbers_')
                        occn_hwg_K[K][(J,p)] = OccupationsHWGData(_pth_hwg / ff)
                
                ## Set the J levels and its deformations for the PAV
                energy_by_J[K] = {}
                b20_by_K_and_J[K] = {}
                for id_, dat in enumerate(data_pav_K[K]):
                    kval = [(i, k) for i, k in enumerate(dat.KJ)]
                    kval = list(filter(lambda x: x[1]==K, enumerate(dat.KJ)))
                    
                    ener, jval, b20J = [], [], []
                    for i, _ in kval:
                        if abs(dat.proj_norm[i]) < 1.0e-6:
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
                list_dat = list(filter(lambda x: x.endswith('.dat'), os.listdir(_aux_path_multi)))
                data_hwg_K[-135] = [DataTaurusMIX(_aux_path_multi / f) for f in list_dat]
                list_dat = list(filter(lambda x: x.startswith('collective_wavefunction_'), os.listdir(_aux_path_multi)))
                coll_hwg_K[-135], occn_hwg_K[-135] = {}, {}
                FLD_KPAV  [-135] = _aux_path_multi
                for f in list_dat:
                    ff = f[:-4].replace('collective_wavefunction_', '')
                    p = 0 if 'p' == ff[-1] else 1
                    J = int(ff[:-1])
                    coll_hwg_K[-135][(J,p)] = CollectiveWFData  (_aux_path_multi / f)
                    ff = f.replace('collective_wavefunction_', 'occupation_numbers_')
                    occn_hwg_K[-135][(J,p)] = OccupationsHWGData(_aux_path_multi / ff)
            
            # K loop end
        if occnumb_vap_K:
            _plot_occupationNumbers_vap_by_b20(occnumb_vap_K, 
                                                FOLDER2SAVE=FOLDER2SAVE, nucl=nucl)
        
        _generate_images_hfb_vapK_pav_hwg(
            b20_hfb, data_hfb, b20_vap_K, E_vap_K, data_vap_K,
            b20_pav_K, b20_by_K_and_J, energy_by_J, Jmax_2_plot, 
            data_hwg_K,
            plot_PAV=True, plot_PPR_interpolation=plot_PPR_interpolation, 
            FOLDER2SAVE=FOLDER2SAVE, parity=parity, nucl=nucl, 
            ## kwargs
            FLD_KPAV=FLD_KPAV
        )
        if coll_hwg_K and occn_hwg_K:
            _plot_images_collective_wf_hwg(
                b20_hfb, b20_vap_K, data_vap_K, data_hwg_K, coll_hwg_K, occn_hwg_K, 
                FOLDER2SAVE=FOLDER2SAVE, Jmax_2_plot=Jmax_2_plot,
                nucl=nucl, MAX_SIGMA=5,
            )

def get_sp_states_from_interaction(FLD, INTERACTION_TITLE):
    """
    Read the hamiltonian and store the sp-states basis with the QN jj object
    """
    sp_states = {}
    with open(FLD / f'{INTERACTION_TITLE}.sho', 'r') as f:
        sts = f.readlines()[2].strip().split()[1:]
    i = 0
    for st in sts:
        n, l, j = readAntoine(st, l_ge_10=True)
        for m in range(j, -j-1, -2): 
            i += 1
            sp_states[i] = QN_1body_jj(n, l, j, m=m)
    return sp_states

def _plotKblockedSurfacesWithMultiplets(folders_2_import, MAIN_FLD_TEMP, K_vals, 
                                        parity=0, plot_PAV=False, max_2_plot=None,):
    """
    :folders_2_import <nucleus>  ((z,n), Folder_BU, export_filename.txt) - templates
    
    This script is meant for multi-K calculations, that employ all the K blocked
    surfaces from the same VAP-false odd-even calculation. The deformations are 
    independent.
    
    Main purpose: plot-scatter the sets of solutions that get to different outcomes
        for the same b20-K. With the possibility of plotting the diagonal-PAV
        components-norm.
    Requiement for PAV:   PNPAMP/
                            gcm: "_b20str_K_P.bin  k"
                              k/ 
                                OUT puts
    """
    FOLDER2SAVE= Path(MAIN_FLD_TEMP)#.parent
    global GLOBAL_TAIL_INTER, INTERACTION_TITLE
    _parity_str = '+' if parity == 0 else '-'
    for folder_args in folders_2_import:
        index_b20 = {}
        b20_hfb, b20_vap_K, b20_pav_K = [], {}, {}
        data_hfb, data_vap_K, data_pav_K = [], {}, {}
        data_norm_K = {}
        E_hfb, E_vap_K = [], {}
        
        z, n = folder_args[0]
        _core_0 = 0
        z_real, n_real = z + _core_0, n + _core_0
        nucl = (f"{z_real+n_real}{elementNameByZ[z_real]}", 
                rf"$^{{{z_real+n_real}}}${{{elementNameByZ[z_real]}}}")
        
        print(f"[{z}, {n}]   ******** ")
        FLD  = folder_args[1].format(MAIN_FLD=MAIN_FLD_TEMP, z=z, n=n)
        FLD  = Path(FLD)
        
        FLD_KVAP = [(k, FLD / f"{k}_{parity}_VAP") for k in K_vals]
        FLD_KPAV = FLD / 'PNAMP'
        FLD_KVAP = dict(FLD_KVAP)
        #FLD_KPAV = dict(FLD_KPAV)
        
        if not FLD.exists():
            print(f" [ERROR Imprt] Folder [{FLD}] not found, skipping!")
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
        
        ## Read VAP-K blocked state - along wiht the PAV folder(diagonal) if exists.
        index_K_elements_log = {}
        for K in K_vals:
            print(" * Importing K =", K)
            b20_vap_K[K], data_vap_K[K] = [], []
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
            ## read if there is the folder norm_continuity
            _norm_cont = FLD_KVAP[K] / f"norm_continuity/norm_overlaps.txt"
            if os.path.exists(_norm_cont):
                with open(_norm_cont, 'r') as f:
                    data_norm_K[K] = {}
                    for line in f.readlines():
                        b20, vals = line.split(OUTPUT_HEADER_SEPARATOR)
                        indx, b20 = b20.split(':')
                        indx, b20 = int(indx), float(b20)
                        data_norm_K[K][indx] = []
                        for val in vals.split(','):
                            q, e, n = val.split(':')
                            q, e, n = int(q), float(e), float(n)
                            data_norm_K[K][indx].append( [b20, n, e, q] )
                ## import the qp states 
                sp_states = get_sp_states_from_interaction(FLD, INTERACTION_TITLE)
            
        if FLD_KPAV.exists():
            for K in K_vals:
                index_K_elements_log[K], b20_pav_K[K], data_pav_K[K] = [], [], []
            
            if 'gcm' in os.listdir(FLD_KPAV):
                with open(FLD_KPAV / 'gcm', "r") as f:
                    for line in f.readlines():
                        wf1, wf2, i, i2 = line.strip().split()
                        _ERR = f"This script only works for diagonal gcm elements. Check {FLD_KPAV}/gcm"
                        assert   i ==  i2, _ERR
                        assert wf1 == wf2, _ERR
                        args = wf1.replace('.bin', '').split('_')
                        b20    = float(args[-3])
                        K, par = int(args[-2]), int(args[-1])
                        if len(args) == 4 and args[0] == '': b20 *= -1
                        
                        b20_pav_K[K].append(b20)
                        obj_ = DataTaurusPAV(z, n, FLD_KPAV / f'{i}/OUT')
                        index_K_elements_log[K].append(f"K:{K}, b20:{b20:4.3f}, par:{par}  {i}/OUT")
                        data_pav_K[K].append(obj_)
            else:
                print(" ERROR: TODO: define a method to get the diagonal elements without gcm file.")
        
        
        _ = 0
        b20_pav_K_cp  = deepcopy( b20_pav_K)
        data_vap_K_cp = deepcopy(data_vap_K)
        ## Find duplicates for the states in folders
        DUP_FLDS = list(filter(lambda x: x.startswith('BU_states_d'), os.listdir(FLD)))
        for fld_ in DUP_FLDS:
            print("   duplicate in:", fld_)
            _, K = fld_.replace('BU_states_d', '').split('K')
            for fn in os.listdir(FLD / fld_):
                if not fn.endswith('OUT'): continue
                if int(K) not in K_vals:   continue
                
                obj_ = DataTaurus(z, n, FLD / (fld_+'/'+fn))
                data_vap_K[int(K)].append(obj_)
                
        ## Figures,
        ## Intrinsic states for all K and diagonal PAV
        fig, ax =  plt.subplots(2, 1, figsize=(5, 6),
                                gridspec_kw={'height_ratios': [1.5, 1]})
        _markers_, _colors_ = '.osh^v<>+d', ['red', 'blue', 'green', 'magenta', 'orange', 'olive']*2
        for i in range(2):
            for K, dat_list in data_vap_K.items():
                if i == 1 and K > 5: continue
                x = [obj_.b20_isoscalar for obj_ in dat_list]
                y = [obj_.E_HFB         for obj_ in dat_list]
                ax[i].scatter(x, y, label=f'K={K}/2',
                           marker=_markers_[K//2], 
                           facecolor='None',  #_markers_[K_vals.index(K)], 
                           color=_colors_[K//2],)
            #add rectangle
            if i == 0:
                from matplotlib.patches import Rectangle
                ax[i].add_patch(Rectangle((0,-163),0.75,15,
                                linestyle='--',
                                edgecolor='black',
                                facecolor='none',
                                lw=1))
            ax[i].tick_params(direction='in')
            if i == 0:
                ax[i].set_xlim( (-0.7, 0.8) )
                ax[i].set_ylim( (-170, -115) )
            else:
                ax[i].set_xlim( (0.0, 0.75) )
                ax[i].set_ylim( (-163, -148) )
            if i == 0:
                ax[i].set_title(f"Results for all $K^{_parity_str}$ - {nucl[1]}")
            ax[i].plot(b20_hfb, E_hfb, '.-k', label='false OE')
            ax[i].set_xlabel(r'$\beta_{20}$')
            ax[i].set_ylabel(r'$E_{HFB}$')
            ax[i].legend()
        plt.tight_layout()
        plt.savefig(FOLDER2SAVE / f"scatter_blocking_Kvap_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf")
        plt.show()
        
        ## Intrinsic overlap by K and tuplet properties, 
        #   data_norm_K[K][indx].append( [b20, n, e, q] )
        lims_K = {}  ## NOTE: Fix the limits for scatter K-norm
        for K, dat in data_vap_K.items():
            dat = list(map(lambda x: x.E_HFB, dat))
            e0, e1 = min(dat), max(dat)
            yr = e1 - e0
            lims_K[K] = e0 - 0.02*yr, e0 + 0.20*yr
        
        # lims_K = {1: (-143, -126), 3: (-143, -126), 
                  # 5: (-143, -126), 7: (-143, -126), 9:(-143, -126)}
        _markers_ = '..^oPv<>+d'
        _colors_.insert(0, 'black')
        _L_ = 'spdfghijklmn'
        for K, dat_list in data_norm_K.items():
            fig1, ax = plt.subplots(2, 1, figsize=(5, 5),
                                   gridspec_kw={'height_ratios': [1.2, 1]})
            ## process the data:
            sort_ = sorted(list(dat_list))
            qp_list   = []
            b20_by_qp = {}
            ene_by_qp = {}
            nor_by_qp = {}
            for idx in sort_:
                vals = dat_list[idx]
                for val in vals:
                    b20, n, e, qp = val
                    n = abs(n)
                    if qp not in qp_list: 
                        qp_list.append(qp)
                        b20_by_qp[qp] = [b20, ]
                        ene_by_qp[qp] = [e, ]
                        nor_by_qp[qp] = [n, ]
                    else:
                        b20_by_qp[qp].append(b20)
                        ene_by_qp[qp].append(e)
                        nor_by_qp[qp].append(n)
            ##
            
            for i, qp in enumerate(sorted(qp_list)):
                s = sp_states.get(qp, None)
                kwargs = {'label': 'all' if qp==0 else r"${}{}_{{{}/2}}$".format(s.n,_L_[s.l],s.j),
                          'marker': _markers_[i], 
                          'facecolor': 'None' if i!=0 else 'k',  
                          'color': _colors_[i],}
                ax[0].scatter (b20_by_qp[qp], ene_by_qp[qp], **kwargs)
                ax[1].scatter (b20_by_qp[qp], nor_by_qp[qp], **kwargs)
                
            # fig1.suptitle(f'Norm overlap on K=${K}/2^+$ tuplets by blocking.')
            ax[1].set_xlabel(r'$\beta_{20}$')
            ax[0].set_ylabel(r'$E_{HFB}$')
            ax[1].set_ylabel(r'Norm$({\beta})$')
            ax[0].legend()
            
            ax[0].set_ylim( lims_K[K] )
            # ax[1].set_ylim( (0, 1.1) )
            plt.tight_layout()
            plt.savefig(FOLDER2SAVE / f"hfb_norm_K{K}_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf")
        plt.show()
        _colors_.pop(0)
        
        if not FLD_KPAV.exists(): return 
        for K, dat_list in data_pav_K.items():
            fig, ax =  plt.subplots(1, 2, figsize=(9, 4))
            
            y = [obj_.E_HFB for obj_ in data_vap_K_cp[K]]
            ax[0].scatter(b20_vap_K[K], y, label=f'K={K}/2',
                          marker='.', linestyle='-', facecolor='None',   
                          color=_colors_[K_vals.index(K)],)
            elem : DataTaurusPAV = None
            x = b20_pav_K_cp[K]
            J_vals = [j for j in range(max_2_plot)]
            surf_by_J = dict([(j, []) for j in J_vals])
            for elem in dat_list:
                for i, J in enumerate(elem.J):
                    if J in surf_by_J: 
                        surf_by_J[J].append( (elem.proj_norm[i], elem.proj_energy[i]))
            _ylim = (999999, -999999)
            for J in J_vals:
                if surf_by_J[J] == []: continue
                n, e = zip(*surf_by_J[J])
                if len(e) != len(x): continue
                
                # split left and right side b20
                x1 = list(filter(lambda x: x < 0, x))
                x2 = [x[i] for i in range(len(x1), len(x))]
                e1 = [e[i] for i in range(len(x1))]
                e2 = [e[i] for i in range(len(x1), len(x))]
                
                ax[0].plot(x1, e1, linestyle='--', linewidth=1, markersize=3, 
                           marker=_markers_[(J)//2], color=_colors_[(J)//2])
                ax[0].plot(x2, e2, label=f'j={J}/2', linestyle='--', linewidth=1, markersize=3, 
                           marker=_markers_[(J)//2], color=_colors_[(J)//2])
                ax[1].plot(x, n, label=f'j={J}/2', linestyle='--', linewidth=1, markersize=3, 
                           marker=_markers_[(J)//2], color=_colors_[(J)//2])
                _ylim = (min(_ylim[0], min(e)), max(_ylim[1], max(e)))
            
            if K != 9: ax[0].set_ylim( (_ylim[0]*1.01, _ylim[1]*1.25) )
            ax[0].set_title(f"Energies")
            ax[0].set_xlabel(r'$\beta_{20}$')
            ax[0].set_ylabel(r'$E_{HFB}-PNPAMP$')
            ax[0].legend()
            
            ax[1].set_title(f"Norm-overlap by J")
            ax[1].set_xlabel(r'$\beta_{20}$')
            ax[1].set_ylabel(r'$\langle{\Phi|P^{NZ\ J}\Phi}\rangle$')
            ax[1].legend()
            fig.suptitle(f'{nucl[1]} diagonal-PAV $K={K}/2^{_parity_str}$')
            plt.tight_layout()
            plt.savefig(FOLDER2SAVE / f"Kpav_proj_K{K}_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf")
            
        plt.show()

def _plotK_SPindependent_vapTES(folders_2_import, MAIN_FLD_TEMP, K_vals, parity=0):
    """
    Plot All export_tes files for each K_P_VAP block, folders labeled as the 
    sp state in antoine format.
    """
    FOLDER2SAVE= Path(MAIN_FLD)#.parent
    global GLOBAL_TAIL_INTER
    _parity_str = '+' if parity == 0 else '-'
    for folder_args in folders_2_import:
        index_b20 = {}
        b20_hfb,  b20_vap_Ksp  = [], {}
        data_hfb, data_vap_Ksp = [], {}
        E_hfb, Jz_hfb, E_vap_K_sp = [], [], {}
        
        z, n = folder_args[0]
        _core_0 = 0
        z_real, n_real = z + _core_0, n + _core_0
        nucl = (f"{z_real+n_real}{elementNameByZ[z_real]}", 
                rf"$^{{{z_real+n_real}}}${{{elementNameByZ[z_real]}}}")
        
        print(f"[{z}, {n}]   ******** ")
        FLD  = folder_args[1].format(MAIN_FLD=MAIN_FLD, z=z, n=n)
        FLD  = Path(FLD)
        
        FLD_KVAP = [(k, FLD / f"{k}_{parity}_VAP") for k in K_vals]
        FLD_KVAP = dict(FLD_KVAP)
        
        if not FLD.exists():
            print(f" [ERROR Imprt] Folder [{FLD}] not found, skipping!")
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
                Jz_hfb.append(obj_.Jz)
        
        ## Read VAP-K blocked state - along wiht the PAV folder(diagonal) if exists.
        sh_nlj, sp_by_sh_ = {}, {}
        for K in K_vals:
            print(" * Importing K =", K)
            b20_vap_Ksp[K], data_vap_Ksp[K] = {}, {}
            # with open(export_fn.replace('TESb20_', f'TESb20_K{K}'), 'r') as f:
            exp_fn_k = export_fn.replace('TESb20_', f'TESb20_K{K}_')
            _fldk = FLD_KVAP[K]
            list_sps = list(filter(lambda x: os.path.isdir(_fldk/x) and x.isdigit(), 
                                   os.listdir(_fldk)))
            for sp_ in list_sps:
                sh_ = int(sp_)
                sh_nlj[sp_] = sh_//10000, (sh_%10000)//100, (sh_%10000)%100
                sp_by_sh_[sh_] = sp_
                b20_vap_Ksp[K][sp_], data_vap_Ksp[K][sp_] = [], []
                with open(FLD_KVAP[K] / f"{sp_}/{exp_fn_k}", "r") as f:
                    for line in f.readlines()[1:]:
                        head_, line = line.split(OUTPUT_HEADER_SEPARATOR)
                        i, b20 = head_.strip().split()
                        if i not in index_b20: 
                            print(" [Warning] deformation index", i, "not from VAP.")
                        b20_vap_Ksp[K][sp_].append(float(b20))
                        
                        obj_ = DataTaurus(z, n, None, empty_data=True)
                        obj_.setDataFromCSVLine(line)
                        data_vap_Ksp[K][sp_].append(obj_)
        ## Figures,
        ## Intrinsic states for all K and diagonal PAV
        fig, ax =  plt.subplots(1, 1, figsize=(6, 4))
        fig2, ax2 =  plt.subplots(1, 1, figsize=(6, 4))
        _markers_, _colors_ = '.*o^v<>+d', ['red', 'blue', 'green', 'magenta', 'orange', 'olive']
        _L = 'spdfghij'
        for K in data_vap_Ksp:
            i=0
            for sh_, dat_list in data_vap_Ksp[K].items():
                x = [obj_.b20_isoscalar for obj_ in dat_list]
                ye = [obj_.E_HFB         for obj_ in dat_list]
                yj = [obj_.Jz            for obj_ in dat_list]
                sh_ = int(sh_)
                sp_ = sp_by_sh_[sh_]
                sp_str = K, sh_nlj[sp_][0], _L[sh_nlj[sp_][1]], sh_nlj[sp_][2]
                # sp_str = "${}{}_{{{{{}}}/2}}$".format(*sp_str)
                sp_str = "2K={}: {}{}_{}/2".format(*sp_str)
                ax.scatter(x, ye, label=sp_str,
                           marker=_markers_[i], facecolor='None',
                           color=_colors_[K_vals.index(K)],)
                ax2.scatter(x, yj, label=sp_str,
                            marker=_markers_[i], facecolor='None',
                            color=_colors_[K_vals.index(K)],)
                i+=1
        #ax.set_ylim( (-145, -85) )
        ax.set_title(f"Results for all $K^{_parity_str}$ for each starting s.p. state blocked.")
        ax.plot(b20_hfb, E_hfb, '.-k', label='false OE')
        ax.set_xlabel(r'$\beta_{20}$')
        ax.set_ylabel(r'$E_{HFB}$')
        # ax.legend()
        
        ax2.set_title(f"$<J_z>$ for all $K^{_parity_str}$ for each starting s.p. state blocked.")
        ax2.plot(b20_hfb, Jz_hfb, '.-k', label='false OE')
        ax2.set_xlabel(r'$\beta_{20}$')
        ax2.set_ylabel(r'$\langle{J_z}\rangle$')
        ax2.legend()
        fig.tight_layout()
        fig.savefig(FOLDER2SAVE / f"blocking_Kvap_spIndividual_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf")
        fig2.tight_layout()
        fig2.savefig(FOLDER2SAVE / f"blocking_Kvap_spIndividual_Jz_{nucl[0]}{GLOBAL_TAIL_INTER}.pdf")
        plt.show()
        
                    

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

def _getEnergySps(sp1_str, sp2_str):
    "0s1/2(+1)"
    _ls = 'spdfghijk'
    n1, n2 = sp1_str[0], sp1_str[0]
    l1, l2 = _ls.index(sp1_str[1]), _ls.index(sp2_str[1])
    m1 = int(sp1_str.split('(')[1].replace(')', ''))
    m2 = int(sp2_str.split('(')[1].replace(')', ''))
    
    e = 2*(int(n1) + int(n2)) + l1 + l2 + 0.01*m1 + 0.011*m2
    return e
        
def _plotOddOddAllCombinations(z, n, inter, main_fld):
    """
    Exportage of the results from the analysis of K-Pi from every blockeable 
    combinations of proton/neutron states.
    """
    def_map = {-4: -0.400, -3: -0.275, -2: 0.007, -1: 0.260, 0: 0.527, 1: 0.653}
    
    MAIN_FLD = Path(main_fld) / f'BU_folder_{inter}_z{z}n{n}'
    if not MAIN_FLD.exists(): return
    
    MAIN_FLD = MAIN_FLD / 'VAP_BLOCKINGS'
    
    
    list_defs_flds, defs_flds = [], []
    for x in os.listdir(MAIN_FLD):
        if os.path.isdir(MAIN_FLD / x):
            list_defs_flds.append(MAIN_FLD / x)
            x = int(x.replace('_', '-').replace('def', ''))
            defs_flds.append(x)
    
    maps_states  = {}
    data_states  = {}
    set_energies = {}
    sort_sts_blk = {}
    
    for i_def, b20 in def_map.items():
        if not i_def in defs_flds: continue
        print(f"Printing K-Par for def={i_def}: {b20:5.3f}")
        fld_d = f'def{i_def}'.replace('-', '_')
        KP_flds = os.listdir(MAIN_FLD / fld_d)
        
        maps_states [i_def] = {}
        data_states [i_def] = {}
        set_energies[i_def] = {}
        sort_sts_blk[i_def] = {}
        
        for fld_kp in KP_flds:
            K, P = fld_kp.split('_')
            K, P = int(K), int(P)
            
            fld_kp = Path(MAIN_FLD / fld_d) / Path(fld_kp)
            
            maps_states [i_def][(K,P)] = {}
            data_states [i_def][(K,P)] = {}
            set_energies[i_def][(K,P)] = {}
            with open(fld_kp / 'map_folders_sp.dat', 'r') as f:
                for l in f.readlines():
                    l = l.split()
                    i, s1, s2, ss1, ss2 = l
                    e12 = _getEnergySps(ss1, ss2)
                    i = int(i)
                    maps_states[i_def][(K,P)][i] = (int(s1), int(s2), ss1, ss2, e12)
            list_ = [(i, x[4]) for i, x in maps_states[i_def][(K,P)].items()]
            list_.sort(key=lambda x: x[1])
            sort_sts_blk[i_def][(K,P)] = deepcopy(list_)
            
            flds_bloc = filter(lambda x: os.path.isdir(fld_kp / x) and x.isdigit(), 
                               os.listdir(fld_kp))
            flds_bloc = list(map(lambda x: int(x), flds_bloc))
            flds_bloc = sorted(flds_bloc)
            for fld_bloc in flds_bloc:
                ## clear the interaction files to save memory
                fld_ = fld_kp / str(fld_bloc)
                for tail_ in ('.2b', '.com', '.sho', '.red'):
                    if os.path.exists(fld_ / (inter + tail_)):
                        os.remove( fld_ / (inter + tail_) )
                    
                data_states[i_def][(K,P)][fld_bloc] = DataTaurus(z, n, fld_ / 'OUT')
                if not data_states[i_def][(K,P)][fld_bloc].properly_finished:
                    print("shit:", i_def, (K,P),fld_bloc)
                    continue
                e_ = getattr(data_states[i_def][(K,P)][fld_bloc], 'E_HFB')
                e_ = f"{e_:5.4f}"
                if not e_ in set_energies[i_def][(K,P)]:
                    set_energies[i_def][(K,P)][e_]  = 1
                else:
                    set_energies[i_def][(K,P)][e_] += 1
            
            sort_ec = [(float(e_), c) for e_, c in set_energies[i_def][(K,P)].items()]
            sort_ec.sort(key=lambda x: x[0])
            
            i, total = 1, 0
            for e_, count_ in sort_ec:
                print(f"   [{K}-{P}] ({i:2}): E={e_:>10}  count={count_:5}")
                i += 1
                total += count_
            print(f"                                   {total:5}")
        ## plot.
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sorted_keys = sorted(list(data_states[i_def].keys()))
        for KP in sorted_keys:
            vals = data_states[i_def][KP]
            K, P = KP
            ps = '+' if P==0 else '-'
            xy = [(i, obj.E_HFB) for i, obj in vals.items()]
            xy = zip(*xy)
            ax[0].plot(*xy, '*-', label=f'K={K//2} {ps}')
            
            y = []
            x = [i+1 for i in range(len(sort_sts_blk[i_def][KP]))]
            for i, _ in sort_sts_blk[i_def][KP]:
                y.append(vals[i].E_HFB)
            ax[1].plot(x, y, '*-', label=f'K={K//2} {ps}')
        
        ax[0].set_title ('Energy by blocked states p-n.')
        ax[0].legend()
        ax[0].set_xlabel('states-duplet index')
        ax[0].set_ylabel('E (MeV)')
        ax[1].set_title ('States sorted by s.p. energies E(sp1)+E(sp2)')
        ax[1].set_xlabel('states-duplet index (sorted)')
        ax[1].legend()
        plt.savefig(MAIN_FLD / f'testBlockingRandomizationOddOdd22Na_d{i_def}.pdf')
        plt.show()
            
        
        

if __name__ == '__main__':
    #===========================================================================
    # # PLOT FROM FOLDERS
    #===========================================================================
    SUBFLD_ = '../BU_folder_B1_MZ3_z2n1/PNAMP/'
    
    K_val  = 1
    
    list_dat_file = f'list_k{K_val}_pav.dat'
    
    MAIN_FLD = '..'
    MAIN_FLD = '../DATA_RESULTS/SD_Kblocking/K_blocking_noPAV'
    MAIN_FLD = '../DATA_RESULTS/SD_Kblocking/K{K_val}_block_PAV'
    MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_fewDefs/K{K_val}_block_PAV'
    # MAIN_FLD = '../DATA_RESULTS/example_singleJ/K{K_val}_block_PAV'
    MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Mg'
    # MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Cl'
    # MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Mg1ststateSwap_multiK'
    
    nuclei = [( 7, 8 + 2*i)  for i in range(0, 1)] # 7
    # nuclei = [( 9, 8 + 2*i) for i in range(0, 1)] # 7
    # nuclei = [(11, 8 + 2*i) for i in range(0, 1)] # 7
    nuclei = [(12,11 + 2*i)  for i in range(0, 6)] # 6
    # nuclei = [(15,  8 + 2*i) for i in range(0, 6)]
    nuclei = [(17,10 + 2*i)  for i in range(5, 6)]
    nuclei = [( 9,16), ( 9,18), ( 9,20),]
    nuclei = [(12,13), ]#( 0,10), ] #(12,19), (1, 12)]
    # nuclei = [(12,11 + 2*i) for i in range(0, 6)] # 6
    # nuclei = [(15,14),  ] # (17,12),
    # nuclei = [( 9,20),  ]
    # nuclei = [( 4, 5), ]
    
    GLOBAL_TAIL_INTER = '_B1'
    INTERACTION_TITLE = 'B1_MZ4'
    
    folders_2_import = [
        # ((2, 1), '../BU_folder_B1_MZ3_z2n1/PNAMP/', DataTaurusPAV),
        #((2, 3), '../BU_folder_B1_MZ3_z2n3/PNAMP/', DataTaurusPAV),
        # ((8, 11), f'{MAIN_FLD}/BU_folder_B1_MZ3_z8n11/', 'export_TESb20_z{}n{}_B1_MZ3.txt'),
        
        ((z, n), '{MAIN_FLD}/BU_folder_B1_MZ4_z{z}n{n}_axial/', 'export_TESb20_z{}n{}_B1_MZ4.txt') for z,n in nuclei
        # ((z, n), '{MAIN_FLD}/BU_folder_B1_MZ4_z{z}n{n}/', 'export_TESb20_z{}n{}_B1_MZ4.txt') for z,n in nuclei
        # ((z, n), '{MAIN_FLD}/BU_folder_SDPF_MIX_J_z{z}n{n}/', 'export_TESb20_z{}n{}_SDPF_MIX_J.txt') for z,n in nuclei
        # ((z, n), '{MAIN_FLD}/BU_folder_usdb_JF27_z{z}n{n}/', 'export_TESb20_z{}n{}_usdb_JF27.txt') for z,n in nuclei
        # ((z, n), '{MAIN_FLD}/BU_folder_usdb_JO26_z{z}n{n}/', 'export_TESb20_z{}n{}_usdb_JO26.txt') for z,n in nuclei
        # ((z, n), '{MAIN_FLD}/BU_folder_usdb_J_z{z}n{n}/', 'export_TESb20_z{}n{}_usdb_J.txt') for z,n in nuclei
        #  '{MAIN_FLD}/BU_folder_B1_h11o2_z{z}n{n}/', 
        #  'export_TESb20_z{}n{}_B1_h11o2.txt') for z,n in nuclei
    ]
    # K_val = 3
    # _plotPAVresultsFromFolders(folders_2_import, MAIN_FLD, K_val, parity=0,
    #                            plot_SCL_interpolation=False)
    # raise Exception("STOP HERE")
    
    K_vals  = [1, 3, 5,]
    # _plotPAVresultsFromFolders_mulipleK(folders_2_import, MAIN_FLD, K_vals,
    #                                     plot_SCL_interpolation=1,
    #                                     Jmax_2_plot=9)
    _plotPAVresultsSameFolder_mulipleK(folders_2_import, MAIN_FLD, K_vals,
                                       plot_PPR_interpolation=1, parity=0,
                                       Jmax_2_plot=11)
    # _plotKblockedSurfacesWithMultiplets(folders_2_import, MAIN_FLD, K_vals, 
    #                                     parity=0, plot_PAV=True, max_2_plot=11)
    # _plotK_SPindependent_vapTES(folders_2_import, MAIN_FLD, K_vals)
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
        (12, 19), #(12, 13), (12, 15), #(12, 17), # (12, 9),
        # (13, 8), (13, 10), (13, 12), (13, 14), (13, 16), (13, 18),
        # (14, 9), (14, 11), (14, 13), (14, 15), (14, 17), (14, 19),
    ]
    folders2import = dict([
        ((z, n), f'BU_folder_B1_MZ{MZMAX}_z{z}n{n}/')
        for z, n in nuclei])
    
    _plotscript1_OEK_withNoProjection(nuclei, K_MAX, MZMAX, 
                                      folders2import=folders2import, 
                                      main_folder='../DATA_RESULTS/SD_Kblocking_multiK/Mg_31/')
    
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
    # _plotscript1_OEK_withNoProjection(nuclei, K_MAX, MZMAX)
    # -------------------------------------------------------------------------
    
    #===========================================================================
    # PLOT TEST M-F VAP RESULTS FROM EVERY COMBINATION OF ODDODD SP BLOCKINGS
    #===========================================================================
    args = (11, 11, 'B1_MZ4', 
            '../DATA_RESULTS/SD_Kblocking_multiK/test_AllSPcombinationsForOddOdd/')
    _plotOddOddAllCombinations(*args)
    
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
