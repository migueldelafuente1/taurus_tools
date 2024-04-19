'''
Created on 21 mar 2024

@author: delafuente
'''
import os
import shutil
import numpy as np

from tools.plotter_1d import _Plotter1D, Plotter1D_Taurus,\
    Plotter1D_CanonicalBasis
from tools.helpers import elementNameByZ
# from _legacy.exe_isotopeChain_taurus import DataTaurus
from copy import deepcopy
from tools.data import DataTaurus, DataTaurusPAV

__K_COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black']

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
        print(FLD_)
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


def particlePlusRotor_SCL_spectra(data_1o2, data_3o2, b20_deform, J2_vals, K_val):
    """
    Get the interpolation curves for two set of data, the first J values must be
    in order in J_vals. 
    
    The formula is for the strong coupling limit: E = A + B (J(J+1) - K**2)
    """
    from scipy.interpolate import interp1d
    K_val /= 2
    J_vals = [j/2 for j in J2_vals]
    
    x_new = np.linspace(min(b20_deform), max(b20_deform), 101, endpoint=True)
    y01 = np.array(data_1o2)
    y02 = np.array(data_3o2)
    
    B = (y01 - y02) / (J_vals[0]*(J_vals[0] + 1) - J_vals[1]*(J_vals[1] + 1))
    A = y01 - B*(J_vals[0]*(J_vals[0] + 1) - K_val**2)
    
    y0 = [y01, y02,]
    for J in J_vals[2:]:
        y0.append(A + B*(J*(J+1) - K_val**2))
    
    data_interpolated = {}
    for i, y_j in enumerate(y0):
        J = J_vals[i]
        # Create the interpolation function
        interp_func = interp1d(b20_deform, y_j, kind='cubic') # 'quadratic'
        
        # Generate points for the interpolated curve
        y_new = interp_func(x_new)
        
        data_interpolated[J2_vals[i]] = (x_new, y_new)
    
    return data_interpolated

def _plotPAVresultsFromFolders(folders_2_import):
    """
    folders_2_import: <list> of ((z, n), folder path, Datatype, )
    """
    for folder_args in folders_2_import:
        FLD_, list_dat_file = None, None
        
        for i, arg in enumerate(folder_args):
            if i == 0:
                z, n = arg
            elif i == 1:
                assert os.path.exists(arg), "Unfound Folder"
                FLD_ = arg if arg.endswith('/') else arg+'/'
                FLD_PNPAMP = FLD_ + 'PNAMP/'
            elif i == 2:
                export_VAP_file = arg.format(z, n)
            elif i == 3:
                assert os.path.exists(FLD_+arg), "unfound list.dat file"
                list_dat_file = arg
        
        if not list_dat_file:
            list_dat_file = filter(lambda x: x.endswith('.dat'), os.listdir(FLD_PNPAMP))
            list_dat_file = list(filter(lambda x: x.startswith('list'), list_dat_file))
            if len(list_dat_file)== 1: 
                list_dat_file = list_dat_file[0]
            else: Exception("unfound unique list.dat file for data order.")
        
        
        with open(FLD_PNPAMP+list_dat_file, 'r') as f:
            K = int(list_dat_file.replace('_pav.dat', '').replace('list_k', ''))
            list_dat_file = [ff.strip() for ff in f.readlines()]
            b20_deform = [ff.replace('d','').replace('.OUT','') for ff in list_dat_file]
            b20_deform = [int(ff.split('_')[1]) for ff in b20_deform]
        
        data = [DataTaurusPAV(z, n, FLD_PNPAMP+f) for f in list_dat_file]
        
        _ = 0
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1)
        energy_by_J = {}
        for id_, dat in enumerate(data):
            kval = [(i, k) for i, k in enumerate(dat.KJ)]
            kval = list(filter(lambda x: x[1]==K, enumerate(dat.KJ)))
            
            ener, jval = [], []
            for i, _ in kval:
                ener.append(dat.proj_energy[i])
                jval.append(dat.J[i])
            
            for i, J in enumerate(jval):
                if J not in energy_by_J: energy_by_J[J] = []

                energy_by_J[J].append(ener[i])
        
        E_vap, b20_vap = [], []
        ## Import the values form HFB.
        with open(FLD_+export_VAP_file, 'r') as f:
            dataVAP = f.readlines()
            Dty_, _ = dataVAP[0].split(',')
            from tools.helpers import OUTPUT_HEADER_SEPARATOR
            dataVAP = [l.split(OUTPUT_HEADER_SEPARATOR) for l in dataVAP[1:]] 
            
            
            for h, line in dataVAP:
                i, b20 = h.split(':')
                i, b20 = int(i), float(b20)
                
                b20_vap.append(b20)
                if i in b20_deform:
                    b20_deform[b20_deform.index(i)] = b20
                
                res = DataTaurus(z, n, None, empty_data=True)
                res.setDataFromCSVLine(line)
                E_vap.append(res.E_HFB)
        
        
        # b20_deform = [i-(len(energy_by_J[1])//2) for i in range(len(energy_by_J[1]))]
        # b20_deform = [i for i in range(len(energy_by_J[1]))]
        J_vals = [i for i in energy_by_J.keys()]
        interp = particlePlusRotor_SCL_spectra(energy_by_J[1], energy_by_J[3], 
                                               b20_deform, J_vals, 1)
        
        for J, ener in energy_by_J.items():
            ax.plot(b20_deform, ener, '*-', label=f'J={J}/2 +')
        for J in energy_by_J.keys():
            ax.plot(*interp[J], '--', label=f'PPRM-SCL J={J}/2 +')
        ax.plot(b20_vap, E_vap, "k.-", label='HFB')
        
        plt.title("PAV projections for K=+1/2 and PPRM interpolation")
        plt.legend()
        plt.show()



if __name__ == '__main__':
    
    #===========================================================================
    # # PLOT FROM FOLDERS
    #===========================================================================
    if True:
        SUBFLD_ = '../BU_folder_B1_MZ3_z2n1/PNAMP/'
        list_dat_file = 'list_k1_pav.dat'
        
        MAIN_FLD = '..'
        MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_results'
        folders_2_import = [
            # ((2, 1), '../BU_folder_B1_MZ3_z2n1/PNAMP/', DataTaurusPAV),
            #((2, 3), '../BU_folder_B1_MZ3_z2n3/PNAMP/', DataTaurusPAV),
            ((8, 11), f'{MAIN_FLD}/BU_folder_B1_MZ3_z8n11/', 'export_TESb20_z{}n{}_B1_MZ3.txt'),
        ]
        _plotPAVresultsFromFolders(folders_2_import)
        raise Exception("STOP HERE")
        
        K_MAX = 9
        MZMAX = 4
        nuclei = [
            # (8, 9), (8, 11), (8, 13), 
            (9, 8), (9, 10), (9, 12), (9, 14),
            # (10, 9), (10, 11), (10, 13), (10, 15),
            # (11, 8), (11, 10), (11, 12), (11, 14),# 
            (12, 11), (12, 13), (12, 15),(12, 17), # (12, 9),
            # (13, 8), (13, 10), (13, 12), (13, 14), (13, 16), (13, 18),
            # (14, 9), (14, 11), (14, 13), (14, 15), (14, 17), (14, 19),
        ]
        folders2import = dict([
            ((z, n), f'BU_folder_B1_MZ{MZMAX}_z{z}n{n}/')
            for z, n in nuclei])
        
        _plotscript1_OEK_withNoProjection(nuclei, K_MAX, MZMAX, 
                                          folders2import=folders2import, 
                                          main_folder='../DATA_RESULTS/SD_Kblocking_results/')
        
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