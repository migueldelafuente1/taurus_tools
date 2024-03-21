'''
Created on 21 mar 2024

@author: delafuente
'''
from tools.plotter_1d import _Plotter1D, Plotter1D_Taurus,\
    Plotter1D_CanonicalBasis
from tools.helpers import elementNameByZ
from _legacy.exe_isotopeChain_taurus import DataTaurus
from copy import deepcopy

__K_COLORS = ['red', 'blue', 'green', 'magenta', 'cyan', 'yellow', 'black']
if __name__ == '__main__':
    
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
        (12, 19), 
        # (12, 21),
        # (8, 9),
        # (12, 22),
        ]
    K_MAX = 5
    MZMAX = 3
    # -------------------------------------------------------------------------
    
    for z, n in nuclei:
        NUC = elementNameByZ[z]
        
        SUBFLD_ = f'B1/{NUC}/{z+n}/HFB/'  
        # SUBFLD_ = f'Mg_B1/{z+n}/VAP9/' 
        _Plotter1D.setFolderPath2Import('../DATA_RESULTS/K_OddEven/'+SUBFLD_)
        _Plotter1D.EXPORT_PDF_AND_MERGE = True
        FLD_ = _Plotter1D.FOLDER_PATH
        print(FLD_)
        ## Set the files and the labels to plot
        aa = z+n
        files_, labels_by_files, kwargs = [], [], {}
        for K in range(-K_MAX, K_MAX+1, 2):
            _fld = f"{K}_VAP".replace('-', '_')
            files_.append(f"{_fld}/export_TESb20_K{K}_z{z}n{n}_B1_MZ{MZMAX}.txt")
            kwargs[files_[-1]] = dict(linestyle='', #'--',
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
        plt_obj.setTitle(f"$Quadrupole\ TES,\ ^{{{z+n}}}Mg\ B1$")
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
        
        attr2plot_list = ['E_HFB', 'hf', 'pair', 'r_isoscalar', 'b40_isoscalar', 
                          'b30_isoscalar', 'Jz', 'Jz_2']  # DataTaurus
        for attr2plot in attr2plot_list:
            plt_obj.setXlabel(r"$\beta_{20}$")
            plt_obj.setLabelsForLegendByFileData(labels_by_files)
            plt_obj.defaultPlot(attr2plot, show_errors=True,
                                show_plot=attr2plot==attr2plot_list[-1])
        plt_obj.mergeFiguresInto1PDF(f'{NUC}{z+n}_MZ{MZMAX}_Kblocking.pdf')
        
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