'''
Created on 15 feb 2024

@author: delafuente

Plot from DATA_RESULTS the evolution of single particle states.

'''
import os, shutil
import numpy as np
# from tools.executors import 
from tools.data import DataTaurus, EigenbasisData, OccupationNumberData

# AA = 28
def _evolutionOfEigenstatesWithoutDeformationTES(AA, show_plot=False):
    """
    The difference between this and the P
    """
    #===========================================================================
    # 
    FLD_IMPORT = f'../DATA_RESULTS/Beta20/Mg_GDD_test/{AA}/FOLDER_GDD_A{AA}/'
    FLD_EXPORT = f'../DATA_RESULTS/Beta20/Mg_GDD_test/'
    ## Note:  eigenbasis_h   does not have [H11, v2]
    ##        eigenbasisH11  does not have [h, v2, fermi_energ_neutr/prot]
    ##        canonicalbasis does not have [H11]
    FILE_DAT = 'eigenbasis_h'
    
    frac_t3_2print = []
    frac_t3 = []
    attr_2_plot = ['h', ]
    #===========================================================================
    
    list_files = list(filter(lambda x: x.startswith(FILE_DAT), 
                             os.listdir(FLD_IMPORT)))
    data = []
    for fd_ in sorted(list_files):
        t3_ = fd_.replace('.dat', '').split('_')[-1]
        if int(t3_) > 260: continue
        frac_t3.append( int(t3_) / 100)
        frac_t3_2print.append(t3_)
        
        res  = EigenbasisData(FLD_IMPORT+fd_)
        res.getResults()
        data.append(res)
    
    ## separate between proton and neutron states
    _ = 0
    L_KEYS = "spdfghijk"
    results_by_pn = [{}, {}] ## proton - neutron
    labels_nlj_by = [{}, {}]
    for attr_ in attr_2_plot:
        for it in (0, 1):
            results_by_pn[it][attr_] = {}
            labels_nlj_by[it][attr_] = {}
        
        for i, res in enumerate(data):
            t3_ = frac_t3_2print[i]
            levels = getattr(res, attr_)
            is_neutron = [int(x > 0.5) for x in getattr(res, 'avg_neutron')]
            
            for it in (0, 1):
                results_by_pn[it][attr_][t3_] = []
                labels_nlj_by[it][attr_][t3_] = []
            
            for k in range(len(levels)):
                label_ = [
                    "{:1.0f}".format( getattr(res, 'avg_n')[k]),
                    L_KEYS[int(round(getattr(res, 'avg_l')[k])) ],
                    "{:1.0f}" .format(np.round(getattr(res, 'avg_j')[k]+.1,1)*2),
                ]
                degenerate_lev = False
                for e_ in results_by_pn[is_neutron[k]][attr_][t3_]:
                    degenerate_lev = abs(e_ - levels[k]) < 0.1
                
                label_1 = "".join(label_ + ["/2", ])
                # if label_ in [ii[1] for ii in labels_nlj_by[is_neutron[k]][attr_][t3_]]:
                if label_1 in labels_nlj_by[is_neutron[k]][attr_][t3_]:
                    if not degenerate_lev:
                        label_.append("({:+1.0f})"
                            .format(np.round(getattr(res,'avg_jz')[k]+.1)*2))
                        label_1 = "".join(label_)
                    else:
                        label_1 = None
                results_by_pn[is_neutron[k]][attr_][t3_] .append(levels[k])
                labels_nlj_by[is_neutron[k]][attr_][t3_] .append(label_1)
    ## plot
    COLORS_ = "kbgrcmy"*2
    import matplotlib.pyplot as plt
    for attr_ in attr_2_plot:
        
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        for pn_i in (0, 1):            
            y_levels = []
            for ii, t3_ in enumerate(frac_t3_2print):
                for lev_i, lev in enumerate(results_by_pn[pn_i][attr_][t3_]):
                    if ii == 0:
                        y_levels.append([lev, ])
                    else:
                        y_levels[lev_i].append(lev)
            
            for lev_i in range(len(y_levels)):
                label_ = labels_nlj_by[pn_i][attr_][frac_t3_2print[-1]][lev_i]
                
                ## select the color by N
                N_sh = 2*int(label_[0])+L_KEYS.index(label_[1]) if label_ else -1
                color_  = COLORS_[ N_sh]
                linestyle_= '--' if label_ != None else ''
                marker_ = '*' if label_ != None else None
                
                kwargs = dict(label=label_, linestyle=linestyle_, 
                              marker=marker_, color=color_)
                ax[pn_i].plot(frac_t3, y_levels[lev_i], **kwargs)
            
            ax[pn_i].axvline(x=1.0, color='m', linestyle='--')
            ax[pn_i].set_title ("protons" if pn_i == 0 else 'neutrons')
            ax[pn_i].set_xlabel(r"$t_3'/t_3\ fraction$")
            ax[pn_i].legend(fontsize='xx-small')
            
        ax[0].set_ylabel(attr_)
        fig.suptitle(f"Single particle ({attr_}) evolution ({FILE_DAT})  Mg {AA}",
                     fontsize= 12, fontweight= 'bold')
        pdf_ = f"{FLD_IMPORT}sp_{attr_}_evolution_t3fraction_vs_EDF_A{AA}.pdf"
        fig.savefig(pdf_)
        shutil.copy(pdf_, FLD_EXPORT)
    
    if show_plot: plt.show()


#===============================================================================
# 
#===============================================================================
if __name__ == '__main__':
    
    for AA in range(20, 35, 2):
        _evolutionOfEigenstatesWithoutDeformationTES(AA, AA==34)
    
    