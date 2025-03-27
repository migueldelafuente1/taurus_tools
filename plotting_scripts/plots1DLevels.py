'''
Created on 3 sept 2024

@author: delafuente

Simpler scripts for plotting excitation levels for publishing
'''
from tools.plotter_1d import MATPLOTLIB_INSTALLED
from tools.plotter_levels import getAllLevelsAsString
from tools.helpers import elementNameByZ

if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "xtick.labelsize" : 14,
        "ytick.labelsize" : 14,
    })
else:
    MATPLOTLIB_INSTALLED = False
    print("WARNING :: Matplotlib not installed. Do not evaluate plot modules.")


class _EnergyLevelSimpleGraph():
    
    """
    Plots like this (no energy indication)
    
            TITLE, Results And Details
        
               0+ -------  2- 5+ ========
        E                  1+ 3- ========   
        N   2+ 1+ =======
        E      3+ -------
        R                     5+ --------
               0+ -------     0+ --------
        
               20^Ne(20.3)     22^O(15.6)
    """
    
    RELATIVE_PLOT      = True # always
    ONLY_PAIRED_STATES = False
    MAX_NUM_OF_SIGMAS  = 99
    MAX_ENERGY_DISPLAY = 99
    Y_BOX_HEIGTH       = 0.02
    X_BOX_MARGIN       = 0.05
    
    def __init__(self, title=None):
        
        self.global_title = title
        
        self._E     = []
        self._E_abs = []
        self._fundamental_energies = []
        self._JP    = []
        self._plot_y_vals = []
        self._plot_x_vals = []
        self._plot_titles = []
        
        self._max_group   = []
        self._plot_data   = []
        self._plot_data_indexing = []
    
    def setData(self, levels_data, 
                title_data='', from_pav=False, direct_vals=False,
                first_n_levels = None):
        """
        Import the levels from strings, default lines from taurus_MIX output:
            J  P   n    Energy     E_exc 
          1/2  +   1   -142.800    0.000   ...
          1/2  +   2   -134.419    8.381   ...
        :from_pav
            J 2*MJ 2*KJ    P |     1     |      E     |
            5    5    5    1   0.14406434   -139.24448   ...
        :direct_vals
            [(j, p, E abs), ...]
        """
        
        self._plot_titles.append(title_data)
        item_ = self._E.__len__()
        
        self._E.append([])
        self._E_abs.append([])
        self._JP.append([])
        self._plot_x_vals.append([])
        self._plot_y_vals.append([])
        
        ## process _plot data text to E and J-P
        if direct_vals:
            for line in levels_data:
                jt_str = f"{line[0]}/2" if line[0]%2==1 else f"{line[0]/2}"
                _p = '+' if line[1]== 1 else '-'
                                
                self._E_abs[item_]   .append(line[2])
                self._JP[item_]   .append(jt_str+_p)
            self._E[item_] = [e - min(self._E_abs[item_]) for e in self._E_abs[item_]]
        elif from_pav:
            for line in levels_data.split('\n'):
                if line == '': continue
                line = line.strip().split()
                line = [x.strip() for x in line]
                j    = int(line[0])
                jt_str = f"{j}/2" if j%2==1 else f"{j/2}"
                _p = '+' if line[3]=='1' else '-'
                
                self._E_abs[item_]   .append(float(line[5]))
                self._JP[item_]   .append(jt_str+_p)
            self._E[item_] = [e - min(self._E_abs[item_]) for e in self._E_abs[item_]]
        else:  
            ## HWG
            for line in levels_data.split('\n'):
                if line == '': continue
                line = line.strip().split()
                jt_str = line[0] + line[1]
                
                self._E [item_]   .append(float(line[4]))
                self._E_abs[item_].append(float(line[3]))
                self._JP[item_]   .append(jt_str)
        
        self._fundamental_energies.append( min(self._E_abs[item_]))
        for i, e in enumerate(self._E_abs[item_]):
            self._E[item_][i] = e - self._fundamental_energies[item_]
        
        ## ensure the levels are shorted
        for n in range(len(self._E [item_]) - 1, 0, -1):
            # Inner loop to compare adjacent elements
            for i in range(n):
                if  self._E [item_][i] > self._E [item_][i+1]:
                    for name in ('_E', '_E_abs', '_JP'):
                        a = getattr(self, name)[item_][i]
                        b = getattr(self, name)[item_][i+1]
                        
                        getattr(self, name)[item_][i+1] = a
                        getattr(self, name)[item_][i]   = b
        
        self._clearStates(item_, first_n_levels)
        self._groupByEnergyRange(item_)
    
    def _clearStates(self, item_, first_n_levels=None):
        
        """ 
        Remove states that exceed the number of sigma-excitation and/or the
        energy range to display.
        """
        states = set(self._JP[item_])
        count_sigmas = dict([(jp, 0) for jp in states])
        
        if first_n_levels == None:  first_n_levels = 99
        
        index_rm = []
        for i, e in enumerate(self._E[item_]):
            jp = self._JP[item_][i]
            
            if e > self.MAX_ENERGY_DISPLAY:
                index_rm.append(i)
                continue
            
            if count_sigmas[jp] > self.MAX_NUM_OF_SIGMAS:
                index_rm.append(i)
                continue
            
            ## This removes individual states when exceeded a certain value
            if first_n_levels > 0:
                first_n_levels -= 1
            else:
                index_rm.append(i)
                continue
            
            count_sigmas[jp] += 1
        
        index_rm.sort(reverse=True)
        for i in index_rm:
            self._E [item_]   .pop(i)
            self._E_abs[item_].pop(i)
            self._JP[item_]   .pop(i)
    
    def _groupByEnergyRange(self, item_):
        
        y_box  = min(self.MAX_ENERGY_DISPLAY, max(self._E[item_])) 
        y_box *= self.Y_BOX_HEIGTH
        y_box  = round(y_box, 1)
        
        plotteable_list = [[], ]
        index_list      = []
        e_i = 0.0
        for i, e in enumerate(self._E[item_]):
            if e > e_i + y_box:
                e_i = e
                plotteable_list.append([])
                self._plot_y_vals[item_].append(e_i)
                self._plot_x_vals[item_].append(  0)
            
            plotteable_list[-1].append( [self._JP[item_][i],
                                         e, self._E_abs[item_][i]] )
            index_list.append( (len(plotteable_list) - 1, 
                                len(plotteable_list[-1]) - 1))
        
        self._max_group.append( max([len(i) for i in plotteable_list]) )
        self._plot_data.append( plotteable_list )
        self._plot_data_indexing.append(index_list)
        
    def _shiftToTheAbsoluteEnergy(self):
        """
        To get the picture of non relative excitation energies to compare 
        between groups, append the excess energy with respect to the fundamental
        energy
        """
        E_0 = min(self._fundamental_energies)
        for item_ in range(len(self._E)):
            E_0_diff = self._fundamental_energies[item_] - E_0
            for i in range(len(self._E[item_])):
                # self._E[item_][i] = self._E[item_][i] + E_0_diff
                k1, k2 = self._plot_data_indexing[item_][i]
                self._plot_data[item_][k1][k2][1] = self._E[item_][i] + E_0_diff
        
    def plot(self, fld2saveWithFilename=None):
        
        if not self.RELATIVE_PLOT: self._shiftToTheAbsoluteEnergy()
        
        items_ = self._plot_data.__len__()
        x_len_global = 5 + (3*items_//2 - 1)        
        x_len = x_len_global / items_
        x_margin = x_len * self.X_BOX_MARGIN
        x_0_item  = 0
        y_max     = -999999999
        
        fig, ax = plt.subplots(1, 1, figsize=(6.5,7))#figsize=(x_len_global, 8))
        print(" States Plotted and energies.")
        for item_, data_it in enumerate(self._plot_data):
            print(f"  * Data set [{item_}]")
            n_ = self._max_group[item_]
            x_unit = x_len / 10.0
            x = [
                x_0_item, 
                x_0_item + ((n_ + 1) * x_unit), 
                x_0_item - x_unit + x_len - x_margin
            ]
            
            for data in data_it:
                jp_str = []
                for sig, elems_ in enumerate(data):
                    ax.plot(x[1:], (elems_[1], )*2, linewidth= 1, color='black')
                    jp_str.append(f"${elems_[0]}$"
                                   .replace('+', '^+').replace('-', '^-'))
                    print(f"  E[{elems_[0]}] = {elems_[1]:9.3f}")
                jp_str = ' '.join(jp_str) + ' '
                y_max = max(y_max, data[0][1] + 0.1)
                ax.annotate(jp_str, 
                            xy= (x[0] + (x[1]-x[0])/2, data[0][1] + 0.1),
                            #xytext=(0, 4),  # 4 points vertical offset.
                            # textcoords='offset points',
                            # fontsize= 9 - len(data) + 1,
                            fontsize= 12 - len(data) + 1,
                            ha='center', va='center',
                            rotation=20*(len(data) != 1),
                            )            
            if item_ == 0: 
                suptitl_ = "\n$E_0=$ {:6.2f}".format(self._fundamental_energies[item_])
            else: 
                suptitl_ = "\n{:6.2f}".format(self._fundamental_energies[item_])
            ax.annotate(self._plot_titles[item_] + suptitl_ ,
                        xy=(x[1] + (x[2]-x[1])/2, 
                            -2 * self.Y_BOX_HEIGTH * self.MAX_ENERGY_DISPLAY ),
                        fontsize= 15,
                        ha='center', va='top')
            
            x_0_item += x_len
        
        ax.set_xlim(            -2*self.X_BOX_MARGIN * x_len, 
                    x_len_global + self.X_BOX_MARGIN * x_len)
        ax.set_ylim( -0.5, y_max )
        ax.spines['top']   .set_visible(False)
        ax.spines['right'] .set_visible(False)
        ax.spines['left']  .set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax .xaxis.set_visible(False)
        # ax .set_ylabel("Excitation Energy (MeV)", fontsize=15)
        fig.suptitle(self.global_title, size=15)
        fig.tight_layout()
        
        if fld2saveWithFilename:
            if (isinstance(fld2saveWithFilename, str) 
                and not fld2saveWithFilename.endswith('.pdf')):
                fld2saveWithFilename += 'file.pdf'
            fig.savefig(fld2saveWithFilename)
    


if __name__ == '__main__':
    
    # MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/F_nocom'
    # _levels = []
    # for tail_ in (1, 3, 5):
    #     fld = MAIN_FLD + '/BU_folder_usdb_JF27_z1n10/' + f"{tail_}_0_PNPAMP_HWG/HWG/"
    #     _levels.append(fld)
    # _levels.append(MAIN_FLD + '/BU_folder_usdb_JO26_z0n10/0_0_PNPAMP_HWG/HWG/')
    # _levels.append(MAIN_FLD + '/BU_folder_usdb_JF27_z1n10/kmix_PNPAMP/HWG/')
    #
    # example_levels = [getAllLevelsAsString(fld) for fld in _levels]
    #
    # # _EnergyLevelSimpleGraph.MAX_NUM_OF_SIGMAS  = 10
    # _EnergyLevelSimpleGraph.MAX_ENERGY_DISPLAY = 7.0
    # _graph = _EnergyLevelSimpleGraph("HWG spectra from each K blocking and e-e O core")
    # _graph.setData(example_levels[0], '$^{27}$F K=1/2')
    # _graph.setData(example_levels[1], '$^{27}$F K=3/2')
    # _graph.setData(example_levels[2], '$^{27}$F K=5/2')
    # _graph.setData(example_levels[3], '$^{26}$O')
    # _graph.setData(example_levels[4], '$^{27}$F mix')
    # _graph.plot(fld2saveWithFilename=MAIN_FLD+'/hwg-spectra_F27O26_usdb.pdf')
    
    
    
    # MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Mg'
    # _levels = []
    # for tail_ in (1, 3, 5):
    #     fld = MAIN_FLD + '/BU_folder_usdb_J_z4n5/' + f"{tail_}_0_PNPAMP_HWG/HWG/"
    #     _levels.append(fld)
    # _levels.append(MAIN_FLD + '/BU_folder_usdb_J_z4n4/0_0_PNPAMP_HWG/HWG/')
    # _levels.append(MAIN_FLD + '/BU_folder_usdb_J_z4n5/kmix_PNPAMP/HWG/')
    # _levels.append(MAIN_FLD + '/BU_folder_usdb_J_z5n4/kmix_PNPAMP/HWG/')
    #
    # example_levels = [getAllLevelsAsString(fld) for fld in _levels]
    #
    # # _EnergyLevelSimpleGraph.MAX_NUM_OF_SIGMAS  = 10
    # _EnergyLevelSimpleGraph.RELATIVE_PLOT      = False
    # _EnergyLevelSimpleGraph.MAX_ENERGY_DISPLAY = 7.0
    # _graph = _EnergyLevelSimpleGraph("HWG spectra from each K blocking and e-e O core")
    # _graph.setData(example_levels[0], '$^{25}$Mg K=1/2')
    # _graph.setData(example_levels[1], '$^{25}$Mg K=3/2')
    # _graph.setData(example_levels[2], '$^{25}$Mg K=5/2')
    # _graph.setData(example_levels[3], '$^{24}$Mg')
    # _graph.setData(example_levels[4], '$^{25}$Mg mix')
    # _graph.setData(example_levels[5], '$^{25}$Al mix')
    # _graph.plot(fld2saveWithFilename=MAIN_FLD+'/hwg-spectra_MgAl25Mg26_usdb.pdf')
    #
    
    
    
    # for z, n in ((4, 5), (5, 4)):
    #
    #     MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Mg'
    #     _levels = []
    #     for tail_ in (1, 3, 5):
    #         fld = MAIN_FLD + f'/BU_folder_usdb_J_z{z}n{n}/{tail_}_0_PNPAMP_HWG/HWG/'
    #         _levels.append(fld)
    #     # _levels.append(MAIN_FLD + '/BU_folder_usdb_JO26_z0n10/0_0_PNPAMP_HWG/HWG/')
    #     _levels.append(MAIN_FLD + f'/BU_folder_usdb_J_z{z}n{n}/kmix_PNPAMP/HWG/')
    #
    #     example_levels = [getAllLevelsAsString(fld) for fld in _levels]
    #
    #     X = elementNameByZ[z + 8]
    #
    #     # _EnergyLevelSimpleGraph.RELATIVE_PLOT      = False
    #     # _EnergyLevelSimpleGraph.MAX_NUM_OF_SIGMAS  = 10
    #     _EnergyLevelSimpleGraph.MAX_ENERGY_DISPLAY = 7.0
    #     _graph = _EnergyLevelSimpleGraph("HWG spectra from each K blocking")
    #     _graph.setData(example_levels[0], f'$^{{{25}}}${X} K=1/2')
    #     _graph.setData(example_levels[1], f'$^{{{25}}}${X} K=3/2')
    #     _graph.setData(example_levels[2], f'$^{{{25}}}${X} K=5/2')
    #     # _graph.setData(example_levels[3], '$^{26}$O')
    #     _graph.setData(example_levels[3], f'$^{{{25}}}${X} K-mix')
    #     _graph.plot(fld2saveWithFilename=MAIN_FLD+f'/hwg-spectra_{X}z{z}n{n}_usdb.pdf')
    

    
    # MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Mg_31'
    # _levels = []
    # for tail_ in (1, 3, 5):
    #     fld = MAIN_FLD + '/BU_folder_B1_MZ4_z12n19/' + f"{tail_}_0_PNPAMP_HWG/HWG/"
    #     _levels.append(fld)
    # _levels.append(MAIN_FLD + '/BU_folder_B1_MZ4_z12n19/kmix_PNPAMP/HWG/')
    #
    # example_levels = [getAllLevelsAsString(fld) for fld in _levels]
    #
    # # _EnergyLevelSimpleGraph.MAX_NUM_OF_SIGMAS  = 10
    # _EnergyLevelSimpleGraph.MAX_ENERGY_DISPLAY = 7.0
    # _graph = _EnergyLevelSimpleGraph("HWG spectra from each K blocking and parity-mix")
    # _graph.setData(example_levels[0], '$^{31}$Mg K=1/2')
    # _graph.setData(example_levels[1], '$^{31}$Mg K=3/2')
    # _graph.setData(example_levels[2], '$^{31}$Mg K=5/2')
    # _graph.setData(example_levels[3], '$^{31}$Mg mix')
    # _graph.plot()
    # plt.show()#fld2saveWithFilename=MAIN_FLD+'/hwg-spectra_Mg31_usdb.pdf')
    
    z, n = 17, 20
    MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Cl'
    _levels = []
    for tail_ in (1, 3, 5):
        fld = MAIN_FLD + f'/BU_folder_B1_MZ4_z{z}n{n}/' + f"{tail_}_0_PNPAMP_HWG/HWG/"
        _levels.append(fld)
    _levels.append(MAIN_FLD + f'/BU_folder_B1_MZ4_z{z}n{n}/kmix_PNPAMP/HWG/')
    
    example_levels = []
    example_levels.append([(1,1,-138.80), (5,1,-135.83), (3,1,-134.17), (7,1,-107.72), (9,1,-104.62)])
    
    str_ = """    1    1    1    1   0.04156936   -142.44626  12.0000000 -11  13.0000000  -8  25.0000000  -8   0.50000   0.50000 -14   1.00000   0.54597   0.50000  -8
    3    1    1    1   0.08263686   -142.59769  12.0000000 -10  13.0000000  -8  25.0000000  -8   1.50000   0.50000 -14   1.00000   0.54642   0.50000  -8
    5    1    1    1   0.08319358   -140.28190  12.0000000 -11  13.0000000  -8  25.0000000  -8   2.50000   0.50000 -15   1.00000   0.54523   0.50000  -9
    7    1    1    1   0.10896529   -140.61578  12.0000000 -11  13.0000000  -8  25.0000000  -8   3.50000   0.50000 -15   1.00000   0.54597   0.50000  -9
    9    1    1    1   0.06676973   -136.24634  12.0000000 -10  13.0000000  -8  25.0000000  -8   4.50000   0.50000 -15   1.00000   0.54521   0.50000  -8
   11    1    1    1   0.07710340   -136.69312  12.0000000 -10  13.0000000  -8  25.0000000  -8   5.50000   0.50000 -15   1.00000   0.54566   0.50000  -8"""
    example_levels.append(str_)
    example_levels.append(getAllLevelsAsString(_levels[0]))
    str_ = """    3    3    3    1   0.07557750   -137.53389  12.0000000 -10  13.0000000 -10  25.0000000 -10   1.50000   1.50000 -12   1.00000   0.91161   0.50000 -12
    5    3    3    1   0.09381480   -136.70876  12.0000000 -11  13.0000000 -11  25.0000000 -10   2.50000   1.50000 -12   1.00000   0.91111   0.50000 -12
    7    3    3    1   0.09492244   -135.38811  12.0000000 -10  13.0000000 -10  25.0000000  -9   3.50000   1.50000 -12   1.00000   0.91061   0.50000 -12
    9    3    3    1   0.08499807   -133.91737  12.0000000 -11  13.0000000 -11  25.0000000 -10   4.50000   1.50000 -12   1.00000   0.90956   0.50000 -12
   11    3    3    1   0.06495421   -131.46485  12.0000000 -10  13.0000000 -10  25.0000000 -11   5.50000   1.50000 -12   1.00000   0.90913   0.50000 -12"""
    example_levels.append(str_)
    example_levels.append(getAllLevelsAsString(_levels[1]))
    str_ = """    5    5    5    1   0.14406434   -139.24448  12.0000000 -11  13.0000000 -11  25.0000000 -10   2.50000   2.50000 -12   1.00000   0.62509   0.50000 -12
    7    5    5    1   0.11847903   -137.47126  12.0000000 -10  13.0000000 -10  25.0000000 -10   3.50000   2.50000 -11   1.00000   0.61945   0.50000 -12
    9    5    5    1   0.08090782   -135.11147  12.0000000 -11  13.0000000 -12  25.0000000 -10   4.50000   2.50000 -11   1.00000   0.61554   0.50000 -12
   11    5    5    1   0.04652668   -131.97205  12.0000000 -10  13.0000000 -10  25.0000000  -9   5.50000   2.50000 -13   1.00000   0.61315   0.50000 -12"""
    example_levels.append(str_)
    example_levels.append(getAllLevelsAsString(_levels[2]))
    example_levels.append(getAllLevelsAsString(_levels[3]))
    
    # _EnergyLevelSimpleGraph.MAX_NUM_OF_SIGMAS  = 10
    _EnergyLevelSimpleGraph.MAX_ENERGY_DISPLAY = 12.0
    _EnergyLevelSimpleGraph.RELATIVE_PLOT = False
    _graph = _EnergyLevelSimpleGraph("") #"PVC(K) and HWG spectra from each K blocking")
    # _graph.setData(example_levels[0], 'PNP-VAP',     direct_vals=True)
    # _graph.setData(example_levels[1], 'AMP-PAV\n (1/2)', from_pav =True, first_n_levels=6)
    _graph.setData(example_levels[2], 'PVC\n (1/2)',                   first_n_levels=8)
    # _graph.setData(example_levels[3], 'AMP-PAV\n (3/2)', from_pav =True, first_n_levels=2)
    _graph.setData(example_levels[4], 'PVC\n (3/2)',                   first_n_levels=7)
    # _graph.setData(example_levels[5], 'AMP-PAV\n (5/2)', from_pav =True, first_n_levels=2)
    _graph.setData(example_levels[6], 'PVC\n (5/2)',                   first_n_levels=4)
    _graph.setData(example_levels[7], 'GCM \nK-mix',                   first_n_levels=14)
    _graph.plot(fld2saveWithFilename=MAIN_FLD+f'/pvchwg-spectra_Cl{z+n}_B1.pdf')
    plt.show() #fld2saveWithFilename=MAIN_FLD+'/hwg-spectra_Mg25_B1.pdf')
    
    