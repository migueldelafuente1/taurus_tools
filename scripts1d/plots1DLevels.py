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
    
    def setData(self, levels_data, title_data=''):
        
        self._plot_titles.append(title_data)
        item_ = self._E.__len__()
        
        self._E.append([])
        self._E_abs.append([])
        self._JP.append([])
        self._plot_x_vals.append([])
        self._plot_y_vals.append([])
        
        ## process _plot data text to E and J-P
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
        
        self._clearStates(item_)
        self._groupByEnergyRange(item_)
    
    def _clearStates(self, item_):
        
        """ 
        Remove states that exceed the number of sigma-excitation and/or the
        energy range to display.
        """
        states = set(self._JP[item_])
        count_sigmas = dict([(jp, 0) for jp in states])
        
        index_rm = []
        for i, e in enumerate(self._E[item_]):
            jp = self._JP[item_][i]
            
            if e > self.MAX_ENERGY_DISPLAY:
                index_rm.append(i)
                continue
            
            if count_sigmas[jp] > self.MAX_NUM_OF_SIGMAS:
                index_rm.append(i)
            else:
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
        
        fig, ax = plt.subplots(1, 1, figsize=(5.5,6))#figsize=(x_len_global, 8))
        
        for item_, data_it in enumerate(self._plot_data):
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
                jp_str = ' '.join(jp_str) + ' '
                ax.annotate(jp_str, 
                            xy= (x[0] + (x[1]-x[0])/2, data[0][1] + 0.1),
                            #xytext=(0, 4),  # 4 points vertical offset.
                            # textcoords='offset points',
                            fontsize= 9 - len(data) + 1,
                            ha='center', va='center',
                            rotation=20*(len(data) != 1),
                            )            
            suptitl_ = "\n$E_0=${:6.2f}".format(self._fundamental_energies[item_])
            ax.annotate(self._plot_titles[item_] + suptitl_ ,
                        xy=(x[1] + (x[2]-x[1])/2, 
                            -2 * self.Y_BOX_HEIGTH * self.MAX_ENERGY_DISPLAY ),
                        fontsize= 12,
                        ha='center', va='top')
            
            x_0_item += x_len
        
        ax.set_xlim(             - self.X_BOX_MARGIN * x_len, 
                    x_len_global + self.X_BOX_MARGIN * x_len)
        ax.spines['top']   .set_visible(False)
        ax.spines['right'] .set_visible(False)
        ax.spines['left']  .set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        ax .xaxis.set_visible(False)
        ax .set_ylabel("Excitation Energy (MeV)")
        fig.suptitle(self.global_title, size=15)
        
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
    

    
    MAIN_FLD = '../DATA_RESULTS/SD_Kblocking_multiK/Mg_31'
    _levels = []
    for tail_ in (1, 3, 5):
        fld = MAIN_FLD + '/BU_folder_B1_MZ4_z12n19/' + f"{tail_}_0_PNPAMP_HWG/HWG/"
        _levels.append(fld)
    _levels.append(MAIN_FLD + '/BU_folder_B1_MZ4_z12n19/kmix_PNPAMP/HWG/')
    
    example_levels = [getAllLevelsAsString(fld) for fld in _levels]
    
    # _EnergyLevelSimpleGraph.MAX_NUM_OF_SIGMAS  = 10
    _EnergyLevelSimpleGraph.MAX_ENERGY_DISPLAY = 7.0
    _graph = _EnergyLevelSimpleGraph("HWG spectra from each K blocking and parity-mix")
    _graph.setData(example_levels[0], '$^{31}$Mg K=1/2')
    _graph.setData(example_levels[1], '$^{31}$Mg K=3/2')
    _graph.setData(example_levels[2], '$^{31}$Mg K=5/2')
    _graph.setData(example_levels[3], '$^{31}$Mg mix')
    _graph.plot(fld2saveWithFilename=MAIN_FLD+'/hwg-spectra_Mg31_usdb.pdf')
    
    