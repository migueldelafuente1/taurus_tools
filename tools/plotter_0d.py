'''
Created on Mar 13, 2023

@author: Miguel
'''
from tools.data import _DataObjectBase, OccupationNumberData
from tools.plotter_1d import PlotException, _taurus_object_test,\
    MATPLOTLIB_INSTALLED
import os
from tools.helpers import printf
if MATPLOTLIB_INSTALLED:
    import matplotlib.pyplot as plt
else:
    MATPLOTLIB_INSTALLED = False
    printf("WARNING :: Matplotlib not installed. Do not evaluate plot modules.")
    

class _Plotter0D(object):
    '''
    classdocs
    '''

    FOLDER_PATH  = 'DATA_RESULTS/'
    LATEX_FORMAT = False
    DTYPE : _DataObjectBase = None
    
    @classmethod
    def setFolderPath2Import(cls, folder_path):
        # check folder exists or smthg
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            cls.FOLDER_PATH = folder_path
        else:
            raise PlotException("Could not find export folder "
                                f"[{folder_path}] CWD: [{os.getcwd()}]")
    
    
    def setConstraintBase(self, constr_x : list):
        """
        set constraint of the files for plotting, could be a list of values:
        """
        if isinstance(constr_x, str):
            constr_x = [constr_x, ]
        
        self.constraints = []
        for val in constr_x:
            if val in _taurus_object_test.__dict__:
                self.constraints.append(val)
            else:
                printf(f"[PLT WARNING] variable [{val}] not a data "
                      f"attribute [{self.DYPE.__class__.__name__}]")
        
    def setTitle(self, title):
        self._title   = title
        
    def setXlabel(self, title):
        self._x_label = title
    
    def setYlabel(self, title):
        self._y_label = title
    
    def __init__(self, filenames, attr2plot=None):
        '''
        Constructor
        '''
        if not MATPLOTLIB_INSTALLED:
            printf("[PLT WARNING] Matplotlib not installed")
            return 
        
        self._title   = ''
        self._x_label = ''
        self._y_label = ''
        
        self.import_files = []
        self.export_figure = False
        self.export_figure_filename = None
        
        self.constraints   = []
        self.constraints_str  = {}
        self._executor_program = []
        self.attr2plot     = attr2plot if attr2plot else []
        self.attr2plot_str = {} # string to plot with the 
        self.import_files  = []
        self._results      = {}
        self._x_values     = {}
        self._legend_labels  = {}
        
        self._axes = []
        
        if isinstance(filenames, str):
            filenames = [filenames, ]
        # TODO: check if file is in folder path, raise comment if not
        self._getDataFromFiles(filenames)
        
        if self.__class__.__name__.startswith("_"):
            ## Interface implementation requirement
            raise PlotException("Abstract Class, define DType Instance")
    
    def _getDataFromFiles(self, filenames):
        
        raise PlotException("abstract Method, implement me!")
    
    def defaultPlot(self):
        """
        Plot the imported data
        """
        raise PlotException("abstract Method, implement me!")


class Plotter0D_OccupationNumbers(_Plotter0D):
    
    """
    Plotting of the occupation by setting an unfilled bar chart on the 
    shell states.
    For protons and neutrons, projected case will only show in case of projected
    column to be non zero.
    """
    
    def __init__(self, filenames, attr2plot=None):
        self._hasProjections = False
        self._results_proj   = {}
        self._results_unproj = {}
        
        _Plotter0D.__init__(self, filenames, attr2plot=attr2plot)        
    
    def defaultPlot(self):
        """
        Plot a bar chart of the occupation numbers.
        TODO: How to do it for several files.
        """
        if self.LATEX_FORMAT:
            plt.rcParams.update({"text.usetex": True,  "font.family": "Helvetica",
                                 "font.size": 20, "font.weight": 'bold'})
        else:
            plt.rcParams.update({"text.usetex": False, "font.family": 'DejaVu Sans',
                                 "font.size"  : 12}) 
        
        fig , axs = plt.subplots(nrows=1, ncols=2, sharey=True)
        self._axes = axs
        prot_sum, neut_sum = 0., 0.
        for file_ in self.import_files:
            labels_sorted = [x for x in  self._x_values[file_].keys()]
            tot_occ = [self._x_values[file_][k][2]+1 for k in labels_sorted]
            labels =  [str(x) for x in labels_sorted]
            sh_dim = len(labels)
            # Proton axis
            vals_p = [self._results_unproj[file_][-1][k] for k in labels_sorted]
            prot_sum = sum(vals_p)
            rem_p  = [tot_occ[i]-vals_p[i] for i in range(sh_dim)]
            axs[0].barh(labels, vals_p, label=f'occ. Unproj [{prot_sum:4.2f}]', 
                        color='red', height=0.5)
            axs[0].barh(labels, rem_p, left=vals_p,  
                        label='empty', color='lightblue', height=0.5)
            
            # Neutron axis
            vals_n = [self._results_unproj[file_][ 1][k] for k in labels_sorted]
            neut_sum = sum(vals_n)
            rem_n  = [tot_occ[i]-vals_n[i] for i in range(sh_dim)]
            axs[1].barh(labels, vals_n, label=f'occ. Unproj [{neut_sum:4.2f}]', color='blue', height=0.5)
            axs[1].barh(labels, rem_n, left=vals_n,  
                        label='empty', color='lightblue', height=0.5)
            
            if self._hasProjections:
                vals_p = [self._results_proj[file_][-1][k] for k in labels_sorted]
                axs[0].hbar(labels, vals_p, 
                            label='occ. Proj', color='crimson', height=0.5)
                
                vals_n = [self._results_proj[file_][ 1][k] for k in labels_sorted]
                axs[1].hbar(labels, vals_n, 
                            label='occ. Proj', color='indigo', height=0.5)
                
        axs[0].set_title("Protons")
        axs[1].set_title("Neutrons")
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        plt.show()
        _=0
    
    def _getDataFromFiles(self, filenames):
        
        for file_ in filenames:
            if os.path.exists(self.FOLDER_PATH+file_):
                self.import_files.append(file_)
            else:
                printf(f"[PLT WARNING] file_[{file_}] not found, Skipping.")
                continue
            
            # with open(self.FOLDER_PATH+file_, 'r') as f:
            #     data = f.readlines()
            data = OccupationNumberData(filename=self.FOLDER_PATH+file_)
            data.getResults()
            
            self._x_values[file_] = data.get_numbers
            self._hasProjections = data.hasProjectedOccupations
            unproj, proj = data.get_occupations
            
            self._results_unproj[file_] = unproj
            self._results_proj  [file_] = proj
            
            

if __name__ == '__main__':
    #===========================================================================
    # TODO: Plot OCCUPATION NUMBER as BAR PLOT
    #===========================================================================
    Plotter0D_OccupationNumbers.setFolderPath2Import('../DATA_RESULTS/')
    
    files_ = [
        # 'occupation_numbers_z2n1_2-dbase.dat',
        'occupation_numbers_z18n14_0-dbase.dat']
    
    plt_obj = Plotter0D_OccupationNumbers(files_)
    
    plt_obj.defaultPlot()
    
    #===========================================================================
    # TODO: (sky pie) Plot SPECTROSCOPIC SCHEMME
    #===========================================================================
    
    