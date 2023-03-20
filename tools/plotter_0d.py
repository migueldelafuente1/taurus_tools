'''
Created on Mar 13, 2023

@author: Miguel
'''
from tools.data import _DataObjectBase
from tools.plotter_1d import PlotException, _taurus_object_test,\
    MATPLOTLIB_INSTALLED
import os

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
                print(f"[PLT WARNING] variable [{val}] not a data "
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
            print("[PLT WARNING] Matplotlib not installed")
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


class Plotter0D_OccupationNumbers(_Plotter0D):
    pass



if __name__ == '__name__':
    #===========================================================================
    # TODO: Plot OCCUPATION NUMBER as BAR PLOT
    #===========================================================================
    
    #===========================================================================
    # Plot SPECTROSCOPIC SCHEMME
    #===========================================================================
    pass
    