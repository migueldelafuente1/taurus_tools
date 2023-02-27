'''
Created on Jan 27, 2023

@author: Miguel
'''
import os
import numpy as np
from tools.data import DataTaurus, DataAxial, _DataObjectBase, EigenbasisData
from tools.Enums import Enum
from tools.executors import _Base1DAxialExecutor
from tools.helpers import OUTPUT_HEADER_SEPARATOR
from copy import copy


MATPLOTLIB_INSTALLED   = True

try:
    import matplotlib.pyplot as plt
except ImportError as err:
    MATPLOTLIB_INSTALLED = False
    print("WARNING :: "+str(err)+" (i.e) Do not evaluate plot modules.")

_taurus_object_test = DataTaurus(0, 0, None, empty_data=True)
_axial_object_test  = DataAxial (0, 0, None, empty_data=True)

class PlotException(BaseException):
    pass

class _Ploter1D(object):
    '''
    classdocs
    '''
    FOLDER_PATH  = 'DATA_RESULTS/'
    LATEX_FORMAT = False
    DTYPE : _DataObjectBase = None
    
    @classmethod
    def setFolderPath2Import(cls, folder_path):
        # TODO: check folder exists or smthg
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
    
    
    
    def _getDataFromFiles(self, filenames):
        
        constrs, exe_progr = [], []
        for file_ in filenames:
            if os.path.exists(self.FOLDER_PATH+file_):
                self.import_files.append(file_)
            else:
                print(f"[PLT WARNING] file_[{file_}] not found, Skipping.")
                continue
            
            header_ = []
            data_results = []
            
            # with open(self.FOLDER_PATH+file_, 'w') as f:
            #     data = f.readlines()
            #     if not data[0].startswith('P_T'):
            #         data = [,] + data
            
            with open(self.FOLDER_PATH+file_, 'r') as f:
                data = f.readlines()
                
                constr_, dtypeClss_ = self._selectDataObjectAndMainConstraint(data)
                
                constrs  .append(constr_)
                exe_progr.append(dtypeClss_)
                if constr_ or dtypeClss_:
                    data = data[1:]
                
                _introduced_by_index = False
                if OUTPUT_HEADER_SEPARATOR in data[0]:
                    _introduced_by_index = True
                
                for line in data:
                    index_, result = line.split(OUTPUT_HEADER_SEPARATOR)
                    _i, val = index_.strip().split(":")
                    header_.append( (int(_i), float(val)) )
                    
                    ## How to difference between DataTaurus, DataAxial, ...
                    res = self._selectDataReaderByProgram(dtypeClss_, result)
                    # res = DataTaurus(0, 0, None, empty_data=True)
                    res.setDataFromCSVLine(result)
                    data_results.append(res)
                    
                    ## TODO: Verify if the index match with the attribute (??)
                
                self._results [file_] = data_results
                if _introduced_by_index:
                    self._x_values[file_] = dict(header_)
                self._legend_labels[file_] = constr_
                
        self.constraints = constrs
        self._executor_program = exe_progr
                
    def _selectDataObjectAndMainConstraint(self, data):
        """
        read the header from the 1D file, introduced by the constraint and dype: 
        :return: constr_, dataObjClass_
        """
        _0line = data[0].replace('\n','').strip()
        _0line = _0line.split(',')
        constr_, dataObjClass_ = None, None
        for val in _0line:
            val = val.strip()
            if val in _taurus_object_test.__dict__.keys():
                constr_ = val
            elif val in ('DataAxial', 'DataTaurus'):
                dataObjClass_ = val
        
        if not dataObjClass_:
            print("[PLT WARNING] Undefined DataObject to import, continue with [DataTaurus]")
            dataObjClass_ = 'DataTaurus'
        
        return constr_, dataObjClass_
    
    def _selectDataReaderByProgram(self, dataObjectClass_, result_raw_line=None):
        """
        all data results has a PROGRAM attribute. return the DataObject
        """
        if   dataObjectClass_ == 'DataAxial':
            return DataAxial(0, 0, None, empty_data=True)
        elif dataObjectClass_ == 'DataTaurus':
            return DataTaurus(0, 0, None, empty_data=True)
        elif dataObjectClass_ == 'EigenbasisData':
            raise PlotException("Undefined Process to import [EigenbasisData]")
            return EigenbasisData(None)
        else:
            raise PlotException("Undefined DataObject to import, "
                                f"raw_result_line={result_raw_line}")
    
    def defaultPlot(self, attr2plot=None):
        
        if attr2plot:
            self.attr2plot = attr2plot
            self.attr2plot_str = {attr2plot : self._getVariableStringForDisplay(attr2plot)}
            self._y_label = self.attr2plot_str[attr2plot]
        
        if self.LATEX_FORMAT:
            plt.rcParams.update({"text.usetex": True,  "font.family": "Helvetica",
                                 "font.size": 20, "font.weight": 'bold'})
        else:
            plt.rcParams.update({"text.usetex": False, "font.family": 'DejaVu Sans',
                                 "font.size"  : 12}) 
        
        self.constraints_str = {}
        for constr_ in self.constraints:
            self.constraints_str[constr_] = self._getVariableStringForDisplay(constr_)        
        
        fig , ax = plt.subplots()
        self._axes = ax
        
        for if_, file_ in enumerate(self.import_files):
            
            y_values = [getattr(val, self.attr2plot, None) for val in self._results[file_]]
            constr_  = self.constraints[if_]
            if len(self.constraints) == 1: # just set the program
                lab_ = self._executor_program[if_]
                if lab_ == "DataTaurus": 
                    lab_ = "Taurus"
            else:
                lab_ = self.constraints_str[constr_]
            self._axes.plot(self._x_values[file_].values(), y_values, '.-', label=lab_)
        
        if len(self.import_files) == 0:
            print("[WARNING] Not founded any file to plot, Exiting")
            return
        # Global Labels.
        if self._title   == '':
            lab_ = self.import_files[0].replace("export", '').replace(".txt", "")
            self._title = self._getVariableStringForDisplay(lab_.replace("_", " "))        
        if self._x_label == '':
            print("[PLT WARNINGN] Several constraints for X-axis, Suggestion: setXlabel()")
            lab_ = self.constraints_str[self.constraints[0]]
            self._x_label = lab_
        if self._y_label == '':
            self._y_label = self._getVariableStringForDisplay(self.attr2plot)
            
        self._axes.set_title (self._title)
        self._axes.set_xlabel(self._x_label)
        self._axes.set_ylabel(self._y_label)
        
        self._axes.legend()
        plt.tight_layout()
        plt.show()
    
    
    def setTitle(self, title):
        self._title   = title
        
    def setXlabel(self, title):
        self._x_label = title
    
    def setYlabel(self, title):
        self._y_label = title
    
    def setExportFigureFilename(self, output_filename):
        """ set output filename: formats ('.pdf', '.png', '.jpeg') """
        if not any([output_filename.endswith(ext_) for ext_ in ('.pdf', '.png', '.jpeg')]):
            print(f"[PLT WARNGING] invalid output file for figure: [{output_filename}]. Skip saving.")
            return
        
        self.export_figure = True
        self.export_figure_filename = output_filename
        self._axes.savefig(output_filename)
    
    def _getVariableStringForDisplay(self, var):
        """ 
        The string in a acceptable appearance (specially LaTeX) 
        """
        if not self.LATEX_FORMAT:
            return var
        
        new_var  = var
        if var == None:
            _ = 0
        args = var.split("_")
        args2 = copy(args) + ['', ] # omit 'tot' for the text to be less verbose
        ## 
        if   var in "zn":    new_var = var.upper()
        elif var == "MZmax": new_var = "M_Z^{{max}}"
        elif var == 'ho_b_length'   : new_var = "b^{SHO}\ fm"
        elif var == 'ho_hbaromega'  : new_var = "\\hbar\\omega^{SHO}\ MeV"
        elif var.endswith("dim")    : new_var = f"{args[0]}_{{{args[1]}}}"
        elif var.endswith("proton_numb")  : new_var = "\\langle{Z}\\rangle"
        elif var.endswith("neutron_numb") : new_var = "\\langle{N}\\rangle"
        elif var.startswith("var")  : new_var = f"\\sigma^2_{args[1]}"
        #
        elif var.startswith("kin")  : new_var = f"K_{{{args2[1]}}}\ MeV"
        elif var.startswith("hf")   : new_var = f"E^{{HF}}_{{{args2[1]}}}\ MeV"
        elif var.startswith("pair") : new_var = f"E^{{pair}}_{{{args2[1]}}}\ MeV"
        elif var.startswith("V_2B") : new_var = f"V^{{2b}}_{{{args2[2]}}}\ MeV"
        elif var.startswith("E_HFB"): new_var = f"E^{{HFB}}_{{{args2[2]}}}\ MeV"
        #
        elif var.startswith("beta") or var.startswith("gamma") : 
            new_var = f"\\{args[0]}^{{{args[1][:7]}}}"
        elif var.startswith("b") or var.startswith("q"):
            x = "\\beta" if var.startswith("b") else "Q"
            new_var = f"{x}_{{{args[0][1:]}}}^{{{args[1][:7]}}}"
        elif var.startswith("r_") : new_var = f"\\sqrt{{}}\\langle{{r^2}}\\rangle_{{({args[1][:7]})}}\ fm"
        elif var.startswith("J"):
            if var.endswith("_var"): new_var = f"\\sigma^2({{{args[0]}}})"
            else:                    new_var = f"\\langle{{{args[0]}^{{{args2[1]}}} }}\\rangle"
        elif var.startswith("P_T"):
            if   "_T00_" in var:
                x = args[2][1:].replace("p","\ M_J=+").replace("m", "\ M_J=-").replace("0", "\ M_J=0")
                new_var = f"\\delta^{{T=0}}_{{(J={x})}}"
            elif "_T1" in var:
                x = args[1][1:].replace("1p1","nn").replace("1m1", "pp").replace("10", "pn")
                new_var = f"\\delta^{{T=1,\ {x}}}" # _{{(J=0)}}
        #
        elif var == "iter_max":          new_var = "Iterations"
        elif var == "iter_time_seconds": new_var = "Total\ time(s)"
        elif var == "time_per_iter":     new_var = "Step\ lapse(s)"
        #
        else:
            ## pass the text with a capital and separate spaces
            var = var[0].upper() + var[1:]
            new_var = var.replace(" ", "\\ ") 
        
        new_var = r'$${}$$'.format(new_var)
        return new_var
            
        

class Ploter1D_Taurus(_Ploter1D):
    
    DTYPE = DataTaurus



if __name__ == "__main__":
    
    
    # _Ploter1D.setFolderPath2Import('../DATA_RESULTS/')
    #
    # files_ = ['export_TESb20_z12n10_hamil_MZ4.txt',]
    #
    # attr2plot = 'E_HFB'
    # # attr2plot = 'gamma_isoscalar'
    # # attr2plot = 'gamma_isovector'
    # # attr2plot = "b22_isoscalar"
    # # attr2plot = "b20_isoscalar"
    # # attr2plot = "pair_pn"
    # plt_obj = Ploter1D_Taurus(files_, attr2plot)
    # plt_obj.LATEX_FORMAT = True
    # # for attr_ in _taurus_object_test.__dict__.keys():
    # #     if attr_.startswith("_"): continue
    # #     print(f"{attr_:20} :: ", plt_obj._getVariableStringForDisplay(attr_))
    #
    #
    # for attr2plot in ('E_HFB', ):
    #     plt_obj.setConstraintBase('b20_isoscalar')
    #     plt_obj.setTitle(r"$$TES\ D1S\ MZ=4\qquad z,n=(12,10)$$")
    #     plt_obj.defaultPlot(attr2plot)
    #     # plt_obj.LATEX_FORMAT = False
    # _taurus_object_test.E_HFB
    
    
    
    #===========================================================================
    # PLOT the P_T surfaces
    #===========================================================================
    # SUBFLD_ = 'Mg_MZ4/'
    # SUBFLD_ = 'Mg_MZ5/'
    SUBFLD_ = 'SDnuclei_MZ4_new/'
    # SUBFLD_ = 'SDnuclei_MZ5/'
    Ploter1D_Taurus.setFolderPath2Import('../DATA_RESULTS/PN_mixing/'+SUBFLD_)
    
    nuclei = [(z, z) for z in range(8, 21, 2)]
    pair_constr = ['P_T00_J10', 'P_T1p1_J00', 'P_T1m1_J00',  'P_T10_J00']
    nuclei = [(16, 17),]
    
    for z, n in nuclei:
        if SUBFLD_.startswith('Mg' ):
            if z != 12: continue
        # files_ = [f"export_PSz{z}n{n}_D1S_{pp.replace('_', '')}.txt" for pp in pair_constr]
        files_ = [f"export_TES_{pp}_z{z}n{n}_hamil_MZ4.txt" for pp in pair_constr]
        
        attr2plot_list = [
            'E_HFB', 'pair', *pair_constr]
        
        plt_obj = Ploter1D_Taurus(files_)
        plt_obj.LATEX_FORMAT = True
        
        for attr2plot in attr2plot_list:
            plt_obj.setXlabel("Pair Constr. value")
            plt_obj.defaultPlot(attr2plot)
    
    