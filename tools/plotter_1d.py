'''
Created on Jan 27, 2023

@author: Miguel
'''
import os
import numpy as np
from tools.data import DataTaurus, DataAxial, _DataObjectBase, EigenbasisData,\
    OccupationNumberData, DataAttributeHandler
from tools.Enums import Enum
from tools.executors import _Base1DAxialExecutor
from tools.helpers import OUTPUT_HEADER_SEPARATOR
from copy import copy
import zipfile
import shutil
from tools.inputs import InputTaurus


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

class _PlotterBase(object):
    
    '''
    classdocs
    '''
    FOLDER_PATH  = 'DATA_RESULTS/'
    LATEX_FORMAT = False
    DTYPE : _DataObjectBase = None
    EXPORT_PDF_AND_MERGE = False
    
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
        self.minimum_result= {} # Data-result for the minimum (index 0) by file
        self._x_values     = {}
        self._legend_labels  = {}
        
        self._axes = []
        self._figs = []
        self._figures_titles_pdf = []
        
        if isinstance(filenames, str):
            filenames = [filenames, ]
        # check if file is in folder path, raise comment if not
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
                                    
                self._results [file_] = data_results
                if _introduced_by_index:
                    self._x_values[file_] = dict(header_)
                self._legend_labels[file_] = constr_
                
        self.constraints = constrs
        self._executor_program = exe_progr
        self._setMinimumAndSetXValuesForTheResults()
    
    def _setMinimumAndSetXValuesForTheResults(self):
        """ 
        Set the minumum result for the E_HFB
        In case there are no indexing for the results, also define x_label values
        """
        for file_ in self.import_files:
            ## Put one for comparison
            self.minimum_result[file_] = self._results[file_][0]
            ind_0 = 0
            ## In case there are no indexing, set the x values and index
            if not self._x_values.get(file_, None):
                self._findEnergyMinumumResultInResultsList(file_)
            else:
                aux_keys = list(self._x_values[file_].keys())
                if 0 in aux_keys: # there is a minimum (indx < 0 and > 0) 
                    ind_0 = aux_keys.index(0)
                    self.minimum_result[file_] = self._results[file_][ind_0]
                else: # indexes were not shorted, get 1
                    self._findEnergyMinumumResultInResultsList(file_)
    
    def _findEnergyMinumumResultInResultsList(self, file_):
        """ Search in the result list the state of the lowest energy."""
        ind_0 = 0
        for ind_, res in enumerate(self._results[file_]):
            if self.minimum_result[file_].E_HFB > res.E_HFB:
                self.minimum_result[file_] = res
                ind_0 = ind_
        ## repeat the loop to set the index of the files
        self._x_values[file_] = {}
        for ind_, res in enumerate(self._results[file_]):
            self._x_values[file_][ind_-ind_0] = getattr(res, 
                                                        self.constraints[0])
    
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
    
    def _getListOfDataNonProperlyFinished(self, file_, y_values=None):
        """ 
        Get a list to scatter for the states that could not be converged 
        Use select the index to select the y_values to present.!!
        """
        x_npf, y_npf = [], []
        
        ppfin = ( x.properly_finished for x in self._results[file_])
        x_npf = self._x_values[file_].values()
        
        if y_values == None:
            y_npf = []
            for res in self._results[file_]:
                if isinstance(self.attr2plot, DataAttributeHandler):
                    y_npf.append(self.attr2plot.getValue(res))
                else:
                    y_npf.append(getattr(res, self.attr2plot))
        else:
            y_npf = (val for val in y_values)
        
        non_conv = list(filter(lambda x: not x[2], zip(x_npf, y_npf, ppfin)))
        if len(non_conv) == 0:
            return [], []

        x_npf, y_npf, _ = zip(*non_conv)
        x_npf, y_npf = list(x_npf), list(y_npf)
        return x_npf, y_npf   
    
    def defaultPlot(self, attr2plot=None, show_plot=True):
        """ General automated definition for the imported data, 
        select the attribute to plot.
        TODO: modify to import functions of several attributes from the data
        
        :show_plot <bool> is to prompt the images or not (in case of several attributes)"""
        
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
        self._figs = fig
        
        for if_, file_ in enumerate(self.import_files):
            
            y_values = []
            for res in self._results[file_]:
                if isinstance(self.attr2plot, DataAttributeHandler):
                    y_values.append(self.attr2plot.getValue(res))
                else:
                    y_values.append(getattr(res, self.attr2plot))
            
            constr_  = self.constraints[if_]
            if len(self.constraints) == 1: # just set the program
                lab_ = self._executor_program[if_]
                if lab_ == "DataTaurus": 
                    lab_ = "Taurus"
            else:
                lab_ = self.constraints_str[constr_]
            self._axes.plot(self._x_values[file_].values(), y_values, '.-', label=lab_)
            
            ## Print Un-converged results marked different
            x_npf, y_npf = self._getListOfDataNonProperlyFinished(file_)
            if len(y_npf)>0:
                self._axes.scatter(x_npf, y_npf, marker='X', c='k', label='non converged',
                                   zorder=2.5, )
        
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
        if show_plot:
            plt.show()
        
        if self.EXPORT_PDF_AND_MERGE:
            exp_file = f'{self.FOLDER_PATH}{self.attr2plot}.pdf'
            self._figures_titles_pdf.append(exp_file)
            self._axes.savefig(self._figures_titles_pdf[-1])
    
    def mergeFiguresInto1PDF(self, *pdf_title):
        
        if not self.EXPORT_PDF_AND_MERGE:
            print("[WARNING] To merge pdfs activate EXPORT_PDF_AND_MERGE before"
                  " calling self.defaultPlot() (Skipping)")
            return
                
        from PyPDF2 import PdfMerger
        merger = PdfMerger()
        for fn_pdf in self._figures_titles_pdf:
            merger.append(fn_pdf)
        aux = '-'.join([str(x) for x in pdf_title])
        if aux != '': aux = f"({aux})"
        pdf_title = f"{self.FOLDER_PATH}results_plots{aux}.pdf"
        merger.write(pdf_title)
        merger.close()
        
        # for fn_pdf in self._figures_titles_pdf:
        #     os.remove(fn_pdf) ## Permission errors
        print(f"   [DONE] Exporting merged PDF into [{pdf_title}]")
    
    
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
        
        if self.EXPORT_PDF_AND_MERGE:
            self._figures_titles_pdf.append(output_filename)
        
        self.export_figure = True
        self.export_figure_filename = output_filename
        self._figs.savefig(output_filename)
    
    def _getVariableStringForDisplay(self, var):
        """ 
        The string in a acceptable appearance (specially LaTeX) 
        """
        var = str(var)
        if not self.LATEX_FORMAT:
            return var
        
        new_var  = var
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
    

class _Plotter1D(_PlotterBase):
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
                      f"attribute [{self.DTYPE.__class__.__name__}]")
        
    
    def __init__(self, filenames, attr2plot=None, importNow=True):
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
        self.attr2plot_str = {}  # string to plot with the 
        self.import_files  = []
        self._results      = {}  ## {file(constraint) : [DataResults, ( in order)], }
        self.minimum_result= {} # Data-result for the minimum (index 0) by file
        self._x_values     = {}  ## {file(constraint) : { index_ : value, }, }
        self._legend_labels  = {}
        
        self._axes = []
        self._figs = []
        self._figures_titles_pdf = []
        
        if isinstance(filenames, str):
            filenames = [filenames, ]
        # TODO: check if file is in folder path, raise comment if not
        if importNow:
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
                
                for li_, line in enumerate(data):
                    if _introduced_by_index:
                        index_, result = line.split(OUTPUT_HEADER_SEPARATOR)
                        _i, val = index_.strip().split(":")
                    else:
                        result = line
                        _i  = li_
                        val = None
                    
                    ## How to difference between DataTaurus, DataAxial, ...
                    res = self._selectDataReaderByProgram(dtypeClss_, result)
                    # res = DataTaurus(0, 0, None, empty_data=True)
                    res.setDataFromCSVLine(result)
                    data_results.append(res)
                    
                    if val == None:
                        val = 0.0 if not constr_ else getattr(res, constr_, 0.0)
                    header_.append( (int(_i), float(val)) )
                                    
                self._results [file_] = data_results
                if _introduced_by_index:
                    self._x_values[file_] = dict(header_)
                self._legend_labels[file_] = constr_
        
        if not any(self.constraints):
            self.constraints = constrs
        self._executor_program = exe_progr
        self._setMinimumAndSetXValuesForTheResults()
                
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
    
    def _set_axisLimits(self, y_values):
        """ Select some ranges for the y axis """
        return
        if self.DTYPE == DataTaurus:
            max_, min_ = max(y_values), abs(min(y_values))
            if self.attr2plot.startswith('P_T'):
                min_ = 1.0 if min_ < 1 else min_
                max_ = 1.0 if max_ < 1 else max_
                self._axes.set_ylim([-min_, max_])
            elif self.attr2plot.startswith('var_'):
                max_ = 1.0 if max_ < 1 else max_
                self._axes.set_ylim([-0.01, max_])
            else:
                if any(abs(y) > 1.e-4 for y in y_values):
                    self._axes.relim()
                    self._ylim_tops = self._axes.get_ylim()
                else:
                    if hasattr(self, '_ylim_tops'):
                        self._axes.set_ylim(self._ylim_tops)
                    else:
                        self._axes.set_ylim([ -0.01, 1.0])
    
    def defaultPlot(self, attr2plot=None, show_plot=True):
        
        if hasattr(self, '_ylim_tops'): delattr(self, '_ylim_tops')
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
        self._figs = fig
        
        for if_, file_ in enumerate(self.import_files):
            
            y_values = []
            for res in self._results[file_]:
                if isinstance(self.attr2plot, DataAttributeHandler):
                    y_values.append(self.attr2plot.getValue(res))
                else:
                    y_values.append(getattr(res, self.attr2plot))
            
            constr_  = self.constraints[if_]
            if len(self.constraints) == 1: # just set the program
                lab_ = self._executor_program[if_]
                if lab_ == "DataTaurus": 
                    lab_ = "Taurus"
            else:
                lab_ = self.constraints_str[constr_]
            self._axes.plot(self._x_values[file_].values(), y_values, '.-', label=lab_)
            self._set_axisLimits(y_values)
            
            ## Print Un-converged results marked different
            x_npf, y_npf = self._getListOfDataNonProperlyFinished(file_)
            if len(y_npf)>0:
                self._axes.scatter(x_npf, y_npf, marker='X', c='k', #label='non converged',
                                   zorder=2.5, )
            x0_indx = abs(min( list(self._x_values[file_].keys()) ))
            self._axes.scatter(self._x_values[file_][0], y_values[x0_indx], marker='o')
        
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
        if show_plot:
            plt.show()
        
        if self.EXPORT_PDF_AND_MERGE:
            exp_file = f'{self.FOLDER_PATH}{self.attr2plot}.pdf'
            self._figures_titles_pdf.append(exp_file)
            fig.savefig(self._figures_titles_pdf[-1])
            print(f"   [DONE] Image for attr [{self.attr2plot}] saved in [{exp_file}]")
        

class Plotter1D_Taurus(_Plotter1D):
    
    DTYPE = DataTaurus

class Plotter1D_Axial(_Plotter1D):
    
    DTYPE = DataAxial

class Plotter1D_CanonicalBasis(_Plotter1D):
    
    DTYPE = EigenbasisData
    
    ## TODO: Modify the importing files to identify each deformation of the
    ## with each occupation or state in the deform map:
    ## ** _getDataFromFiles
    ## ** _getVariableForDisplay
    class AttrEigenbasisEnum(Enum):
        fermi_energ_prot = 'fermi_energ_prot'
        fermi_energ_neut = 'fermi_energ_neut'
        avg_proton  = 'avg_proton'
        avg_neutron = 'avg_neutron'
        avg_n = 'avg_n'
        avg_l = 'avg_l'
        avg_parity = 'avg_parity'
        avg_j  = 'avg_j'
        avg_jz = 'avg_jz'
        v2  = 'v2'
        h   = 'h'
        H11 = 'H11'
        
    __INVALID_ATTR2PLOT_EIGENB_H = [
        AttrEigenbasisEnum.H11,
        AttrEigenbasisEnum.v2
        ]
    __INVALID_ATTR2PLOT_CANONICALBASIS = [
        AttrEigenbasisEnum.H11, 
        ]
    __INVALID_ATTR2PLOT_EIGENB_H11 = [
        AttrEigenbasisEnum.h, 
        AttrEigenbasisEnum.v2,
        AttrEigenbasisEnum.fermi_energ_neut, 
        AttrEigenbasisEnum.fermi_energ_prot
        ]
        
    def __init__(self, filenames,  global_folder=None, attr2plot=None):
        """
        :filenames: export_file
            * <list>: only the filenames, the zip will be imported from the default
                pattern for zip/BU naming
            * <dict>: put the folder/zip for each constraint (several constraints to print)
                {export_file_x.txt: folder / file_x.zip, }
        :global_folder: string to set one folder/zip to import (single constraint 2print)
        
        """
        self._folders_for_filenames = {}
        self._automatic_naming_folders = True  ## Name of the folder from default zip/BU from filenames
        
        self._eigenbasis_h_dats   = {}
        self._eigenbasis_H11_dats = {}
        self._canonicalbasis_dats = {}
             
        ## TODO: Different plot (set up from Plotter1D_OccupationNumbers i.e.)
        # self._occupation_numbers_dats = {}
        self._has_eigen_h    = False
        self._has_eigen_H11  = False
        self._has_canon_bas  = False
        # self._has_occ_number = False
        
        self._zn_header = ''
        self._sp_dim_by_file = {}
        
        if global_folder:
            if isinstance(filenames, list):
                assert len(filenames)==1, "If global_folder give only one file(constraint)"
                filenames = filenames[0]
            self._folders_for_filenames = {filenames: global_folder, }
            self._automatic_naming_folders = False
        elif isinstance(filenames, dict):
            self._folders_for_filenames = filenames
            self._automatic_naming_folders = False
        
        self._reset_labelsTitle = {
            '_x_label': False, '_y_label': False, '_title': False}
        
        return _Plotter1D.__init__(self, filenames, attr2plot=attr2plot)
    
    def getFolderToImportDATfiles(self, file_):
        """ 
        Required to call the folder or zip file containing the dat files 
        """
        ## export_TES_{constr}_{zn}_{interaction}
        file_ = file_.replace('export_TES_', '')
        if file_.endswith('.txt'): 
            file_ = file_.replace('.txt', '')
        
        constr_, interaction_ = [], []
        zn = None
        for arg in file_.split('_'):
            if zn == None:
                if 'z' in arg and 'n' in arg:
                    zn = arg
                else:
                    constr_.append(arg)
            else:
                interaction_.append(arg)
        constr_ = '_'.join(constr_)
        interaction_ = '_'.join(interaction_)
        self._zn_header = zn
        
        fold_def  = f'BU_folder_{interaction_}_{zn}'
        fold_path = self.FOLDER_PATH + fold_def
        zip_def   = f'BU_{constr_}_{interaction_}-{zn}.zip'
        zip_path  = self.FOLDER_PATH + zip_def
        
        if os.path.exists(fold_path):
            assert os.path.isdir(fold_path), f"[{fold_path}] Must be folder!"
            print("[WARNING] Option folder chosen, so all the files must be there")
            return fold_path
        if os.path.exists(zip_path):
            ## decompress a zip file.
            files_inZip = []
            with zipfile.ZipFile(zip_path, 'r') as zipobj:
                members = zipobj.namelist()
                files_inZip = [x.replace(f'{fold_def}/', '') for x in members]
                zipobj.extractall(members=members, path=fold_path)
            # if fold_def.replace(self.FOLDER_PATH, '') in fold_def:
            aux_fold_path = fold_def+'/'+fold_def
            for f_ in files_inZip[1:]: #os.listdir():
                shutil.move('/'.join([fold_path, fold_def, f_]), fold_path)
            return fold_path
        else:
            raise PlotException("Invalid path to import the results")
    
    
    def _getDataFromFiles(self, filenames):
        """ Extract the .dat data, called from the __init__ of plotter1D"""
        
        _Plotter1D._getDataFromFiles(self, filenames)
        
        ## Now we have the deformations list and indexes, proceed to read the
        ## folder with the data files.
        EI_HBA = DataTaurus.DatFileExportEnum.eigenbasis_h
        EI_H11 = DataTaurus.DatFileExportEnum.eigenbasis_H11
        EI_CBA = DataTaurus.DatFileExportEnum.canonicalbasis 
        # OCCNUM = DataTaurus.DatFileExportEnum.occupation_numbers 
        
        if self._automatic_naming_folders:
            
            for export_file in self.import_files:
                
                self._sp_dim_by_file[export_file] = 2 * int(self._results[export_file][0].sp_dim)
                
                fold_ = self.getFolderToImportDATfiles(export_file)
                
                self._eigenbasis_h_dats  [export_file] = []
                self._eigenbasis_H11_dats[export_file] = []
                self._canonicalbasis_dats[export_file] = []
                # self._occupation_numbers_dats[export_file] = []
                
                all_f = os.listdir(fold_)
                list_eigenh = list(filter(lambda x: x.startswith(EI_HBA), all_f))
                list_eigh11 = list(filter(lambda x: x.startswith(EI_H11), all_f))
                list_canbas = list(filter(lambda x: x.startswith(EI_CBA), all_f))
                # list_occnum = list(filter(lambda x: x.startswith(OCCNUM), all_f))
                n_dbase = list(filter(lambda x: x.endswith('dbase.bin'), all_f))
                for indx_ in self._x_values[export_file].keys():
                    
                    if indx_ == 0:
                        ## TODO: Base cases (select the correct seed!) 0-
                        aux = '0-' if len(n_dbase)>1 else ''
                        filedat = '_{}_{}dbase.dat'.format(self._zn_header, aux)
                    else:
                        filedat = '_{}_d{}.dat'.format(self._zn_header, indx_)
                    
                    if list_eigenh and (EI_HBA+filedat in list_eigenh):
                        self._has_eigen_h = True
                        res = EigenbasisData(fold_+'/'+EI_HBA+filedat)
                        res.getResults()
                        self._eigenbasis_h_dats[export_file].append(res)
                        
                    if list_eigh11 and (EI_H11+filedat in list_eigh11):
                        self._has_eigen_H11 = True
                        res = EigenbasisData(fold_+'/'+EI_H11+filedat)
                        res.getResults()
                        self._eigenbasis_H11_dats[export_file].append(res)
                        
                    if list_canbas and (EI_CBA+filedat in list_canbas):
                        self._has_canon_bas = True
                        res = EigenbasisData(fold_+'/'+EI_CBA+filedat)
                        res.getResults()
                        self._canonicalbasis_dats[export_file].append(res)
                    ## occupation numbers has different object
                    # if list_occnum and (OCCNUM+filedat in list_occnum):
                    #     res = OccupationNumberData(fold_+'/'+OCCNUM+filedat)
                    #     res.getResults()
                    #     self._occupation_numbers_dats[export_file].append(res)
        #
    
    def _getVariableEigenStringForDisplay(self, var):
        
        if not self.LATEX_FORMAT:
            return var

        new_var  = var
        args = var.split("_")
        args2 = copy(args) + ['', ] # omit 'tot' for the text to be less verbose
        ## 
        
        if var in (self.AttrEigenbasisEnum.fermi_energ_neut, 
                   self.AttrEigenbasisEnum.fermi_energ_prot):
            new_var = f"E^{{Fermi}}_{{{args[2]}}}"
        elif var.startswith("avg_"):
            if var == self.AttrEigenbasisEnum.avg_parity:
                args[1] = "\\Pi"
            new_var = f"\\langle{{ {args[1]} }}\\rangle_{{sp}}"
        elif var == self.AttrEigenbasisEnum.v2:
            new_var = "v^2_{sp}"
        elif var == self.AttrEigenbasisEnum.h:
            new_var = "h_{sp}"
        elif var == self.AttrEigenbasisEnum.H11:
            new_var = "H^{11}_{sp}"
        else:
            ## pass the text with a capital and separate spaces
            var = var[0].upper() + var[1:]
            new_var = var.replace(" ", "\\ ") 
        
        new_var = r'$${}$$'.format(new_var)
        return new_var
          
    
    def __plotSurfaces(self, attr2plot, index_2_print, datfile, show_plot=True):
        ## process the surfaces by index for any attribute but fermi energies
        ## TODO: in case of attr2plot = h, plot also the fermi_ energy
        ## "index_2_print" is a set of index to be printed
        data = getattr(self, '_{}_dats'.format(datfile), [])
        # otherwise the default titles wont change between dat files
        for attr_ in self._reset_labelsTitle.keys():
            self._reset_labelsTitle[attr_] = getattr(self, attr_, '') == ''
        
        for if_, file_export in enumerate(self.import_files):
            
            if index_2_print:
                index_lims = index_2_print
            else:
                index_lims = [l+1 for l in range(self._sp_dim_by_file[file_export])]
            ## 1. Sort the data by the indexes of the eigenstates_
            # labels_  = {}
            y_by_lab = {}
            
            for i_x, res_x in enumerate(data[file_export]):
                if attr2plot.startswith('fermi_energ'):
                    lab_ind = attr2plot[13:]
                    value = getattr(res_x, attr2plot) 
                    if i_x == 0:
                        y_by_lab[lab_ind] = [value, ]
                    else:
                        y_by_lab[lab_ind].append(value)
                else:
                    for i_lab, value in enumerate(getattr(res_x, attr2plot)):
                        lab_ind = res_x.index[i_lab]
                        if i_x == 0:
                            y_by_lab[lab_ind] = [value, ]
                        else:
                            y_by_lab[lab_ind].append(value)                
            
            ## 2. Plot surfaces, (select the important labels, there are hundreds)
            fig , ax = plt.subplots()
            self._axes = ax
            
            for lab_ind, y_values in y_by_lab.items():
                if (not attr2plot.startswith('fermi_energ')) and lab_ind not in index_lims: 
                    continue
                self._axes.plot(self._x_values[file_export].values(), y_values, 
                                '.-', label=lab_ind)
                ## Print Un-converged results marked different
                x_npf, y_npf = self._getDataNonProperlyFinished(file_export, 
                                                                y_values)
                if len(y_npf)>0:
                    self._axes.scatter(x_npf, y_npf, marker='X', c='k', label='non converged',
                                        zorder=2.5, )
            
            if len(self.import_files) == 0:
                print("[WARNING] Not founded any file to plot, Exiting")
                return
            # Global Labels.
            if self._title   == '':
                lab_ = self.import_files[0].replace("export_TES_", '').replace(".txt", "")
                lab_ = "{}  {}".format(datfile, lab_)
                self._title = self._getVariableEigenStringForDisplay(lab_.replace("_", " "))        
            if self._x_label == '':
                print("[PLT WARNINGN] Several constraints for X-axis, Suggestion: setXlabel()")
                lab_ = self.constraints_str[self.constraints[0]]
                self._x_label = lab_
            if self._y_label == '':
                self._y_label = self._getVariableEigenStringForDisplay(attr2plot)
                
            self._axes.set_title (self._title)
            self._axes.set_xlabel(self._x_label)
            self._axes.set_ylabel(self._y_label)
            
            self._axes.legend()
            plt.tight_layout()
            if show_plot:
                plt.show()
            
            ## return undefined labels to its null value
            for attr_, reset_ in self._reset_labelsTitle.items():
                if reset_: setattr(self, attr_, '') 
                    
        
    
    def defaultPlot(self, attr2plot=None, index_2_print=None,
                    attr2plotExport=None, show_plot=True):
        """ 
        If attr2plot (single or list) will print that results before the
        single particle energies/occupations 
        """
        # otherwise the default titles wont change between dat files
        for attr_ in self._reset_labelsTitle.keys():
            self._reset_labelsTitle[attr_] = getattr(self, attr_, '') == ''
        
        ## If attr2plot, show results from the export_file
        if attr2plotExport != None:
            if isinstance(attr2plotExport, list):
                for attr_ in attr2plotExport:
                    _Plotter1D.defaultPlot(self, attr_, False)
            else:
                _Plotter1D.defaultPlot(self, attr2plotExport, False)
        
        ## return undefined labels to its null value
        for attr_, reset_ in self._reset_labelsTitle.items():
            if reset_: setattr(self, attr_, '') 
        self.attr2plot = None
        self.attr2plot_str = {}
        
        ## Plot the values from the canonical basis or the occupation number
        ## along the constraint deformation
        if attr2plot == None: return
        
        if index_2_print:
            index_2_print = [l for l in range(index_2_print[0], index_2_print[1]+1)]
        
        if self._has_canon_bas:
            if attr2plot not in self.__INVALID_ATTR2PLOT_CANONICALBASIS:
                self.__plotSurfaces(attr2plot, index_2_print,
                                    DataTaurus.DatFileExportEnum.canonicalbasis,
                                    show_plot)
        if self._has_eigen_h:
            if attr2plot not in self.__INVALID_ATTR2PLOT_EIGENB_H:
                self.__plotSurfaces(attr2plot, index_2_print,
                                    DataTaurus.DatFileExportEnum.eigenbasis_h,
                                    show_plot)
        if self._has_eigen_H11:
            if attr2plot not in self.__INVALID_ATTR2PLOT_EIGENB_H11:
                self.__plotSurfaces(attr2plot, index_2_print,
                                    DataTaurus.DatFileExportEnum.eigenbasis_H11,
                                    show_plot)
        
    
    
class Plotter1D_OccupationNumbers(Plotter1D_CanonicalBasis):
    
    """
    TODO:
    Crear una imagen en la que se identifica la ocupacion relativa del estado SHO 
    siendo el eje X la constraint evaluada. (Colormap) y el Y donde ponemos la
    "label" del estado
    
    """
    class AttrOccupationNumbEnum(Enum):
        projected   = 'projected'
        unprojected = 'unprojected'
        
        
    def __init__(self, filenames,  global_folder=None, attr2plot=None):
        """
        :filenames: export_file
            * <list>: only the filenames, the zip will be imported from the default
                pattern for zip/BU naming
            * <dict>: put the folder/zip for each constraint (several constraints to print)
                {export_file_x.txt: folder / file_x.zip, }
        :global_folder: string to set one folder/zip to import (single constraint 2print)
        
        """
        self._folders_for_filenames = {}
        self._automatic_naming_folders = True  ## Name of the folder from default zip/BU from filenames
        
        self._canonicalbasis_dats = {}
             
        ## TODO: Different plot (set up from Plotter1D_OccupationNumbers i.e.)
        self._occupation_numbers_dats = {}
        self._has_occ_number = False
        
        self._zn_header = ''
        
        if global_folder:
            if isinstance(filenames, list):
                assert len(filenames)==1, "If global_folder give only one file(constraint)"
                filenames = filenames[0]
            self._folders_for_filenames = {filenames: global_folder, }
            self._automatic_naming_folders = False
        elif isinstance(filenames, dict):
            self._folders_for_filenames = filenames
            self._automatic_naming_folders = False
        
        return _Plotter1D.__init__(self, filenames, attr2plot=attr2plot)
    
    def _getDataFromFiles(self, filenames):
        """ Extract the .dat data, called from the __init__ of plotter1D"""
        
        _Plotter1D._getDataFromFiles(self, filenames)
        
        ## Now we have the deformations list and indexes, proceed to read the
        ## folder with the data files.
        OCCNUM = DataTaurus.DatFileExportEnum.occupation_numbers 
        
        if self._automatic_naming_folders:
            
            for export_file in self.import_files:
                
                fold_ = self.getFolderToImportDATfiles(export_file)
                
                self._occupation_numbers_dats[export_file] = []
                
                all_f = os.listdir(fold_)
                list_occnum = list(filter(lambda x: x.startswith(OCCNUM), all_f))
                
                for indx_ in self._x_values[export_file].keys():
                    
                    if indx_ == 0:
                        ## TODO: Base cases (select the correct seed!)
                        filedat = '_{}_0-dbase.dat'.format(self._zn_header)
                    else:
                        filedat = '_{}_d{}.dat'.format(self._zn_header, indx_)
                    
                    ## occupation numbers has different object
                    if list_occnum and (OCCNUM+filedat in list_occnum):
                        res = OccupationNumberData(fold_+'/'+OCCNUM+filedat)
                        res.getResults()
                        self._occupation_numbers_dats[export_file].append(res)
        #

if __name__ == "__main__":
    
    
    #===========================================================================
    # PLOT OF DEFORMATION SURFACES
    #===========================================================================
    
    _Plotter1D.setFolderPath2Import('../DATA_RESULTS/')
    
    nuclei = [
        # (10,11), 
        (12,10), #(12,12),
        ]
    for z, n in nuclei:
        files_ = [f'export_TESb20_z{z}n{n}_hamil_MZ4.txt',
                  f'export_TESb20_z12n12_hamil_MZ4.txt']
    
        plt_obj = Plotter1D_Taurus(files_)
        plt_obj.LATEX_FORMAT = True
    
        E_J0 = plt_obj.minimum_result[files_[0]].E_HFB
        inertia_moment = DataAttributeHandler(
            lambda jx, erot: 0.5*(jx*(jx+1)) / (erot - E_J0 + 1.e-3), 'Jx', 'E_HFB')
        inertia_moment.setName("Mom inertia Jx")
    
        attr2plot_list = [
            'E_HFB', 
            inertia_moment
            ]
    
        # plt_obj.setConstraintBase('b20_isoscalar')
        # plt_obj.setTitle(r"TES\ D1S\ MZ=4\qquad z,n=(12,10)")
    
        for attr2plot in attr2plot_list:
            plt_obj.setXlabel("beta {20}")
            plt_obj.defaultPlot(attr2plot, show_plot=attr2plot==attr2plot_list[-1])
        
        _=0
    
    #===========================================================================
    # PLOT the P_T surfaces
    #===========================================================================
    
    # SUBFLD_ = 'Mg_MZ4/'
    # # SUBFLD_ = 'Mg_MZ5/'
    # # SUBFLD_ = 'SDnuclei_MZ4_new/z17n17_seed5/'
    # SUBFLD_ = 'SDnuclei_MZ4_new/'
    # # SUBFLD_ = 'SDnuclei_MZ5/'
    # SUBFLD_ = 'Mg_MZ4_cranked/Jeq4/'
    #
    # Plotter1D_Taurus.setFolderPath2Import('../DATA_RESULTS/PN_mixing/'+SUBFLD_)
    #
    # nuclei = [(12, z) for z in range(6, 31, 2)]
    # pair_constr = ['P_T00_J10', 'P_T1p1_J00', 'P_T1m1_J00',  'P_T10_J00']
    # nuclei = [
    #     (10,10), (12,12),
    #     # (8, 9), 
    #     # (10,11), 
    #     # (12,13),
    #     # (16,17),
    #     # (17,17),
    #     ]
    # pairsT1_opp = DataAttributeHandler(
    #     lambda mT0, mTp1, mTm1: (mT0**2) - 2*(mTp1*mTm1),
    #     'P_T10_J00', 'P_T1p1_J00', 'P_T1m1_J00')
    # pairsT1_opp.setName('mod(delta^{T=1,J=0})')
    #
    # pairsT0_opp = DataAttributeHandler(
    #     lambda mJ0, mJp1, mJm1: (mJ0**2) - 2*(mJp1*mJm1),
    #     'P_T00_J10', 'P_T00_J1p1', 'P_T00_J1m1')
    # pairsT0_opp.setName('mod(delta^{T=0,J=1})')
    #
    # inertia_moment = DataAttributeHandler(
    #     lambda jx, erot: 0.5*(jx*(jx+1)) / erot, 
    #     'Jx', 'E_HFB')
    # inertia_moment.setName("Mom inertia Jx")
    #
    # Plotter1D_Taurus.EXPORT_PDF_AND_MERGE = True
    # for z, n in nuclei:
    #     # if SUBFLD_.startswith('Mg' ):
    #     #     if z != 12: continue
    #     files_ = [f"export_PSz{z}n{n}_D1S_{pp.replace('_', '')}.txt" for pp in pair_constr]
    #     # files_ = [f"export_TES_{pp}_z{z}n{n}_D1S_MZ5.txt" for pp in pair_constr]
    #     files_ = [f"export_TES_{pp}_z{z}n{n}_D1S_MZ4.txt" for pp in pair_constr]
    #
    #     plt_obj = Plotter1D_Taurus(files_, importNow=False)
    #     plt_obj.LATEX_FORMAT = True
    #     plt_obj.setConstraintBase(pair_constr)
    #     plt_obj._getDataFromFiles(files_)
    #
    #     # E_J0 = plt_obj.minimum_result[files_[0]].E_HFB
    #     # inertia_moment._function = lambda jx, erot: .5*(jx*(jx+1)) / (erot - E_J0 + 1.e-3)
    #     attr2plot_list = [
    #         'E_HFB', 
    #         'beta_isoscalar', #'gamma_isoscalar', 
    #         'pair', 'pair_nn','pair_pp', 'pair_pn', 
    #         #*pair_constr,
    #         pairsT1_opp, pairsT0_opp,
    #         'Jy', 'Jx'
    #         ]
    #
    #     for attr2plot in attr2plot_list:
    #         plt_obj.setXlabel("Pair Constr. value")
    #         plt_obj.defaultPlot(attr2plot, show_plot=attr2plot==attr2plot_list[-1])
    #     plt_obj.mergeFiguresInto1PDF('pair', z,n, 'D1S-MZ4')
    #
    #     _=0
    
    #===========================================================================
    # PLOT J cranking surfaces
    #===========================================================================
    
    # SUBFLD_ = ''
    # # SUBFLD_ = 'SDnuclei_MZ5/'
    # Plotter1D_Taurus.setFolderPath2Import('../DATA_RESULTS/Cranking/'+SUBFLD_)
    #
    # # nuclei = [(z, z) for z in range(8, 21, 2)]
    # constr = ['Jx']
    # nuclei = [#(10, 10), 
    #           (10, 11), (16, 17), 
    #           #(18,18)
    #           ]
    # pair_constr = ['P_T10_J00', 'P_T1p1_J00', 'P_T1m1_J00',  
    #                'P_T00_J10', 'P_T00_J1p1', 'P_T00_J1m1']
    #
    # pairsT1_opp = DataAttributeHandler(
    #     lambda mT0, mTp1, mTm1: (mT0**2) - 2*(mTp1*mTm1),
    #     'P_T10_J00', 'P_T1p1_J00', 'P_T1m1_J00')
    # pairsT1_opp.setName('mod(delta^{T=1,J=0})')
    #
    # pairsT0_opp = DataAttributeHandler(
    #     lambda mJ0, mJp1, mJm1: (mJ0**2) - 2*(mJp1*mJm1),
    #     'P_T00_J10', 'P_T00_J1p1', 'P_T00_J1m1')
    # pairsT0_opp.setName('mod(delta^{T=0,J=1})')
    #
    # inertia_moment = DataAttributeHandler(
    #     lambda jx, erot: 0.5*(jx*(jx+1)) / erot, 
    #     'Jx', 'E_HFB')
    # inertia_moment.setName("Mom inertia Jx")
    #
    # attr2plot_list = [
    #         'E_HFB', 
    #         # 'hf', 'hf_pp', 'hf_nn', 'hf_pn',
    #         'pair_pp', 'pair_nn', 'pair_pn', 
    #         *pair_constr,
    #         pairsT1_opp, pairsT0_opp,
    #         # 'b30_isoscalar',  'b32_isoscalar',
    #         'var_n', 'var_p',
    #         'beta_isoscalar', 'gamma_isoscalar',
    #         'E_HFB_pp', 'E_HFB_nn', 'E_HFB_pn', 
    #         inertia_moment, 
    #         ]
    #
    # Plotter1D_Taurus.EXPORT_PDF_AND_MERGE = True
    #
    # for z, n in nuclei:
    #     if SUBFLD_.startswith('Mg' ):
    #         if z != 12: continue
    #     # files_ = [f"export_PSz{z}n{n}_D1S_{pp.replace('_', '')}.txt" for pp in pair_constr]
    #     files_ = [f"export_TES_{pp}_z{z}n{n}_D1S_MZ5.txt" for pp in constr]
    #
    #     # x = DataTaurus(0,0,None)
    #     # x.beta
    #
    #     plt_obj = Plotter1D_Taurus(files_)
    #     plt_obj.LATEX_FORMAT = True
    #     # E_J0 = plt_obj._results[files_[0]][0].__getattribute__('E_HFB')
    #     E_J0 = plt_obj.minimum_result[files_[0]].E_HFB
    #     inertia_moment._function = lambda jx, erot: .5*(jx*(jx+1)) / (erot - E_J0 + 1.e-3)
    #     attr2plot_list[-1] = inertia_moment
    #
    #     for attr2plot in attr2plot_list:
    #         plt_obj.setXlabel("Jx Constr. value")
    #         plt_obj.defaultPlot(attr2plot, 
    #                             show_plot = attr2plot==attr2plot_list[-1])
    #     plt_obj.mergeFiguresInto1PDF('pairModule for','cranking', z,n)
    #     _=0
    
    #===========================================================================
    # PLOT NILSON TYPE SURFACES
    #===========================================================================
    
    # SUBFLD_ = 'BU_folder_D1S_MZ5_z10n10'
    # # # SUBFLD_ = 'SDnuclei_MZ5/'
    # Plotter1D_CanonicalBasis.setFolderPath2Import('../DATA_RESULTS/Cranking/')
    # files_ = ['export_TES_Jx_z16n17_D1S_MZ4.txt', ]
    # plt_obj = Plotter1D_CanonicalBasis(files_, )
    # plt_obj.setXlabel("Jx Constr. value")
    # attr2plot_list = [
    #         'E_HFB', 
    #         # 'E_HFB_pp', 'E_HFB_nn', 'E_HFB_pn', 
    #         # 'hf', 'hf_pp', 'hf_nn', 'hf_pn',
    #         'pair_pp', 'pair_nn', 'pair_pn', 
    #         # *pair_constr,
    #         'beta_isoscalar', 'gamma_isoscalar',
    #         # 'b30_isoscalar',  'b32_isoscalar',
    #         'Jz', 'var_n', 'var_p'
    #         ]
    #
    # attr2plot = [
    #     Plotter1D_CanonicalBasis.AttrEigenbasisEnum.h,
    #     # Plotter1D_CanonicalBasis.AttrEigenbasisEnum.avg_neutron,
    #     # Plotter1D_CanonicalBasis.AttrEigenbasisEnum.v2,
    #     Plotter1D_CanonicalBasis.AttrEigenbasisEnum.fermi_energ_prot,
    #     ]
    # for ind_, attr_ in enumerate(attr2plot):
    #     plt_obj.defaultPlot(attr_, index_2_print=None,
    #                         attr2plotExport=attr2plot_list,
    #                         show_plot=ind_==len(attr2plot)-1)
    # _=0
    