'''
Created on Nov 6, 2022

@author: Miguel

* Module to synthesize Taurus outputs 
TODO: * Include dataAxial
TODO: * Convert calculation results to be read as DataTaurus objects.
TODO: * Create Templates for Taurus inputs and the different evaluations.
    with default properties, constraints, parameters etc.
TODO: * Evaluation profile, for taurus execution details as an on fly verifier 
    to check convergence properties and modify to achieve convergence suitable
TODO: 
'''
from collections import OrderedDict
from datetime import datetime, timedelta
import re
import os
import shutil

import numpy as np
from tools.helpers import ValenceSpacesDict_l_ge10_byM, readAntoine,\
    getSingleSpaceDegenerations, almostEqual, LINE_2, LINE_1, printf, \
    liquidDropModelBindingEnergy, getQLMvalueFromBetaLM
from copy import copy, deepcopy
from tools.Enums import Enum, Constants
from tools.inputs import InputTaurus

class DataObjectException(BaseException):
    pass


class _DataObjectBase:
    
    """ Abstract class with common methods """
    DEFAULT_OUTPUT_FILENAME = 'aux_output.OUT'
    INPUT_FILENAME  = 'aux.INP'
    BU_folder       = 'BU_results'
    BU_fold_constr  = 'BU_results_constr'
    EXPORT_LIST_RESULTS = 'export_resultTaurus.txt'
    PROGRAM         = 'taurus_vap.exe'
    
    class HeaderEnum(Enum):
        """ Enumerate for the line headers of every argument to process"""
        pass
    
    @classmethod
    def setUpFolderBackUp(cls, new_export_list_filename=None):
        # Create new BU folder
        if not os.path.exists(cls.BU_folder):
            os.mkdir(cls.BU_folder)
        else:
            shutil.rmtree(cls.BU_folder)
            os.mkdir(cls.BU_folder)
        
        if new_export_list_filename:
            cls.EXPORT_LIST_RESULTS = new_export_list_filename.replace(' ', '')
        if os.path.exists(cls.EXPORT_LIST_RESULTS):
            os.remove(cls.EXPORT_LIST_RESULTS)
        
    def _getValues(self, line, head_rm = ''):
        line = line.replace(head_rm, '').split()
        vals = [float(l) if not '*' in l else np.NaN for l in line]
        return vals
    
    def __init__(self):
        raise BaseException("Abstract Class, cannot make instances!")
    
    def getResults(self):
        raise BaseException("Abstract method, implement me!")
    
    def _getSingleSpaceDegenerations(self, MZmax, MZmin=0):
        """ 
        Auxiliary method to get the valence space for the no-core calculation
        """
        sh_dim, sp_dim = getSingleSpaceDegenerations(MZmax, MZmin)
        
        self.sh_dim = sh_dim
        self.sp_dim = sp_dim * 2 # include z and n
    
    def isAxial(self, and_spherical=False):
        raise DataObjectException("Abstract method, implement me!")
        return False

class EvolTaurus(_DataObjectBase):
    '''
    Execution profile for Taurus execution details as an on fly verifier 
    to check convergence properties and modify to achieve convergence suitable
    
        Stores:
            Timestamps of the process (DUPLICATED)
            Status of the data        (DUPLICATED)
            Gradient properties
            Gradient/energy/varN2
            Spatial Density evolution
            H2Top values evolution
    '''
    class HeaderEnum(Enum):
        Grad_Type = 'Type of gradient             '
        Grad_Tol  = 'Tolerance for gradient       '
        Grad_eta  = 'Parameter eta for gradient   '
        Grad_mu   = 'Parameter mu  for gradient   '
        
        SP_dim  = 'No. of sp states'
        SH_dim  = 'No. of shells   '
        MZmax   = 'Max. value of N '
        HO_hbar = 'hbar*omega (MeV)  '
        HO_len  = 'Osc. length b (fm)'
        
        N_protons  = 'Number of active protons     '
        N_neutrons = 'Number of active neutrons    '
    
    __message_startiter = '                   ITERATIVE MINIMIZATION'
    __message_enditer   = '               QUASIPARTICLE STATE PROPERTIES'
    __message_converged = 'Calculation converged'
    __message_not_conv  = 'Maximum number of iterations reached'
    __endIteration_message = 'TIME_END:' # 'Label of the state: '
    FMT_DT = '%Y/%m/%d %H_%M_%S.%f'
    
    def __init__(self, filename, empty_data=False):
        '''
        Constructor
        '''
        self.filename = filename
        self.z = None
        self.n = None
        self._iter     = 0
        self.grad_type = 0
        self.grad_tol  = None
        self.eta = []
        self.mu  = []
        self.grad  = []
        self.e_hfb = []
        self.var_Z = []
        self.var_N = []
        self.top_h2  = None
        self.sp_dens = None
        
        self.date_start      = None 
        self.date_start_iter = None
        self.date_end_iter   = None
        self.iter_max        = None
        self.iter_time_seconds = None
        self.time_per_iter = None
        
        self.sp_dim = None
        self.MZmax  = None
        self.ho_hbaromega = None
        self.ho_b_length  = None
        self.sh_dim = None
        
        self._dataProcessed = False
        
        if not empty_data:
            try:
                self.get_results()
            except Exception as e:
                printf(" (TEvC)>> EXCEPTION from Taurus Constructor >> self::")
                printf(self)
                printf(" (TEvC)>> exception:: ", e, "<<(TC)")
                printf(" (TEvC)<< EXCEPTION from Taurus Constructor <<<<<<<< ")
    
    def get_results(self):
        
        with open(self.filename, 'r') as f:
            data = f.read()
            if self.__message_converged in data: 
                self.properly_finished = True
            elif not self.__message_enditer in data:
                self.broken_execution  = True
                return
            f.seek(0) # rewind the file reading
            
            if self.__message_converged in data: 
                self.properly_finished = True
            elif not self.__message_enditer in data:
                self.broken_execution  = True
            f.seek(0) # rewind the file reading
            
            data_inp, data_evol  = f.read().split(self.__message_startiter)
            data_evol, data = data_evol.split(self.__message_enditer)
            
        self._processing_data_and_evolution(data_inp, data_evol, True)
        
        printf()
        printf("ITER MAX: ", self.iter_max, self.z, self.n)
        printf("LEN GRAD: ", self.grad.__len__())
        printf("LEN EHFB: ", self.e_hfb.__len__())
        printf("LEN ETA: ", self.eta.__len__())
        printf("LEN MU: ", self.mu.__len__())
        printf("LEN VAR Z:", self.var_Z.__len__())
        printf("LEN VAR N:", self.var_N.__len__())
        if self.top_h2:
            printf("LEN htop_h2: ", self.top_h2.__len__())
        if self.sp_dens:
            printf("LEN sp_dens: ", self.sp_dens.__len__())
    
    # def __str__(self):
    #     aux = OrderedDict(sorted(self.__dict__.items(), key=lambda t: t[0]))
    #     return "\n".join(k+' :\t'+str(v) for k,v in aux.items())
    
    def _read_calculation_evol(self, data_evol):
        
        _SP_DENS_KEY   = "<dens(r)> approx"
        _H2_KEY        = "*Top H2"
        _EVOL_STARTKEY = "Iteration     Gradient       Energy"
        _TR_TEST_STARTKEY = '[Warning] Tr(rho*Gam)/= 2a*Tr(rho*Rea) ='      
        
        if _SP_DENS_KEY in data_evol: self.sp_dens = []
        if _H2_KEY in data_evol:      self.top_h2  = []
        
        _eta_print = False
        if not isinstance(data_evol, list):
            data_evol = data_evol.split('\n')
                
        _1st_step = False
        for line in data_evol:
            
            if not _1st_step:
                _1st_step = line.startswith(_EVOL_STARTKEY)
                continue
            
            line = line.strip()
            if line.endswith(_SP_DENS_KEY):
                line = line.replace('*A*', '').replace(_SP_DENS_KEY, '').split()
                self.sp_dens.append( float(line[0]))
                if self._iter == 1: self.sp_dens.append( self.sp_dens[-1])
            elif line.startswith(_H2_KEY):
                line = line.replace(_H2_KEY, '').split()
                self.top_h2.append( (float(line[0]), float(line[1])) )
                # self.top_h2.append( float(line[1]) )
                if self._iter == 1: self.top_h2.append( self.top_h2[-1])
            elif line.startswith(_TR_TEST_STARTKEY):
                line = line.replace(_TR_TEST_STARTKEY, '').split()
                tr_ratio = float(line[-1]) - 6
                if abs(tr_ratio) > 1.0e-3:
                    printf("[WARNING] Trace Test went wrong (not 6.0), got", line[-1])
            
            else:
                line = line.split()
                _eta_print = len(line) == 9
                if len(line) == 0:
                    # end of process, out
                    break
                elif line[0].isnumeric():
                    # main step
                    self._iter = int(line[0])
                    self.grad .append(float(line[1]))
                    self.e_hfb.append(float(line[2]))
                    self.var_Z.append(float(line[4]))
                    self.var_N.append(float(line[6]))
                    if _eta_print:
                        if self._iter == 1:
                            self.eta.pop()
                            self.mu .pop()
                        self.eta.append(float(line[7]))
                        self.mu .append(float(line[8]))
                    
                elif "---" in line[0]: # hline for the table, dismiss
                    continue
                else:
                    printf("[TEvol. Parse WARNING]: unknown case (skip):", line)
        
        self.iter_max = self._iter
        if (self.iter_time_seconds, self.time_per_iter) == (None,)*2:
            self.iter_time_seconds = 0
            self.time_per_iter     = 0
        if (self.date_start, self.date_start_iter, self.date_end_iter) == (None,)*3:
            void_datetime = "1975-01-01 00:00:01"
            void_datetime = datetime.strptime(void_datetime, "%Y-%m-%d %H:%M:%S")
            self.date_start      = void_datetime
            self.date_start_iter = void_datetime
            self.date_end_iter   = void_datetime
    
    def _read_HO_prop(self, line):
        
        if   line.startswith(self.HeaderEnum.SP_dim):
            self.sp_dim = int(self._getValues(line, self.HeaderEnum.SP_dim)[0])
        elif line.startswith(self.HeaderEnum.MZmax):
            self.MZmax  = int(self._getValues(line, self.HeaderEnum.MZmax )[0])
        elif line.startswith(self.HeaderEnum.HO_hbar):
            self.ho_hbaromega = self._getValues(line, self.HeaderEnum.HO_hbar)[0]
        elif line.startswith(self.HeaderEnum.HO_len):
            self.ho_b_length  = self._getValues(line, self.HeaderEnum.HO_len) [0]
        elif line.startswith(self.HeaderEnum.SH_dim):
            self.sh_dim = int(self._getValues(line, self.HeaderEnum.SH_dim)[0])
            self._read_inp = False
    
    def _read_gradient_prop(self, line):
        
        if   line.startswith(self.HeaderEnum.Grad_Type):
            self.grad_type = int(self._getValues(line, self.HeaderEnum.Grad_Type)[0])
        elif line.startswith(self.HeaderEnum.Grad_eta):
            self.eta.append(self._getValues(line, self.HeaderEnum.Grad_eta)[0])
        elif line.startswith(self.HeaderEnum.Grad_mu):
            self.mu.append(self._getValues(line, self.HeaderEnum.Grad_mu)  [0])
        elif line.startswith(self.HeaderEnum.Grad_Tol):
            self.grad_tol = self._getValues(line, self.HeaderEnum.Grad_Tol)[0]
        elif line.startswith(self.HeaderEnum.N_protons):
            self.z = int(self._getValues(line, self.HeaderEnum.N_protons) [0])
        elif line.startswith(self.HeaderEnum.N_neutrons):
            self.n = int(self._getValues(line, self.HeaderEnum.N_neutrons)[0])
             
    def _processing_data_and_evolution(self, data_inp, data_evol, read_evol = True):
        
        times_execution = []
        # hT1, hT2, hT3 = 'TIME_START: ', 'TIME_START_ITER: ', 'TIME_END: '
        hT1, hT2, hT3 = 'TIME_START: ', 'TIME_START: ', 'TIME_END: '
        self._read_inp   = True
                
        for line in data_inp.split('\n'):
            
            if line.startswith(hT1):
                line = line.replace(hT1, '').replace('\n', '')
                try:
                    times_execution.append(datetime.strptime(line,self.FMT_DT))
                except ValueError:
                    self.FMT_DT = '%Y/%m/%d %H_%M_%S'
                    times_execution.append(datetime.strptime(line,self.FMT_DT))
            ##
            if self._read_inp: ## TODO: Migrate this to Evol Object
                self._read_gradient_prop(line)
                self._read_HO_prop(line)
            else:
                break
        
        if data_evol in ('', None): return
        
        if hT2 in data_evol: ## get the line as the first 80 chars of the bock
            line = data_evol.split(hT2)[1][:80].split('\n')[0]
            times_execution.append(datetime.strptime(line,self.FMT_DT))
        if hT3 in data_evol:
            line = data_evol.split(hT3)[1][:80].split('\n')[0]
            line = line.split(' ITER_FINAL=')
            line, iter_max = line[0], line[1]
            self.iter_max = max(int(iter_max) - 1, 1) # to avoid 0/0
            times_execution.append(datetime.strptime(line, self.FMT_DT))
        # save time related calculations
        if len(times_execution) == 3:
            self.iter_time_seconds = times_execution[2] - times_execution[1]
            self.iter_time_seconds = self.iter_time_seconds.seconds + \
                                     (1.0e-6 * self.iter_time_seconds.microseconds)
            self.time_per_iter = float(self.iter_time_seconds) / self.iter_max
        
            self.date_start      = times_execution[0] 
            self.date_start_iter = times_execution[1]
            self.date_end_iter   = times_execution[2]
        
        ## read the evolution
        if read_evol:
            self._read_calculation_evol(data_evol)
        self._dataProcessed = True
    
    def printEvolutionFigure(self, export_filename=None): 
        """ 
        Plot evolution of parameters.
        export_filename Optional, <string> with extension for figure export (.png, .pdf)
        """
        if not self._dataProcessed:
            self.get_results()
        
        if self.iter_max <= 1:
            printf("[Warning DD evol] Cannot print revolution results for void calculation.")
            return 
        import matplotlib.pyplot as plt
        
        fig,ax = plt.subplots(2, 2)
        
        ax[0,0].plot(self.e_hfb, color='blue', linestyle='-.')
        ax[0,0].set_ylabel("E HFB evol. [MeV]",  color='blue', fontsize=14)
        ax2 = ax[0,0].twinx()
        ax2.plot(self.grad,  color='purple', linestyle='-.')
        ax2.set_ylabel("Gradient evol.", color='purple', fontsize=14, labelpad=-50)
        
        ax[1,0].semilogy(self.eta*self.iter_max, color='red', linestyle='-.')
        ax[1,0].set_ylabel("Grad Eta evol",  color='red', fontsize=14, labelpad=-50)
        ax3 = ax[1,0].twinx()
        ax3.semilogy(self.mu*self.iter_max, color='purple', linestyle='-.')
        ax3.set_ylabel("Grad Mu evol", color='purple', fontsize=14, labelpad=-50)
        if self.top_h2:
            v_dd_max, v_dd_min = zip(*self.top_h2)
            ax[0,1].plot(v_dd_max,'b--')
            ax[0,1].plot(v_dd_min,'r--')
            ax[0,1].set_ylabel("max 2b <v(DD)> [MeV]")
            ax[0,1].fill_between([i for i in range(len(v_dd_max))], 
                                 v_dd_max, v_dd_min)
            
        ax[1,1].plot(self.var_Z, 'r.-', label='var Z')
        ax[1,1].plot(self.var_N, 'b.-', label='var N')
        ax[1,1].legend()
        ax[1,1].set_title("var N,Z")
        
        d = self.date_end_iter - self.date_start
        d = str(d).split(".")[0]
        plt.suptitle(f"Calculation Evolution for Z,N=({self.z}, {self.n})"
                     f"\nsp_dim={self.sp_dim} (MZ:{self.MZmax}) b={self.ho_b_length:5.3f} fm "
                     f"   step_duration {d} [{self.iter_max}]")
        
        fig.tight_layout()
        plt.show()
        if export_filename:
            fig.savefig(export_filename)
    
    def isAxial(self, and_spherical=False):
        raise DataObjectException("Axial property is not deffined for this object!")
        return False

class EigenbasisData(_DataObjectBase):
    
    """ 
    Object to manage the data information of canonicalbasis and eigenbasis_h/H11
    """
    class HeaderEnum(Enum):
        FermiProt = "Proton  fermi energy ="
        FermiNeut = "Neutron fermi energy ="
    
    EIGENBASIS_H_HEADER   = '   #      Z        N        n        l        p        j       jz         h  '
    CANONICALBASIS_HEADER = '   #      Z        N        n        l        p        j       jz         v2           h '
    EIGENBASIS_H11_HEADER = '   #      Z        N        n        l        p        j       jz         H11'
            
    def __init__(self, filename=None):
        
        self.fermi_energ_prot = None
        self.fermi_energ_neut = None 
        self.DAT_FILE = None
        
        self.index = []
        self.avg_proton  = []
        self.avg_neutron = []
        self.avg_n    = []
        self.avg_l    = []
        self.avg_parity  = []
        self.avg_j    = []
        self.avg_jz   = []
        self.v2   = []
        self.h    = []
        self.H11  = []
        
        self.filename = filename
        
    def getResults(self):
        """
        Once specified the filename, get the data of the file.
        """
        with open(self.filename, 'r') as f:
            data = f.readlines()
        self._setDATFile(data)
        
        ## #      Z        N        n        l        p        j       jz    ...
        ## -------------------------------------------------------------------
        _in_values = False
        for line in data:
            line = line.strip()
            if len(line) == 0: 
                continue
            if not _in_values:
                if   line.startswith(self.HeaderEnum.FermiProt):
                    self.fermi_energ_prot = self._getValues(line, self.HeaderEnum.FermiProt)[0]
                elif line.startswith(self.HeaderEnum.FermiNeut):
                    self.fermi_energ_neut = self._getValues(line, self.HeaderEnum.FermiNeut)[0]
                elif line.startswith('---'):
                    _in_values = True
                continue
            else:
                
                vals = self._getValues(line)
                vals[0] = int(vals[0])
                
                self.index.append(vals[0])
                self.avg_proton.append(vals[1])
                self.avg_neutron.append(vals[2])
                self.avg_n.append( vals[3])
                self.avg_l.append( vals[4])
                self.avg_parity.append( vals[5])
                self.avg_j.append( vals[6])
                self.avg_jz.append(vals[7])
                
                if self.DAT_FILE == DataTaurus.DatFileExportEnum.canonicalbasis:
                    self.v2 .append(vals[8])
                    self.h  .append(vals[9])
                if self.DAT_FILE == DataTaurus.DatFileExportEnum.eigenbasis_h:
                    self.h  .append(vals[8])
                if self.DAT_FILE == DataTaurus.DatFileExportEnum.eigenbasis_H11:
                    self.H11.append(vals[8])    
        _=0
    
    def _setDATFile(self, lines_file):
        """ Identify the file by the column names at the top. """
        if   self.EIGENBASIS_H11_HEADER in lines_file[0]:
            self.DAT_FILE = DataTaurus.DatFileExportEnum.eigenbasis_H11
        elif self.EIGENBASIS_H_HEADER   in lines_file[3]:
            self.DAT_FILE = DataTaurus.DatFileExportEnum.eigenbasis_h
        elif self.CANONICALBASIS_HEADER in lines_file[3]:
            self.DAT_FILE = DataTaurus.DatFileExportEnum.canonicalbasis
        else:
            raise DataObjectException(f"Cannot identify the file [{self.filename}]")
    
    def isAxial(self, and_spherical=False):
        raise DataObjectException("TODO: implement me!")
        return False

class OccupationNumberData(_DataObjectBase):
    
    """ 
    Object to manage the data from occupation_number file.
    """
    
    def __init__(self, filename=None):
        
        self._occupations_unprojected = {-1: {}, 1: {}}
        self._occupations_projected   = {-1: {}, 1: {}}
        self._numbers_by_label = {}
        
        self.filename = filename
        
        if self.filename: self.getResults()
    
    def getResults(self):
        """
        Once specified the filename, get the data of the file.
        """
        with open(self.filename, 'r') as f:
            data = f.readlines()[3:]
        
        ##  #    2*mt    n     l    2*j   label   unprojected    projected
        ##----------------------------------------------------------------
        for line in data:
            
            if line.startswith('sum') or line.startswith('---'):
                continue
            _, mt, n, l, j, label, unproj, proj = line.split()
            
            mt, n, l, j, label = int(mt), int(n), int(l), int(j), int(label)
            unproj, proj = float(unproj), float(proj)
            
            if label in self._occupations_unprojected[mt]:
                raise DataObjectException(f"Label [{label}]is already registered:"
                                          f"{self._numbers_by_label.keys()}")
            self._numbers_by_label[label] = (n, l, j)
            self._occupations_projected[mt][label]   = proj
            self._occupations_unprojected[mt][label] = unproj
    
    @property
    def hasProjectedOccupations(self):
        """
        Are there any value of projected occupations.
        """
        if ((len(self._occupations_projected[-1])> 0)
             or (len(self._occupations_projected[1])> 0)):
            for mt in (-1, 1):
                vals = self._occupations_projected[mt].values()
                if any(filter(lambda x: abs(x)>1.0e-6, vals)):
                    return True
            return False
        else:
            return False
    
    
    @property
    def get_numbers(self):
        return self._numbers_by_label
    @property
    def get_occupations(self):
        if self.hasProjectedOccupations:
            return self._occupations_unprojected, self._occupations_projected
        return self._occupations_unprojected, None

    def isAxial(self, and_spherical=False):
        raise DataObjectException("Axial property is not deffined for this object!")
        return False

#===============================================================================
#   RESULTS FROM TAURUS 
#===============================================================================

_TMP_TAURUS_VAP_RESULT_OUTPUT = """
      Quantity            Mean      Variance
--------------------------------------------
Norm                   {overlap:10.6f}
Number of protons      {proton_numb:10.6f}   {var_p:8.6f}
Number of neutrons     {neutron_numb:10.6f}   {var_n:8.6f}
Parity                 {parity:10.6f}

ENERGY DECOMPOSITION
====================

Part \ Iso      p-p          n-n          p-n          Total
-------------------------------------------------------------
Zero-body                                         {E_zero: >12.6f}
One-body   {kin_p: >12.6f} {kin_n: >12.6f}               {kin: >12.6f}
 ph part   {hf_pp: >12.6f} {hf_nn: >12.6f} {hf_pn: >12.6f} {hf: >12.6f}
 pp part   {pair_pp: >12.6f} {pair_nn: >12.6f} {pair_pn: >12.6f} {pair: >12.6f}
Two-body   {V_2B_pp: >12.6f} {V_2B_nn: >12.6f} {V_2B_pn: >12.6f} {V_2B: >12.6f}
Full H     {E_HFB_pp: >12.6f} {E_HFB_nn: >12.6f} {E_HFB_pn: >12.6f} {E_HFB: >12.6f}

MULTIPOLE DEFORMATIONS
======================
                                     Nucleons
Q_lm     Protons    Neutrons   Isoscalar   Isovector
----------------------------------------------------
Q_10  {q10_p: >10.6f}  {q10_n: >10.6f}  {q10_isoscalar: >10.6f}  {q10_isovector: >10.6f}
Q_11  {q11_p: >10.6f}  {q11_n: >10.6f}  {q11_isoscalar: >10.6f}  {q11_isovector: >10.6f}
Q_20  {q20_p: >10.6f}  {q20_n: >10.6f}  {q20_isoscalar: >10.6f}  {q20_isovector: >10.6f}
Q_21  {q21_p: >10.6f}  {q21_n: >10.6f}  {q21_isoscalar: >10.6f}  {q21_isovector: >10.6f}
Q_22  {q22_p: >10.6f}  {q22_n: >10.6f}  {q22_isoscalar: >10.6f}  {q22_isovector: >10.6f}
Q_30  {q30_p: >10.6f}  {q30_n: >10.6f}  {q30_isoscalar: >10.6f}  {q30_isovector: >10.6f}
Q_31  {q31_p: >10.6f}  {q31_n: >10.6f}  {q31_isoscalar: >10.6f}  {q31_isovector: >10.6f}
Q_32  {q32_p: >10.6f}  {q32_n: >10.6f}  {q32_isoscalar: >10.6f}  {q32_isovector: >10.6f}
Q_33  {q33_p: >10.6f}  {q33_n: >10.6f}  {q33_isoscalar: >10.6f}  {q33_isovector: >10.6f}
Q_40  {q40_p: >10.6f}  {q40_n: >10.6f}  {q40_isoscalar: >10.6f}  {q40_isovector: >10.6f}
Q_41  {q41_p: >10.6f}  {q41_n: >10.6f}  {q41_isoscalar: >10.6f}  {q41_isovector: >10.6f}
Q_42  {q42_p: >10.6f}  {q42_n: >10.6f}  {q42_isoscalar: >10.6f}  {q42_isovector: >10.6f}
Q_43  {q43_p: >10.6f}  {q43_n: >10.6f}  {q43_isoscalar: >10.6f}  {q43_isovector: >10.6f}
Q_44  {q44_p: >10.6f}  {q44_n: >10.6f}  {q44_isoscalar: >10.6f}  {q44_isovector: >10.6f}

                                        Nucleons
Beta_lm     Protons    Neutrons   Isoscalar   Isovector
-------------------------------------------------------
Beta_10  {b10_p: >10.6f}  {b10_n: >10.6f}  {b10_isoscalar: >10.6f}  {b10_isovector: >10.6f}
Beta_11  {b11_p: >10.6f}  {b11_n: >10.6f}  {b11_isoscalar: >10.6f}  {b11_isovector: >10.6f}
Beta_20  {b20_p: >10.6f}  {b20_n: >10.6f}  {b20_isoscalar: >10.6f}  {b20_isovector: >10.6f}
Beta_21  {b21_p: >10.6f}  {b21_n: >10.6f}  {b21_isoscalar: >10.6f}  {b21_isovector: >10.6f}
Beta_22  {b22_p: >10.6f}  {b22_n: >10.6f}  {b22_isoscalar: >10.6f}  {b22_isovector: >10.6f}
Beta_30  {b30_p: >10.6f}  {b30_n: >10.6f}  {b30_isoscalar: >10.6f}  {b30_isovector: >10.6f}
Beta_31  {b31_p: >10.6f}  {b31_n: >10.6f}  {b31_isoscalar: >10.6f}  {b31_isovector: >10.6f}
Beta_32  {b32_p: >10.6f}  {b32_n: >10.6f}  {b32_isoscalar: >10.6f}  {b32_isovector: >10.6f}
Beta_33  {b33_p: >10.6f}  {b33_n: >10.6f}  {b33_isoscalar: >10.6f}  {b33_isovector: >10.6f}
Beta_40  {b40_p: >10.6f}  {b40_n: >10.6f}  {b40_isoscalar: >10.6f}  {b40_isovector: >10.6f}
Beta_41  {b41_p: >10.6f}  {b41_n: >10.6f}  {b41_isoscalar: >10.6f}  {b41_isovector: >10.6f}
Beta_42  {b42_p: >10.6f}  {b42_n: >10.6f}  {b42_isoscalar: >10.6f}  {b42_isovector: >10.6f}
Beta_43  {b43_p: >10.6f}  {b43_n: >10.6f}  {b43_isoscalar: >10.6f}  {b43_isovector: >10.6f}
Beta_44  {b44_p: >10.6f}  {b44_n: >10.6f}  {b44_isoscalar: >10.6f}  {b44_isovector: >10.6f}

                                        Nucleons
Triaxial    Protons    Neutrons   Isoscalar   Isovector
-------------------------------------------------------
Beta    {beta_p: >10.6f}  {beta_n: >10.6f}  {beta_isoscalar: >10.6f}  {beta_isovector: >10.6f}
Gamma   {gamma_p: >10.6f}  {gamma_n: >10.6f}  {gamma_isoscalar: >10.6f}  {gamma_isovector: >10.6f}

RADIUS
======
                                        Nucleons
Quantity    Protons    Neutrons   Isoscalar   Isovector     Charge
-------------------------------------------------------------------
  r      {r_p: >10.6f}  {r_n: >10.6f}  {r_isoscalar: >10.6f}  {r_isovector: >10.6f}  {r_charge: >10.6f}
  r^2    {r_p2: >10.6f}  {r_n2: >10.6f}  {r_isoscalar2: >10.6f}  {r_isovector2: >10.6f}  {r_charge2: >10.6f}

Warning: no center-of-mass correction.

ANGULAR MOMENTUM
================

  i        J_i         J_i^2     Var(J_i)
-----------------------------------------
  X   {Jx: >11.6f}  {Jx_2: >11.6f}  {Jx_var: >11.6f}
  Y   {Jy: >11.6f}  {Jy_2: >11.6f}  {Jy_var: >11.6f}
  Z   {Jz: >11.6f}  {Jz_2: >11.6f}  {Jz_var: >11.6f}
Total               {jtot: >11.6f}

PAIR COUPLING
=============

   MJ or MT =       -1           0          +1
-------------------------------------------------
T = 0 ; J = 1  {P_T00_J1m1: >11.6f}  {P_T00_J10: >11.6f}  {P_T00_J1p1: >11.6f}
T = 1 ; J = 0  {P_T1m1_J00: >11.6f}  {P_T10_J00: >11.6f}  {P_T1p1_J00: >11.6f}

    2-BODY OPERATOR (pn - pairs)

   MJ or MT =       -1           0          +1
-------------------------------------------------
2B T= 0; J= 1  {P_T00_J1m1: >11.6f}  {P_T00_J10: >11.6f}  {P_T00_J1p1: >11.6f}
2B T= 1; J= 0  {P_T1m1_J00: >11.6f}  {P_T10_J00: >11.6f}  {P_T1p1_J00: >11.6f}
"""

class DataTaurus(_DataObjectBase):
    
    """
    Object to process all the methods related to the taurus_vap.exe output.
        Exporting:
            Energies ()
    """
    class HeaderEnum(Enum): ## line header to read the output file
        Number_of_protons  = 'Number of protons '
        Number_of_neutrons = 'Number of neutrons'
        Parity   = 'Parity'
        Overlap  = 'Projected overlap'
        Zero_body= 'Zero-body'
        One_body = 'One-body'
        ph_part  = ' ph part'
        pp_part  = ' pp part'
        Two_body = 'Two-body'
        Full_H   = 'Full H'
        Beta_10  = 'Beta_10'
        Beta_11  = 'Beta_11'
        Beta_20  = 'Beta_20'
        Beta_22  = 'Beta_22'
        Beta_30  = 'Beta_30'
        Beta_32  = 'Beta_32'
        Beta_40  = 'Beta_40'
        Beta_42  = 'Beta_40'
        Beta_44  = 'Beta_40'
        Q_10     = 'Q_10'
        Q_11     = 'Q_11'
        Q_20     = 'Q_20'
        Q_21     = 'Q_21'
        Q_22     = 'Q_22'
        Q_30     = 'Q_30'
        Q_32     = 'Q_32'
        Q_40     = 'Q_40'
        Q_42     = 'Q_40'
        Q_44     = 'Q_40'
        Beta     = 'Beta '
        Gamma    = 'Gamma'
        R2med    = '  r^2 '
        Rmed     = '  r '
        Jx       = '  X    '
        Jy       = '  Y    '
        Jz       = '  Z    '
        PairT0J1 = 'T = 0 ; J = 1'
        PairT1J0 = 'T = 1 ; J = 0'
        
        SP_dim  = 'No. of sp states'
        SH_dim  = 'No. of shells   '
        MZmax   = 'Max. value of N '
        HO_hbar = 'hbar*omega (MeV)  '
        HO_len  = 'Osc. length b (fm)'
        dd_evol = ' *Top H2'
        label_st= 'Label of the state:'
    
    class DatFileExportEnum(Enum):
        """ filename of the possible .dat files from Taurus """
        canonicalbasis     = 'canonicalbasis'
        occupation_numbers = 'occupation_numbers'
        eigenbasis_h       = 'eigenbasis_h'
        eigenbasis_H11     = 'eigenbasis_H11'
        spatial_density_R  = 'spatial_density_R'
        spatial_density_XYZ = 'spatial_density_XYZ'
        spatial_density_RThetaPhi = 'spatial_density_RThetaPhi'
        
    
    __message_startiter = '                   ITERATIVE MINIMIZATION'
    __message_endvap    = '                 PROJECTED STATE PROPERTIES'
    __message_enditer   = '               QUASIPARTICLE STATE PROPERTIES'
    __message_converged = 'Calculation converged'
    __message_not_conv  = 'Maximum number of iterations reached'
    __endIteration_message = 'TIME_END:' # 'Label of the state: '
    
    PROGRAM         = 'taurus_vap.exe'
    
    FMT_DT = '%Y/%m/%d %H_%M_%S.%f'
    
    ## Template result to reconstruct the output object
    _TMP_RESULT = _TMP_TAURUS_VAP_RESULT_OUTPUT
    
    @classmethod
    def getDataVariable(cls, variable : str, beta_schm : int):
        """ 
        Method to get the DataTaurus ATTRIBUTE NAME for a Input variable, 
            i.e: InputTaurus.b20 -> DataTaurus.b20_isoscalar
        """
        if variable.startswith('b'):
            if   beta_schm == 0:
                return f'q{variable[1:]}_isoscalar'
            elif beta_schm == 1:
                return f'{variable}_isoscalar'
            elif beta_schm == 2:
                if variable   == 'b20':
                    return 'beta_isoscalar'
                elif variable == 'b22':
                    return 'gamma_isoscalar'
                else: 
                    raise Exception("for beta_schm == 2, only can be set [b20=beta, b22=gamma]")
            else:
                raise Exception("beta_schm <= 2, check")
        elif variable == 'sqrt_r2':
            return 'r_isoscalar'
        elif (variable.startswith('P_T') or variable in ('Jx', 'Jy', 'Jz')):
            return variable
        
    
    def __init__(self, z, n, filename, empty_data=False):
        
        self.z = z
        self.n = n
        self.properly_finished = False
        self.broken_execution  = False
        
        self.ho_b_length = None
        self.ho_hbaromega= None
        self.sp_dim      = None
        self.sh_dim      = None
        self.MZmax       = None
        
        self.proton_numb  = None
        self.neutron_numb = None
        self.var_p    = None
        self.var_n    = None
        self.parity   = None
        self.overlap  = 0
        self.label_state = None
        
        self.E_zero   = None
        self.kin      = None
        self.kin_p    = None
        self.kin_n    = None
        self.hf       = None
        self.hf_pp    = None
        self.hf_nn    = None
        self.hf_pn    = None
        self.pair     = None
        self.pair_pp  = None
        self.pair_nn  = None
        self.pair_pn  = None
        self.V_2B     = None  # e_hf + e_pair
        self.V_2B_pp  = None
        self.V_2B_nn  = None
        self.V_2B_pn  = None
        self.E_HFB    = None
        self.E_HFB_pp = None
        self.E_HFB_nn = None
        self.E_HFB_pn = None
        
        self.beta_p   = None
        self.beta_n   = None
        self.beta_isoscalar  = None
        self.beta_isovector  = None
        self.gamma_p  = None
        self.gamma_n  = None
        self.gamma_isoscalar = None
        self.gamma_isovector = None
        
        self.b10_p = None
        self.b10_n = None
        self.b10_isoscalar = None
        self.b10_isovector = None
        self.b11_p = None
        self.b11_n = None
        self.b11_isoscalar = None
        self.b11_isovector = None    
        self.b20_p = None
        self.b20_n = None
        self.b20_isoscalar = None
        self.b20_isovector = None
        self.b22_p = None
        self.b22_n = None
        self.b22_isoscalar = None
        self.b22_isovector = None
        self.b30_p = None
        self.b30_n = None
        self.b30_isoscalar = None
        self.b30_isovector = None
        self.b32_p = None
        self.b32_n = None
        self.b32_isoscalar = None
        self.b32_isovector = None
        self.b40_p = None
        self.b40_n = None
        self.b40_isoscalar = None
        self.b40_isovector = None
        self.b42_p = None
        self.b42_n = None
        self.b42_isoscalar = None
        self.b42_isovector = None
        self.b44_p = None
        self.b44_n = None
        self.b44_isoscalar = None
        self.b44_isovector = None
        
        self.q10_p = None
        self.q10_n = None
        self.q10_isoscalar = None
        self.q10_isovector = None
        self.q11_p = None
        self.q11_n = None
        self.q11_isoscalar = None
        self.q11_isovector = None
        self.q20_p = None
        self.q20_n = None
        self.q20_isoscalar = None
        self.q20_isovector = None
        self.q22_p = None
        self.q22_n = None
        self.q22_isoscalar = None
        self.q22_isovector = None
        self.q30_p = None
        self.q30_n = None
        self.q30_isoscalar = None
        self.q30_isovector = None
        self.q32_p = None
        self.q32_n = None
        self.q32_isoscalar = None
        self.q32_isovector = None
        self.q40_p = None
        self.q40_n = None
        self.q40_isoscalar = None
        self.q40_isovector = None
        self.q42_p = None
        self.q42_n = None
        self.q42_isoscalar = None
        self.q42_isovector = None
        self.q44_p = None
        self.q44_n = None
        self.q44_isoscalar = None
        self.q44_isovector = None
        
        self.r_p  = None
        self.r_n  = None
        self.r_isoscalar = None
        self.r_isovector = None
        self.r_charge = None
        
        self.Jx     = None
        self.Jx_2   = None
        self.Jx_var = None
        self.Jy     = None
        self.Jy_2   = None
        self.Jy_var = None
        self.Jz     = None
        self.Jz_2   = None
        self.Jz_var = None
        
        self.P_T00_J10  = None
        self.P_T00_J1m1 = None
        self.P_T00_J1p1 = None
        self.P_T10_J00  = None
        self.P_T1m1_J00 = None
        self.P_T1p1_J00 = None
        
        self.date_start      = None
        self.date_start_iter = None
        self.date_end_iter   = None
        self.iter_max  = None
        self.iter_time_seconds = None
        self.time_per_iter = None
        
        self.iter_time_cpu = 0
        self.time_per_iter_cpu = 0
        self.memory_max_KB = 0
        
        self._filename = filename
        self._evol_obj  : EvolTaurus  = None
        self._input_obj : InputTaurus = None
        self._exported_filename = ''
        self._is_vap_calculation = False
        if not empty_data:
            try:
                self.get_results()
            except Exception as e:
                printf(" (TC)>> EXCEPTION from Taurus Constructor >> last 5 lines::",
                       LINE_2)
                with open(self._filename, 'r') as f:
                    printf("".join(f.readlines()[-5:]), LINE_2)
                
                printf(" (TC)>> EXCEPTION from Taurus Constructor >> self::")
                printf(self)
                printf(" (TC)>> exception:: ", e, "<<(TC)")
                printf(" (TC)<< EXCEPTION from Taurus Constructor <<<<<<<< ")
        
    def __str__(self):
        aux = OrderedDict(sorted(self.__dict__.items(), key=lambda t: t[0]))
        return "\n".join(k+' :\t'+str(v) for k,v in aux.items())
    
    def _getValues(self, line, head_rm = ''):
        line = line.replace(head_rm, '').split()
        vals = [float(l) if not '*' in l else np.NaN for l in line]
        return vals
    
    def _read_HO_prop(self, line):
        
        if   line.startswith(self.HeaderEnum.SP_dim):
            self.sp_dim = int(self._getValues(line, self.HeaderEnum.SP_dim)[0])
        elif line.startswith(self.HeaderEnum.MZmax):
            self.MZmax  = int(self._getValues(line, self.HeaderEnum.MZmax )[0])
        elif line.startswith(self.HeaderEnum.HO_hbar):
            self.ho_hbaromega = self._getValues(line, self.HeaderEnum.HO_hbar)[0]
        elif line.startswith(self.HeaderEnum.HO_len):
            self.ho_b_length  = self._getValues(line, self.HeaderEnum.HO_len) [0]
        elif line.startswith(self.HeaderEnum.SH_dim):
            self.sh_dim = int(self._getValues(line, self.HeaderEnum.SH_dim)[0])
            self._read_inp = False
    
    def _processing_data_and_evolution(self, data_inp, data_evol):
        ## Process managed by the EvolTaurus class
        try:
            self._evol_obj = EvolTaurus(None, empty_data=True)
            self._evol_obj._processing_data_and_evolution(data_inp, data_evol, True)
        except BaseException as e_:
            printf(" (TEC)>> EXCEPTION found at Evol Taurus reading. ")
            printf(e_)
            printf(" (TEC)>> seting unreaded _evol_obj values to None.")
            self._evol_obj.date_end_iter = self._evol_obj.date_start_iter
        
        _attr2copy = ['date_start', 'date_start_iter', 'date_end_iter',
                      'iter_max', 'iter_time_seconds', 'time_per_iter',
                      'sp_dim', 'MZmax', 'ho_hbaromega', 'ho_b_length', 'sh_dim']
        
        for attr_ in _attr2copy:
            setattr(self, attr_, getattr(self._evol_obj, attr_, None))
    
    
    def get_results(self):
        """
        Read all the lines and export
            For DataTaurus, only quasiparticle states are saved:
        Note:
            In the case of VAP calculation, the stored energies are the projected 
            ones, the other observables will be from quasiparticles.
            It assign the values twice (QP values appear in second place)
        """
        with open(self._filename, 'r') as f:
            data = f.read()
            
            has_evol_data = self.__message_startiter in data
            if self.__message_converged in data: 
                self.properly_finished = True
            elif not self.__message_enditer in data:
                self.broken_execution  = True
                return
            self._is_vap_calculation = self.__message_endvap in data
            
            f.seek(0) # rewind the file reading
            
            if has_evol_data:
                data_inp, data_evol  = f.read().split(self.__message_startiter)
                if self._is_vap_calculation:
                    data_evol, data = data_evol.split(self.__message_endvap)
                else:
                    data_evol, data = data_evol.split(self.__message_enditer)
            else:
                ## NOTE: Modification if __message_converged do not appear but
                ## results do (iter=0 modification of taurus_vap), data evol is 
                ## ignored. The spliting will occur between the results header.
                if self._is_vap_calculation:
                    data_inp, data = f.read().split(self.__message_endvap)
                else:
                    data_inp, data = f.read().split(self.__message_enditer)
                data_evol = None
            data      = data.split('\n')
                    
        _energies = (self.HeaderEnum.Zero_body,  self.HeaderEnum.One_body,
                     self.HeaderEnum.ph_part,    self.HeaderEnum.pp_part,  
                     self.HeaderEnum.Two_body,   self.HeaderEnum.Full_H)
        
        self._read_inp   = True
        
        self._processing_data_and_evolution(data_inp, data_evol)
        
        for line in data:            
            # printf(line)
            if 'Number of' in line:
                self._getNumberNucleons(line)
            elif self.HeaderEnum.Overlap in line:
                self.overlap = self._getValues(line, self.HeaderEnum.Overlap)[0]
            elif self.HeaderEnum.Parity in line:
                self.parity = self._getValues(line, self.HeaderEnum.Parity)[0]
            elif self.HeaderEnum.label_st in line:
                self.label_state = int(line.replace(self.HeaderEnum.label_st, ''))
            ##
            elif self.HeaderEnum.Rmed in line:
                vals = self._getValues(line, self.HeaderEnum.Rmed)
                self.r_p, self.r_n = vals[0], vals[1]
                self.r_isoscalar, self.r_isovector = vals[2], vals[3]
                self.r_charge = vals[4]
            elif line.startswith('Beta'):
                self._getBetaDeformations(line)
            elif line.startswith('Q_'):
                self._getQDeformations(line)
            elif self.HeaderEnum.Gamma in line:
                vals = self._getValues(line, self.HeaderEnum.Gamma)
                self._roundGamma0(vals)
                self.gamma_p, self.gamma_n                 = vals[0], vals[1]
                self.gamma_isoscalar, self.gamma_isovector = vals[2], vals[3]
            if True in (p in line for p in _energies):
                self._getEnergies(line)
            if True in (d in line for d in 
                        (self.HeaderEnum.Jx, self.HeaderEnum.Jy, self.HeaderEnum.Jz)):
                self._getAngularMomentum(line)
            if self.HeaderEnum.PairT0J1 in line or self.HeaderEnum.PairT1J0 in line:
                self._getPairCoupling(line)
        
        # return dict([(e, float(val)) for e, val in energies.items()]), prop_fin
    
    def _getBetaDeformations(self, line):
        
        if   self.HeaderEnum.Beta in line:
            vals = self._getValues(line, self.HeaderEnum.Beta)
            self.beta_p = vals[0] 
            self.beta_n = vals[1]
            self.beta_isoscalar, self.beta_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_10 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_10)
            self.b10_p, self.b10_n  = vals[0], vals[1]
            self.b10_isoscalar, self.b10_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_11 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_11)
            self.b11_p, self.b11_n  = vals[0], vals[1]
            self.b11_isoscalar, self.b11_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_20 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_20)
            self.b20_p, self.b20_n  = vals[0], vals[1]
            self.b20_isoscalar, self.b20_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_22 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_22)
            self.b22_p, self.b22_n  = vals[0], vals[1]
            self.b22_isoscalar, self.b22_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_30 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_30)
            self.b30_p, self.b30_n  = vals[0], vals[1]
            self.b30_isoscalar, self.b30_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_32 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_32)
            self.b32_p, self.b32_n  = vals[0], vals[1]
            self.b32_isoscalar, self.b32_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_40 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_40)
            self.b40_p, self.b40_n  = vals[0], vals[1]
            self.b40_isoscalar, self.b40_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_42 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_42)
            self.b42_p, self.b42_n  = vals[0], vals[1]
            self.b42_isoscalar, self.b42_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Beta_44 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_44)
            self.b44_p, self.b44_n  = vals[0], vals[1]
            self.b44_isoscalar, self.b44_isovector = vals[2], vals[3]
    
    def _getQDeformations(self, line):
               
        if self.HeaderEnum.Q_10 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_10)
            self.q10_p, self.q10_n  = vals[0], vals[1]
            self.q10_isoscalar, self.q10_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Q_11 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_11)
            self.q11_p, self.q11_n  = vals[0], vals[1]
            self.q11_isoscalar, self.q11_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Q_20 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_20)
            self.q20_p, self.q20_n  = vals[0], vals[1]
            self.q20_isoscalar, self.q20_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Q_22 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_22)
            self.q22_p, self.q22_n  = vals[0], vals[1]
            self.q22_isoscalar, self.q22_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Q_30 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_30)
            self.q30_p, self.q30_n  = vals[0], vals[1]
            self.q30_isoscalar, self.q30_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Q_32 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_32)
            self.q32_p, self.q32_n  = vals[0], vals[1]
            self.q32_isoscalar, self.q32_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Q_40 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_40)
            self.q40_p, self.q40_n  = vals[0], vals[1]
            self.q40_isoscalar, self.q40_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Q_42 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_42)
            self.q42_p, self.q42_n  = vals[0], vals[1]
            self.q42_isoscalar, self.q42_isovector = vals[2], vals[3]
        elif self.HeaderEnum.Q_44 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_44)
            self.q44_p, self.q44_n  = vals[0], vals[1]
            self.q44_isoscalar, self.q44_isovector = vals[2], vals[3]
        #printf("deform results :: [{}]".format(vals))
    
    def _roundGamma0(self, vals):
        if abs(vals[2] - 180) < 1.e-5 or abs(vals[2] - 360) < 1.e-5:
            vals[2] = 0.0
    
    def _getEnergies(self, line):
        """
        In case of vap-calculation, only the store the VAP values
            Note: the lines goes in order, so if Projected states has been writen
            self.E_HFB != 0:
        """
        if self._is_vap_calculation and self.E_HFB != None:
            return
        
        if self.HeaderEnum.Zero_body in line:
            self.E_zero = self._getValues(line, self.HeaderEnum.Zero_body)[0]
        elif self.HeaderEnum.One_body in line:
            vals = self._getValues(line, self.HeaderEnum.One_body)
            self.kin_p, self.kin_n, self.kin = vals[0], vals[1], vals[2]
        else:
            if   self.HeaderEnum.ph_part in line:
                vals = self._getValues(line, self.HeaderEnum.ph_part)
                self.hf_pp, self.hf_nn = vals[0], vals[1]
                self.hf_pn, self.hf    = vals[2], vals[3]
            elif self.HeaderEnum.pp_part in line:
                vals = self._getValues(line, self.HeaderEnum.pp_part)
                self.pair_pp, self.pair_nn = vals[0], vals[1]
                self.pair_pn, self.pair    = vals[2], vals[3]
            elif self.HeaderEnum.Two_body in line:
                vals = self._getValues(line, self.HeaderEnum.Two_body)
                self.V_2B_pp, self.V_2B_nn = vals[0], vals[1]
                self.V_2B_pn, self.V_2B    = vals[2], vals[3]
            elif self.HeaderEnum.Full_H in line:
                vals = self._getValues(line, self.HeaderEnum.Full_H)
                self.E_HFB_pp, self.E_HFB_nn = vals[0], vals[1]
                self.E_HFB_pn, self.E_HFB    = vals[2], vals[3]
                
    def _getNumberNucleons(self, line):
        if self.HeaderEnum.Number_of_protons in line:
            vals = self._getValues(line, self.HeaderEnum.Number_of_protons)
            self.proton_numb, self.var_p = vals[0], vals[1]
        elif self.HeaderEnum.Number_of_neutrons in line:
            vals = self._getValues(line, self.HeaderEnum.Number_of_neutrons)
            self.neutron_numb, self.var_n = vals[0], vals[1]
    
    def _getAngularMomentum(self, line):
        if line == "Part \ No.     Z          N          A":
            return
        J_id = line[:7]
        line = line[6:]
        vals = self._getValues(line) 
        if self.HeaderEnum.Jx in J_id:
            self.Jx, self.Jx_2, self.Jx_var = vals[0], vals[1], vals[2]
        elif self.HeaderEnum.Jy in J_id:
            self.Jy, self.Jy_2, self.Jy_var = vals[0], vals[1], vals[2]
        elif self.HeaderEnum.Jz in J_id:
            self.Jz, self.Jz_2, self.Jz_var = vals[0], vals[1], vals[2]
        
    def _getPairCoupling(self, line):
        
        if self.HeaderEnum.PairT0J1 in line:
            vals = self._getValues(line, self.HeaderEnum.PairT0J1)
            self.P_T00_J10 = vals[1]
            self.P_T00_J1m1, self.P_T00_J1p1 = vals[0], vals[2]
        elif self.HeaderEnum.PairT1J0 in line:
            vals = self._getValues(line, self.HeaderEnum.PairT1J0)
            self.P_T10_J00 = vals[1]
            self.P_T1m1_J00, self.P_T1p1_J00 = vals[0], vals[2]            
    
    def setDataFromCSVLine(self, line_text):
        """ 
        Method to set all available attributes Dict Like 
        """
        elements = line_text.split(',')
        aux = [tuple(l.split(':')) for l in elements]
        aux = filter(lambda x: len(x)==2, aux)
        elements = dict(aux)
        # elements = dict([tuple(l.split(':')) for l in elements])
        for k, val in elements.items():
            
            k = k.strip()
            val = val.strip()
            if k in ('z','n','label_state'):
                if k == 'label_state' and val == 'None': val = 0
                setattr(self, k, int(val))
            elif k in ('properly_finished', 'broken_execution'):
                bool_ = val.lower() in ('true', 't', '1') # bool("len>0 str") = True
                setattr(self, k, bool_)
            elif k.startswith('_'):
                continue
            elif k.startswith('date_'):
                try:
                    setattr(self, k, datetime.strptime(val, self.FMT_DT))
                except ValueError:
                    self.FMT_DT = '%Y/%m/%d %H_%M_%S'
                    setattr(self, k, datetime.strptime(val, self.FMT_DT))
            else:
                try:
                    setattr(self, k, float(val))
                except ValueError:
                    setattr(self, k, None)
    
    @property
    def getAttributesDictLike(self):
        d1, d2, d3 = self.date_start, self.date_start_iter, self.date_end_iter
        #FMT = '%Y/%m/%d %H_%M_%S.%f'
        try:
            self.date_start      = datetime.strftime(self.date_start, self.FMT_DT)
            self.date_start_iter = datetime.strftime(self.date_start_iter, self.FMT_DT)
            self.date_end_iter   = datetime.strftime(self.date_end_iter, self.FMT_DT)
        except ValueError:
            self.FMT_DT = '%Y/%m/%d %H_%M_%S'
            self.date_start      = datetime.strftime(self.date_start, self.FMT_DT)
            self.date_start_iter = datetime.strftime(self.date_start_iter, self.FMT_DT)
            self.date_end_iter   = datetime.strftime(self.date_end_iter, self.FMT_DT)
        except TypeError:
            ## Dates are None (i.e. normal taurusvap) and therefore, no register
            ## is valid, append 
            aux = datetime.fromtimestamp(0)
            self.date_start      = datetime.strftime(aux, self.FMT_DT)
            self.date_start_iter = datetime.strftime(aux, self.FMT_DT)
            self.date_end_iter   = datetime.strftime(aux, self.FMT_DT)
                
        public_attr = filter(lambda a: not a[0].startswith('_'), self.__dict__.items())
        dict_ = ', '.join([k+' : '+str(v) for k,v in public_attr])
        self.date_start, self.date_start_iter, self.date_end_iter = d1, d2, d3
        
        return dict_
    
    @property
    def is_vap_calculation(self):
        return self._is_vap_calculation
    
    def isAxial(self, and_spherical=False):
        """ requires Ji=0 Jz^2=0 and beta20!=0"""
        if self.broken_execution or not self.properly_finished: return False
        
        # if not almostEqual(self.parity, 1, 1.e-5): return False
        TOL = 1.0e-5
        for l in range(1,5):
            mu_min = 0 if and_spherical else 1
            for mu in range(mu_min, l +1, 2):
                if not hasattr(self, f'b{l}{mu}_p'): continue
                b_lm = (getattr(self, f'b{l}{mu}_p'),
                        getattr(self, f'b{l}{mu}_n'))                           
                if abs(b_lm[0]) > TOL: return False
                if abs(b_lm[1]) > TOL: return False
        
        _properies = [ ]
        _properies.append(almostEqual(self.Jz_var , 0, TOL))
        ## Jz /= 0 in odd nuclei, 
        for i in ('x', 'y'):
            _properies.append(almostEqual(getattr(self,f'J{i}'), 0, TOL))
            if and_spherical:
                _properies.append(almostEqual(getattr(self,f'J{i}_2'), 0, TOL))
        if and_spherical:
            _properies.append(almostEqual(self.Jz_2, 0, TOL))
        
        return not False in _properies
    
    def setUpVoidResult(self, ldm_energies=False, energy0_vals=(0., 0., 0.)):
        """
        Set up a void 
        :ldm_energies : True set up kin-Ehfb from Liquid drop model, else set up from 0
        :energy0_vals : 3-len tuple for initial energies: (Kinetic, Pair, E HFB)
        """
        a = self.z + self.n
        self.broken_execution  = False
        self.properly_finished = True
        self.proton_numb  = float(self.z)
        self.neutron_numb = float(self.n)
        self.parity = 1.0
        self.overlap = 0.54321 if self._is_vap_calculation else 1.0
        # self.
        self.ho_b_length = 1.005 * (a**(1/6))
        self.var_p  = 0.0
        self.var_n  = 0.0
        for attr_ in InputTaurus.ConstrEnum.members():
            if attr_.startswith('b') and len(attr_) == 3:
                for t in ('_n', '_p', '_isoscalar', '_isovector'):
                    setattr(self, f"{attr_}{t}", 0.0)
                    setattr(self, f"{attr_.replace('b', 'q')}{t}", 0.0)
                    if attr_ == 'b20':
                        setattr(self, f'beta{t}', 0.0)
                    elif attr_ == 'b22':
                        setattr(self, f'gamma{t}', 0.0)
            elif attr_.startswith('P_T'):
                setattr(self, attr_, 0.0)
            elif attr_ == InputTaurus.ConstrEnum.sqrt_r2:
                for t in ('_n', '_p', '_isoscalar'):
                    setattr(self, f"r{t}", 1.2*(a**(1/3)))
                self.r_isovector = 0.0
                self.r_charge    = 1.05 * self.r_p
            elif attr_ in ("Jx", "Jy", "Jz"):
                for t in ("", "_2", "_var"):
                    setattr(self, f"{attr_}{t}", 0.0)
        self.E_zero = 0.0
        self.pair   = energy0_vals[1]
        if ldm_energies:
            ## estimation for the kinetic energy
            aux  = 9 * np.pi * ((3 / np.pi)**.3333) / (40 * 1.2**2)
            aux *= Constants.HBAR_C**2 / Constants.M_MEAN
            self.kin    = aux * (a + 2.2222 * ((self.z-self.n)**2 / a))
            self.E_HFB  = liquidDropModelBindingEnergy(self.z, self.n)  
        else:
            self.kin, self.E_HFB = 0.0, 0.0
        
        self.kin   += energy0_vals[0]
        self.E_HFB += energy0_vals[2]
        
        self.hf     = self.E_HFB - self.pair - self.kin
        self.V_2B   = self.hf + self.pair
        for t, v in (('_pp',self.z/a), ('_nn',self.n/a), ('_pn', 0.0)):
            setattr(self, f'pair{t}',  getattr(self, f'pair')  * v )
            setattr(self, f'hf{t}',    getattr(self, f'hf')    * v )
            setattr(self, f'E_HFB{t}', getattr(self, f'E_HFB') * v )
            setattr(self, f'V_2B{t}',  getattr(self, f'V_2B')  * v )
            if t != '_pn': setattr(self, f"kin{t[:2]}", self.kin   * v )
        
        
class DataTaurusPAV(_DataObjectBase):
    
    """ Abstract class with common methods """
    PROGRAM = 'taurus_pav.exe'
    DEFAULT_OUTPUT_FILENAME = 'aux_output_pav.OUT'
    EXPORT_LIST_RESULTS     = 'export_resultTaurus.txt'
    
    __message_endpav     = '                  PROJECTED MATRIX ELEMENTS'
    __message_components = 'All non-vanishing projected components'
    __message_sum_JP_components = 'Sum of projected components for J/P'
    __message_sum_KP_components = 'Sum of projected components for KJ/P'
    
    __message_properly_finished = 'This is the end, my only friend, the end.'
    
    class HeaderEnum(Enum):
        """ Enumerate for the line headers of every argument to process"""
        label_states = 'Label of state '
    
    def __init__(self, z, n, filename, empty_data=False):
        
        self.z = z
        self.n = n
        self.properly_finished = False
        self.broken_execution  = False
        self._nanComponentsInResults = False
        self._filename = filename
        self.label_states = (0, 0)
        
        self.J  = []
        self.MJ = []
        self.KJ = []
        self.P  = []
        self.proj_norm   = []
        self.proj_energy = []
        self.proj_Z = []
        self.proj_N = []
        self.proj_A = []
        self.proj_J = []
        self.proj_Jz= []
        self.proj_P = []
        self.proj_T = []
        self.proj_Tz= []
        
        self.sum_JP_norm   = []
        self.sum_JP_energy = []
        
        self.sum_KP_norm   = []
        self.sum_KP_energy = []
        
        if not empty_data:
            try:
                self.get_results()
            except Exception as e:
                if isinstance(e, ValueError):
                    self._nanComponentsInResults = True
                printf(" (TPC)>> EXCEPTION from Taurus PAV Constructor >> last 5 lines::",
                       LINE_2)
                if os.path.exists(self._filename):
                    with open(self._filename, 'r') as f:
                        printf("".join(f.readlines()[-5:]), LINE_2)
                else:
                    printf(f"   [SUPER ERR] File [{self._filename}] not found")
                printf(" (TPC)>> resEXCEPTION from Taurus PAV Constructor >> self::")
                printf(self)
                printf(" (TPC)>> exception:: ", e, "<<(TPC)")
                printf(" (TPC)<< EXCEPTION from Taurus PAV Constructor <<<<<<<< ")
    
    def __str__(self):
        aux = OrderedDict(sorted(self.__dict__.items(), key=lambda t: t[0]))
        return "\n".join(k+' :\t'+str(v) for k,v in aux.items())
    
    @property
    def nanComponentsInResults(self):
        return self._nanComponentsInResults
    
    @property
    def dim(self):
        return self.proj_energy.__len__()
    
    def get_results(self):
        """
        Read all the lines and export
            For DataTaurusPAV, only quasiparticle states are saved:
        Note:
            In the case of VAP calculation, the stored energies are the projected 
            ones, the other observables will be from quasiparticles.
            It assign the values twice (QP values appear in second place)
        """
        with open(self._filename, 'r') as f:
            data = f.read()
            data = data.split('\n')
        if len(data) < 2 or not data[-2] == self.__message_properly_finished:
            self.properly_finished = False
            self.broken_execution  = True
            return           
        
        self._ignorable_block = True
        self._allNVcomp_block = False
        self._sumJPcomp_block = False
        self._sumKPcomp_block = False
        
        self._projecting_JMK  = True
        self._projecting_P    = True
        skip_ = 0   # Skipping index, jump from the Header_block to the 1st line
        for indx, line in enumerate(data[99:]):
            if self._ignorable_block:
                if line.startswith(self.__message_endpav):
                    self._ignorable_block  = False
                    self.properly_finished = True
                elif self.HeaderEnum.label_states in line:
                    self.label_states = tuple([int(x) for x in line.split()[-2:]])
                continue
            else:
                if self._sumKPcomp_block and (line.startswith("%%%")):
                    self.properly_finished = len(self.proj_energy) > 0
                    ## complementary files part, stop
                    return
            
            if skip_ > 0:
                skip_ -= 1
                continue
            elif len(line) == 0:
                continue
            else:
                if   line == self.__message_components:
                    self._allNVcomp_block = True
                    skip_ = 6
                    continue
                elif line == self.__message_sum_JP_components:
                    self._allNVcomp_block = False
                    self._sumJPcomp_block = True
                    skip_ = 4
                    continue
                elif line == self.__message_sum_KP_components:
                    self._sumJPcomp_block = False
                    self._sumKPcomp_block = True
                    skip_ = 4
                    continue
            
            if not self._allNVcomp_block and '****' in line: # Line fails (i.e. J < K blocked)
                printf("[WRN]!, excluded line ", line)
                continue
        
            if   self._allNVcomp_block:
                self._getAllNVComponents_blockline(line)
            elif self._sumJPcomp_block:
                self._getSumJPComponents_blockline(line)
            elif self._sumKPcomp_block:
                self._getSumKPComponents_blockline(line)
            else:
                continue
    
    def _fixingNanValuesInLine(self, line):
        """
        In case of divergences or malfuncitoning, setting all possible numbers
        for each block.
        """
        pass
    
    def _getAllNVComponents_blockline(self, line0):
        """
        Get all values depending on the header, 
        """
        headers = [line0[i:i+5].strip() for i in range(0, 19, 5)]
        
        line = line0[20:].split()
        ## Change to assing norm=0 values (normally with divergences or 1e-150)
        if '***' in line[0]:
            printf(f"[WRN]! excluded line (AllNVComponents) '****'  in norm",
                   f":: [{line0}]")
            return 
        ## NOTE: states with norm ******* give problems, results close to 0 with
        ##  divergences are well identified in the process, putting everithing 
        ##  to 0 could mislead the exclusion process.
        # if abs(float(line[0])) < 1.0e-8: line = [0.000 for i in line]
        
        #     1     |      E     |     Z       v |     N       v |     A       v |
        #    J    |    Jz     v |    P    |    T    |    Tz     v
        
        # _num_indx_val = (3, 5, 7, 10, 14) ## index for the diff exponents E-**
        _attrs_and_index = {
            'proj_norm' :  0, 'proj_energy' :  1,
            'proj_Z'    :  2, 'proj_N'      :  4,   'proj_A'  : 6,
            'proj_J'    :  8, 'proj_Jz'     :  9,   'proj_P'  : 11,
            'proj_T'    : 12, 'proj_Tz'     : 13,
        }
        _accepted_results = {} # attrs and vals
        for attr_, i in _attrs_and_index.items():
            ## Cast valid invalid exceptions for the foat values (Nan, ***, ...)
            valid = True
            if '****' in line[i]:
                if i in (0, 1): valid = False
                else:
                    aux_str = line[i].replace('*', '')
                    ## NOTE: this will exclude, invalid vars
                    if aux_str.__len__() == 0: valid = False
                    else:
                        line[i] = aux_str
                        line.insert(i+1, 0) # for the next indexes to be in place
            else:
                try: float(line[i]) 
                except ValueError as _:
                    if   i in (0, 1, 8, 11, 12): 
                        valid = False
                    elif attr_ in ('proj_Z',  'proj_N', 'proj_A'):
                        _valid_re = (r'[-]\d\.\d{7}[-+]\d{3}', r'\d\.\d{7}[-+]\d{3}')
                    elif attr_ in ('proj_Jz', 'proj_Tz'):
                        _valid_re = (r'[-]\d\.\d{5}[-+]\d{3}', r'\d\.\d{5}[-+]\d{3}')
                    
                    if valid:
                        _matches = [re.match(re_, line[i]) for re_ in _valid_re]
                        if any(_matches):
                            aux_str, mantis = line[i][:-4], line[i][-4:]
                            if '+' in mantis:
                                ## Its not possible to have variance exponents > 0 
                                valid = False
                            line[i] = aux_str
                            line.insert(i+1, mantis) # for the next indexes to be in place
                        else:
                            valid = False
            if valid:
                _accepted_results[attr_] = float(line[i])
            else:
                printf(f"[WRN]! excluded line (AllNVComponents) ATTR/val_",
                           attr_, line[i], f":: [{line}]")
                return
        
        ## Append the line (all correct)
        for attr_, val in _accepted_results.items(): 
            getattr(self, attr_).append(val)
        
        #  ***  Doit after in case the line is ommited ------------------------
        if   headers[0] == '': self._projecting_JMK = False
        else:
            self.J .append(int(headers[0]))
            self.MJ.append(int(headers[1]))
            self.KJ.append(int(headers[2]))
            
        if headers[3] == '': self._projecting_P   = False
        else:                self.P.append(int(headers[3]))
        # ---------------------------------------------------------------------
    
    def _getSumJPComponents_blockline(self, line):
        """ Block for J components"""
        if line.startswith("    Total"): 
            return
        j, p = None, None
        try:
            if self._projecting_JMK: j = int(line[0: 5])
            if self._projecting_P:   p = int(line[5:10])
            vals = [float(x) for x in line[10:].split()]
        except ValueError as ve:
            ## There is a problem with the block line, skipping
            return
        
        self.sum_JP_norm  .append(vals[0])
        self.sum_JP_energy.append(vals[2])
        if abs(vals[1]) > 2.e-7: printf("[ERROR]Imaginary part of 1(J,P)=", j, p)
        if abs(vals[3]) > 2.e-7: printf("[ERROR]Imaginary part of E(J,P)=", j, p)
    
    def _getSumKPComponents_blockline(self, line):
        """ Block for K components"""
        if line.startswith("    Total"): 
            return
        j, p = None, None
        try:
            if self._projecting_JMK: j = int(line[0:5])
            if self._projecting_P:   p = int(line[5:10])
            vals = [float(x) for x in line[10:].split()]
        except ValueError as ve:
            ## There is a problem with the block line, skipping
            return
        
        self.sum_KP_norm  .append(vals[0])
        self.sum_KP_energy.append(vals[2])
        if abs(vals[1]) > 2.e-7: printf("[ERROR]Imaginary part of 1(K,P)=", j, p)
        if abs(vals[3]) > 2.e-7: printf("[ERROR]Imaginary part of E(K,P)=", j, p)
    

class DataTaurusMIX(_DataObjectBase):
    
    """ 
    Class for processing the J_ files from taurus_mix/ program,
        ## NOTES: 
            * designed for the github version.
            * only process the energy  J / Pi final lines (does not track 
                filtering - cutoffs info)
    """
    PROGRAM = 'taurus_mix.exe'
    DEFAULT_OUTPUT_FILENAME = 'aux_output.dat'
    EXPORT_LIST_RESULTS     = 'export_resultTaurus.txt'
    
    __MSG_INPUT_PARAMS    = '                      INPUT PARAMETERS'
    __MSG_SELECTED_STATES = 'States selected'
    __MSG_NATURAL_STATES  = 'Natural states'
    __MSG_ENERGY_EIGENVEC = 'Energy eigenvectors'
    __MSG_ENERGY_SPECTRUM = '                       ENERGY SPECTRUM'
        
    class HeaderEnum(Enum):
        """ Enumerate for the line headers of every argument to process"""
        pass
    
    class ArgsEnum(Enum):
        """ Different variable names for (all) the lists in the arguments """
        i = 'i'
        label = 'label'
        JP = 'JP'
        K = 'K'
        overlap = 'overlap'
        norm_eigenvalues = 'norm_eigenvalues'
        energy = 'energy'
        par_avg = 'par_avg'
        Z_avg  = 'Z_avg'
        N_avg  = 'N_avg'
        A_avg  = 'A_avg'
        J_avg  = 'J_avg'
        T_avg  = 'T_avg'
        Qs_avg = 'Qs_avg'
        mu_avg = 'mu_avg'
        r_p_avg = 'r_p_avg'
        r_n_avg = 'r_n_avg'
        r_m_avg = 'r_m_avg'
        r_ch_avg = 'r_ch_avg'
    
    def __init__(self, filename, empty_data=False):
        
        self.properly_finished = False
        self.broken_execution  = False
        self.empty_result      = True
        self._filename = filename
        
        self.J  = None
        self.P  = None
        self.z  = None  # will be processed from input files
        self.n  = None
        
        self.selected_states = {
            self.ArgsEnum.i     : [], 
            self.ArgsEnum.label   : [],
            self.ArgsEnum.K       : [],
            self.ArgsEnum.overlap : [],
            self.ArgsEnum.energy  : [],
            self.ArgsEnum.par_avg  : [],
            self.ArgsEnum.Z_avg : [],
            self.ArgsEnum.N_avg : [], 
            self.ArgsEnum.A_avg : [],
            self.ArgsEnum.J_avg : [],
            self.ArgsEnum.T_avg : [],    
        }
        self.natural_states = {
            self.ArgsEnum.i     : [], 
            self.ArgsEnum.norm_eigenvalues : [],
            self.ArgsEnum.energy  : [],
            self.ArgsEnum.par_avg  : [],
            self.ArgsEnum.Z_avg : [],
            self.ArgsEnum.N_avg : [], 
            self.ArgsEnum.A_avg : [],
            self.ArgsEnum.J_avg : [],
            self.ArgsEnum.T_avg : [],    
        } 
        self.energy_eigenvectors = deepcopy(self.natural_states)
        del self.energy_eigenvectors[self.ArgsEnum.norm_eigenvalues]
        
        self.energy_spectrum = deepcopy(self.energy_eigenvectors)
        self.energy_spectrum[self.ArgsEnum.JP] = []
        self.energy_spectrum[self.ArgsEnum.Qs_avg] = []
        self.energy_spectrum[self.ArgsEnum.mu_avg] = []
        self.energy_spectrum[self.ArgsEnum.r_p_avg] = []
        self.energy_spectrum[self.ArgsEnum.r_n_avg] = []
        self.energy_spectrum[self.ArgsEnum.r_m_avg] = []
        self.energy_spectrum[self.ArgsEnum.r_ch_avg] = []
        
        self._lines_energy_spectrum = ""
        
        if not empty_data:
            try:
                self.get_results()
            except Exception as e:
                printf(" (TMC)>> EXCEPTION from Taurus MIX Constructor >> last 5 lines::",
                       LINE_2)
                with open(self._filename, 'r') as f:
                    printf("".join(f.readlines()[-5:]), LINE_2)
                printf(" (TMC)>> EXCEPTION from Taurus MIX Constructor >> self::")
                printf(self)
                printf(" (TMC)>> exception:: ", e, "<<(TMC)")
                printf(" (TMC)<< EXCEPTION from Taurus MIX Constructor <<<<<<<< ")
    
    def __str__(self):
        aux = OrderedDict(sorted(self.__dict__.items(), key=lambda t: t[0]))
        return "\n".join(k+' :\t'+str(v) for k,v in aux.items())
    
    def get_results(self):
        """
        Read all the lines and export
            For DataTaurusPAV, only quasiparticle states are saved:
        Note:
            In the case of VAP calculation, the stored energies are the projected 
            ones, the other observables will be from quasiparticles.
            It assign the values twice (QP values appear in second place)
        """
        with open(self._filename, 'r') as f:
            data = f.read()
            data      = data.split('\n')
        
        self._input_part   = True
        self._sel_sts_part = False
        self._nat_sts_part = False
        self._norm_ev_part = False
        self._spectra_part = False
        
        for indx_, line in enumerate(data):
            if line == '' or line.startswith('--') or line.startswith('=='):
                continue
            
            if self._input_part:
                if line.startswith(self.__MSG_SELECTED_STATES):
                    self._input_part   = False
                    self._sel_sts_part = True
                    continue
                self._input_part_processing(line)
            elif self._sel_sts_part:
                if line.startswith(self.__MSG_NATURAL_STATES):
                    self._sel_sts_part = False
                    self._nat_sts_part = True
                    continue
                self._sel_sts_part_processing(line)
            elif self._nat_sts_part:
                if line.startswith(self.__MSG_ENERGY_EIGENVEC):
                    self._nat_sts_part = False
                    self._norm_ev_part = True
                    continue
                self._nat_sts_part_processing(line)
            elif self._norm_ev_part:
                if line.startswith(self.__MSG_ENERGY_SPECTRUM):
                    self._norm_ev_part = False
                    self._spectra_part = True
                    continue
                self._norm_ev_part_processing(line)
            else:
                self._spectra_part_processing(line)
        
    def _input_part_processing(self, line):
        if   line.startswith('Number of active protons  Z'):
            self.z = int(line.split()[-1])
        elif line.startswith('Number of active neutrons N'):
            self.n = int(line.split()[-1])
        elif line.startswith('Angular momentum min(2*J)'):
            self.J = int(line.split()[-1])
        elif line.startswith('Parity min(P)'):
            self.P = int(line.split()[-1])
    
    def _sel_sts_part_processing(self, line):
        line = line.strip().split()
        
        if len(line) == 11:
            if not line[0].isdigit():return
            attrs_ = [
                self.ArgsEnum.i, self.ArgsEnum.label, self.ArgsEnum.K,
                self.ArgsEnum.overlap, self.ArgsEnum.energy, 
                self.ArgsEnum.par_avg, self.ArgsEnum.Z_avg, self.ArgsEnum.N_avg, 
                self.ArgsEnum.A_avg, self.ArgsEnum.J_avg, self.ArgsEnum.T_avg
            ]
            for indx_, attr_ in enumerate(attrs_):
                if indx_ < 3:
                    self.selected_states[attr_].append(  int(line[indx_]))
                else:
                    if '**' in line[indx_]: line[indx_] = -9999999
                    self.selected_states[attr_].append(float(line[indx_]))
    
    def _nat_sts_part_processing(self, line):
        line = line.strip().split()
        
        if len(line) == 9:
            if not line[0].isdigit():return
            attrs_ = [
                self.ArgsEnum.i, self.ArgsEnum.norm_eigenvalues, 
                self.ArgsEnum.energy, self.ArgsEnum.par_avg, 
                self.ArgsEnum.Z_avg, self.ArgsEnum.N_avg, self.ArgsEnum.A_avg,
                self.ArgsEnum.J_avg, self.ArgsEnum.T_avg
            ]
            for indx_, attr_ in enumerate(attrs_):
                if indx_ == 0:
                    self.natural_states[attr_].append(  int(line[indx_]))
                else:
                    if '**' in line[indx_]: line[indx_] = -9999999
                    self.natural_states[attr_].append(float(line[indx_]))
    
    def _norm_ev_part_processing(self, line):
        line = line.strip().split()
        
        if len(line) == 8:
            if not line[0].isdigit():return
            attrs_ = [
                self.ArgsEnum.i, self.ArgsEnum.energy, self.ArgsEnum.par_avg, 
                self.ArgsEnum.Z_avg, self.ArgsEnum.N_avg, self.ArgsEnum.A_avg,
                self.ArgsEnum.J_avg, self.ArgsEnum.T_avg
            ]
            for indx_, attr_ in enumerate(attrs_):
                if indx_ == 0:
                    self.energy_eigenvectors[attr_].append(  int(line[indx_]))
                else:
                    if '**' in line[indx_]: line[indx_] = -9999999
                    self.energy_eigenvectors[attr_].append(float(line[indx_]))
    
    def _spectra_part_processing(self, line0):
        line = line0.strip().split()
        
        if len(line) == 17:
            if not (line[0].isdigit() or  '/' in line[0]): return
            self._lines_energy_spectrum += line0 + '\n'
            
            attrs_ = [
                self.ArgsEnum.JP, self.ArgsEnum.JP, self.ArgsEnum.i, 
                self.ArgsEnum.energy,  None, 
                self.ArgsEnum.Qs_avg,  self.ArgsEnum.mu_avg, 
                self.ArgsEnum.r_p_avg, self.ArgsEnum.r_n_avg, 
                self.ArgsEnum.r_m_avg, self.ArgsEnum.r_ch_avg,
                self.ArgsEnum.par_avg, 
                self.ArgsEnum.Z_avg, self.ArgsEnum.N_avg, self.ArgsEnum.A_avg,
                self.ArgsEnum.J_avg, self.ArgsEnum.T_avg
            ]
            for indx_, attr_ in enumerate(attrs_):
                if attr_ == self.ArgsEnum.JP:
                    if indx_ == 0:
                        self.energy_spectrum[attr_].append(line[indx_])
                    else:
                        self.energy_spectrum[attr_][-1] += " " + line[indx_]
                elif indx_ == 4:
                    continue
                else:
                    if '**' in line[indx_]: line[indx_] = -9999999
                    self.energy_spectrum[attr_].append(float(line[indx_]))
    
    def getSpectrumLines(self):
        return self._lines_energy_spectrum

class CollectiveWFData(_DataObjectBase):
    
    DEFAULT_OUTPUT_FILENAME = 'collective_wavefunction.txt'
    INPUT_FILENAME  = DataTaurusMIX.INPUT_FILENAME
    BU_folder       = None
    BU_fold_constr  = None
    EXPORT_LIST_RESULTS = None
    PROGRAM         = 'taurus_mix.exe'
    
    def __init__(self, filename, empty_data=False):
        
        self.properly_finished = False
        self.broken_execution  = False
        self._nanComponentsInResults = False
        self._filename = filename
        
        self.sigmas   = []
        self.indexes  = {}
        self.labels   = {}
        self.g_values = {}
        self.g2values = {}
        self.norm_g2_total = {}
        
        if not empty_data:
            try:
                self.getResults()
            except Exception as e:
                if isinstance(e, ValueError):
                    self._nanComponentsInResults = True
                printf(" (TCW)>> EXCEPTION from Collective-HWG WF >> last 5 lines::",
                       LINE_2)
                if os.path.exists(self._filename):
                    with open(self._filename, 'r') as f:
                        printf("".join(f.readlines()[-5:]), LINE_2)
                else:
                    printf(f"   [SUPER ERR] File [{self._filename}] not found")
                printf(" (TCW)>> resEXCEPTION from Collective-HWG WF >> self::")
                printf(self)
                printf(" (TCW)>> exception:: \n", e, "\n<<(TCW)")
                printf(" (TCW)<< EXCEPTION from Collective-HWG WF <<<<<<<< ")
    
    def getResults(self):
        
        with open(self._filename, 'r') as f:
            data = f.read()
            data = data.split('\n')
        if len(data) < 2:
            self.properly_finished = False
            self.broken_execution  = True
            return
        
        current_sigma = 0
        for line in data[2:]:
            line = line.strip()
            if not line or ('----' in line): continue
            
            line = line.split()
            if len(line) == 1:
                self.norm_g2_total[current_sigma] = float(line[0])
            else:
                s, i, l, g, g2 = line
                s, i, l = int(s), int(i), int(l)
                g, g2 = float(g), float(g2)
                
                if s > current_sigma:
                    current_sigma = s
                    self.sigmas.append(s)
                    self.indexes [s] = [i, ]
                    self.labels  [s] = [l, ]
                    self.g_values[s] = [g, ]
                    self.g2values[s] = [g2,]
                else:
                    self.indexes [current_sigma].append(i)
                    self.labels  [current_sigma].append(l)
                    self.g_values[current_sigma].append(g)
                    self.g2values[current_sigma].append(g2)


class OccupationsHWGData(_DataObjectBase):
    """
    Evaluation of final occupations corresponding to shell-orbital states
    for each excited state for the HWG of angular-mom and parity: ¨J^P
    """
    
    DEFAULT_OUTPUT_FILENAME = 'occupation_numbers.txt'
    INPUT_FILENAME  = DataTaurusMIX.INPUT_FILENAME
    BU_folder       = None
    BU_fold_constr  = None
    EXPORT_LIST_RESULTS = None
    PROGRAM         = 'taurus_mix.exe'
    
    def __init__(self, filename, empty_data=False):
        
        self.properly_finished = False
        self.broken_execution  = False
        self._nanComponentsInResults = False
        self._filename = filename
        
        self.sigmas   = []
        self.labels   = []
        self.occupations_protons   = {}
        self.occupations_neutrons  = {}
        self.relative_occ_protons  = {}
        self.relative_occ_neutrons = {}
        
        if not empty_data:
            try:
                self.getResults()
            except Exception as e:
                if isinstance(e, ValueError):
                    self._nanComponentsInResults = True
                printf(" (TON)>> EXCEPTION from OccupationNubers-HWG WF >> last 5 lines::",
                       LINE_2)
                if os.path.exists(self._filename):
                    with open(self._filename, 'r') as f:
                        printf("".join(f.readlines()[-5:]), LINE_2)
                else:
                    printf(f"   [SUPER ERR] File [{self._filename}] not found")
                printf(" (TON)>> resEXCEPTION from OccupationNubers-HWG WF >> self::")
                printf(self)
                printf(" (TON)>> exception:: \n", e, "\n<<(TCW)")
                printf(" (TON)<< EXCEPTION from OccupationNubers-HWG WF <<<<<<<< ")
    
    def getResults(self):
        
        with open(self._filename, 'r') as f:
            data = f.read()
            data = data.split('\n')
        if len(data) < 2:
            self.properly_finished = False
            self.broken_execution  = True
            return
        
        current_sigma = 0
        for line in data[2:]:
            line = line.strip()
            if not line or ('sum' in line): continue
            
            line = line.split()
            if len(line) == 1:
                self.norm_g2_total[current_sigma] = float(line[0])
            else:
                s, i, _, _, j, l, o_p, o_n = line
                s, i, l, j = int(s), int(i), int(l), int(j)
                o_p, o_n = float(o_p), float(o_n)
                deg_ = (j + 1)
                
                if not l in self.labels: self.labels.append(l)
                    
                if s > current_sigma:
                    current_sigma = s
                    self.sigmas.append(s)
                    self.occupations_protons  [s] = [o_p, ]
                    self.occupations_neutrons [s] = [o_n, ]
                    self.relative_occ_protons [s] = [o_p / deg_,]
                    self.relative_occ_neutrons[s] = [o_n / deg_,]
                else:
                    self.occupations_protons  [current_sigma].append(o_p)
                    self.occupations_neutrons [current_sigma].append(o_n)
                    self.relative_occ_protons [current_sigma].append(o_p / deg_)
                    self.relative_occ_neutrons[current_sigma].append(o_n / deg_)

class _TestingTaurusOutputGenerator():
    """
    Windows purpose, testing.
        Generate the output file from taurus from dataTaurus info.
        * Properly finished/ uncompleted / Broken Results
        * If inputfile InputTaurus given, create the equivalent result compatible,
        with constraints given.
    """
    
    _GOOD_RESULT_TMP   = "data_resources/testing_files/template_taurus_vap_generator.txt"
    _BROKEN_RESULT_TMP = ""
    
    __line_grad_tmp = "{i: >6}      {gr:7.5E}{e: >14.6f}{z: >12.6f}{var_z: >12.6f}{n: >12.6f}{var_n: >12.6f}"
    
    __header_qp = "               QUASIPARTICLE STATE PROPERTIES"
    __header_proj = "                 PROJECTED STATE PROPERTIES"
    
    __windows_madeUpDT : DataTaurus = None
    __keep_axial = None
    
    @classmethod
    def _update_KeepAxial(cls, keep_axial):
        if cls.__keep_axial == None:
            cls.__keep_axial = keep_axial
            cls._keep_axial  = keep_axial
    
    def __init__(self, res_obj, case_ok=True, case_broken=False, 
                 keep_axial=False):
        """
        :res_obj: template for the file generator:
            <InputTaurus> : Fix the argumets from the constraints given, randomize others
            <DataTaurus>  : Get the file from a DataTaurus object was created.
        :case_ok:     True properly finished, False maximum iterations achieved
        :case_broken: True: return a broken result (no final iteration achieved)
        """
        self._input : InputTaurus = None
        self._data  : DataTaurus  = None
        
        self._text = ''
        assert not(case_ok and case_broken), "cannot be both broken and ok" 
        self._case_ok     = case_ok
        self._case_broken = case_broken
        
        self._update_KeepAxial(keep_axial)
        
        self.K = 0
        self.constraints = [] # for setting up the minimum energy
        self.minimum_def = [] # for setting up the minimum energy
        self.z = res_obj.z
        self.n = res_obj.n
        self.A = res_obj.z + res_obj.n
        self._E_kpt_var = [0, 0, 0]  ## Energies variable for Kinetic/Pair/Tot
        
        self._iter_msg  = ''
        self._iter_block_lines = ''
        
        dt0 = datetime.now()
        dt1 = dt0 + timedelta(seconds=10, milliseconds=10)
        dt2 = dt1 + timedelta(hours=1, minutes=15, seconds=33, milliseconds=123)
                
        fmtdt = [_d.strftime("%Y/%m/%d %H_%M_%S.%f")[:-3] for _d in (dt0, dt1, dt2)]
        
        self.time_start      = fmtdt[0]
        self.time_start_iter = fmtdt[1]
        self.time_end_iter   = fmtdt[2]
        self._iter_final     = 100
        
        self._build_from_input_obj = isinstance(res_obj, InputTaurus)
        if   isinstance(res_obj, DataTaurus):
            self._input = InputTaurus(res_obj.z, res_obj.n, 'hamil')
            self._data  = res_obj
        elif isinstance(res_obj, InputTaurus):
            self._input = res_obj
        else:
            raise BaseException("Invalid argument type given, got", res_obj)
    
    @classmethod
    def _update_madeUpDT(cls, data_obj : DataTaurus):
        cls.__windows_madeUpDT = data_obj
        
    def setUpOutput(self, constraints = [], minimum_def = [], K=0, randomize=False):
        """
        Set up certain properties such as K or the energy as a function of 
        constraints.
        """
        if self._keep_axial: self.K = K
        if not isinstance(constraints, list): constraints = [constraints, ]
        if not isinstance(minimum_def, list): minimum_def = [minimum_def, ]
        
        self.constraints = constraints
        self.minimum_def = minimum_def
        self.randomizeEhfb = randomize
        
        if not all(self.minimum_def):
            ## Cases where we build the 
            self._energy_from_constraints()
            
            dat = DataTaurus(self.z, self.n, None, empty_data=True)
            dat.setUpVoidResult(ldm_energies=True, energy0_vals=self._E_kpt_var)
            
            self._update_madeUpDT(dat)
            
            ## fix the randomized variables for constraining.
            for ic, cnst in enumerate(self.constraints):
                if self.constraints == []: break
                cns_str = f'{cnst}_isoscalar' if not(cnst[0] in 'PJ') else cnst
                setattr(dat, cns_str, self.minimum_def[ic])
        else:
            ## Class attribute storaged (non base case) avoid fixing all the vars
            dat = deepcopy(self.__windows_madeUpDT)
            
            self._energy_from_constraints()
            ## fix the modified energies
            dat.pair  += self._E_kpt_var[1]
            dat.E_HFB += self._E_kpt_var[2]
            dat.hf     = dat.E_HFB - dat.pair - dat.kin
            dat.V_2B   = dat.hf + dat.pair
            
            A = self.A
            for t, v in (('_pp',self.z/A), ('_nn',self.n/A), ('_pn', 0.0)):
                setattr(dat, f'pair{t}',  getattr(dat, f'pair')  * v )
                setattr(dat, f'hf{t}',    getattr(dat, f'hf')    * v )
                setattr(dat, f'E_HFB{t}', getattr(dat, f'E_HFB') * v )
                setattr(dat, f'V_2B{t}',  getattr(dat, f'V_2B')  * v )
                if t != '_pn': setattr(self, f"kin{t[:2]}", dat.kin   * v )
            
            for ic, cnst in enumerate(self.constraints):
                if self.constraints == []: break
                cns_str = f'{cnst}_isoscalar' if not(cnst[0] in 'PJ') else cnst
                setattr(dat, cns_str, getattr(self._input, cnst))
        
        # Update label, and case-execution, update self._data object ot print
        dat.broken_execution  = self._case_broken
        dat.properly_finished = self._case_ok
        dat.label_state = int(abs(dat.E_HFB) * 1.0e+12)
        self._data = dat
        
        ## Iteration section now with the final Ehfb
        self._get_iteration_block()
        
        if self._build_from_input_obj:
            self._setAuxiliaryDataTaurusFromInputDT()
            self.setTemplateArgs_fromInputTaurus()
        else:
            self.setTemplateArgs_fromDataTaurus()
    
    def _energy_from_constraints(self):
        """ 
        Auxiliary function to get an additional energy from the constraints as
        a parabolic shaped in the minimum for the "base_Deformation"
        - If not base deformation, this energy swap is randomized and append
            it to the base energy.
        """
        for ic, cnstr in enumerate(self.constraints):
            x_0 = self.minimum_def[ic]
            if x_0 == None:
                x_0 = np.random.random()
                x   = 0.0 # not fixed for a base calculation
            else: 
                x = getattr(self._input, cnstr, None)
                x = 0.0 if (x == None) else x  
            
            if cnstr.startswith('P_T'):
                self._E_kpt_var[1]  += (x - x_0)**2
            
            rand_ener = np.random.random()* 2 if self.randomizeEhfb else 0
            if self.randomizeEhfb: 
                _ = 0
            ## lets asume for the x_0 for being the minimum
            self._E_kpt_var[2]  += (x - x_0)**2 + rand_ener
            self.minimum_def[ic] = x_0
    
    def _get_iteration_block(self):
        """
        Get the progression of the minimization progress up to the final energy.
        
        NOTE: Call it after the set-energy from constraints to have fixed E-HFB
        """
        grad_min = self._input.grad_tol
        i_max    = self._input.iterations
        grad_0, e_0, e_f  = 100, 100.0, self._data.E_HFB
        var = .0
        z, n = self.z, self.n
        
        if self._case_ok or self._case_broken:
            i_max = i_max // 2
        if not self._case_ok:
            grad_min = grad_min * 10
        self._iter_final = i_max
        
        x = -1 * np.log10(grad_min) / (i_max-1)
        y = -1 * np.log10(abs(e_f)*grad_min) / (i_max-1)
        grad_vals = [grad_0     *(10**(-x*i)) for i in range(i_max)]
        ener_vals = [(e_0)*(10**(-y*i)) + e_f for i in range(i_max)]
        lines = []
        for i in range(i_max):
            kwargs = {
                'i':i, 'z':z, 'n':n, 'var_z':var, 'var_n':var, 
                'e':ener_vals[i], 'gr':grad_vals[i]
            }
            lines.append(self.__line_grad_tmp.format(**kwargs))
            if kwargs['gr'] < self._input.grad_tol: break
        
        self._iter_block_lines = '\n'.join(lines)
        
        if self._case_broken:
            self._iter_msg = ''
        else:
            if self._case_ok:
                self._iter_msg = 'Calculation converged'
            else:
                self._iter_msg = 'Maximum number of iterations reached'            
    
    def _setAuxiliaryDataTaurusFromInputDT(self):
        """
        Define constrained values in the result object and randomize some of the
        unconstrained observables.
        """
        dat = self._data    # defined in SetUpOutput()
        
        if self._keep_axial:
            dat.Jz, dat.Jz_2, dat.Jz_var = self.K/2, self.K**2/4, 0.0
        
        A = self.z + self.n
        for cnst in self._input.ConstrEnum.members():
            val = getattr(self._input, cnst, None)
            if val == None: 
                ## randomize for pairs, J and r
                if cnst.startswith('b'): 
                    l,m = int(cnst[1]), int(cnst[2])
                    if (m % 2) and m>0: continue
                if self._keep_axial: continue
                
                v_p, v_n = np.random.random(), np.random.random()
                val = (v_p + v_n) / 2
            elif isinstance(val, tuple):
                v_p, v_n = val
            else:
                v_p, v_n = val/2, val/2
            
            if cnst.startswith('b') or  cnst.startswith('r'):
                aux = {'_p': v_p, '_n':v_n, '_isoscalar':v_p + v_n, 
                       '_isovector':v_p - v_n}
                for t, value in aux.items():
                    if cnst.startswith('r') and t.startswith('_iso'): value /= 2
                    setattr(dat, f"{cnst}{t}", value)
                    if cnst.startswith('b'):
                        q_cnst = f"{cnst.replace('b', 'q')}{t}"
                        l = int(cnst[1])
                        setattr(dat, q_cnst, getQLMvalueFromBetaLM(l,A,value))
                if cnst.startswith('r'): dat.r_charge = v_p * 1.05
                
            elif cnst.startswith('P_T'): setattr(dat, cnst, val)
            elif cnst.startswith('J'):
                if cnst == InputTaurus.ConstrEnum.Jz: continue
                rand = np.random.random()
                setattr(dat, cnst, val)
                setattr(dat, f"{cnst}_2",   (val*rand)**2)
                setattr(dat, f"{cnst}_var", (val*rand)**2 - val**2)
            
            aux = 4*np.pi/(3*A*(1.2*(A**0.3333))**2)
            for t in ('_p', '_n', '_isoscalar', '_isovector'):
                q20, q22 = getattr(dat,f'q20{t}'), getattr(dat,f'q22{t}')
                setattr(dat, f'beta{t}', aux*(  q20**2 + 2*(q22**2)) )
                setattr(dat, f'gamma{t}',np.arctan2(2**.5 * q22, q20))
        
        ## Fix the variable-constrained values (no randomized)
        for ic, cnst in enumerate(self.constraints):
            cnst2 = cnst if cnst[0] in 'PJ' else f"{cnst}_isoscalar"
            cnst_vals = getattr(self._input, cnst, None)
            if cnst_vals == None: ## Unconstrained, base calculation 
                setattr(dat, cnst2, self.minimum_def[ic])
            else:
                setattr(dat, cnst2, cnst_vals)
        
        self._data = dat
    
    def setTemplateArgs_fromDataTaurus(self):
        ## self._input ! fix this from data
        raise BaseException("Implement me!")
    
    def setTemplateArgs_fromInputTaurus(self):        
        """
        Compose the final output file for InputArgs / Gradient / Results.
        """
        txt_inp    = self._input.getText4file()
        txt_inp_dd = self._input.get_inputDDparamsFile(10, 15)
        
        text_results_1 = ''
        other_kwargs   = {
            'jtot': self._data.Jx_2+self._data.Jy_2+self._data.Jz_2,
            'r_p2': self._data.r_p**2, 'r_n2': self._data.r_n**2, 
            'r_isoscalar2': self._data.r_isoscalar**2, 
            'r_isovector2': self._data.r_isovector**2,
            'r_charge2'   : self._data.r_charge**2,
        }
        if self._data._is_vap_calculation:
            text_results_1  = "%"*60 + '\n' +self.__header_proj+'\n'+"%"*60 +'\n\n'
            text_results_1 += _TMP_TAURUS_VAP_RESULT_OUTPUT.format(**self._data.__dict__,
                                                                   **other_kwargs) +'\n'
        text_results_2  = "%"*60 + '\n' +self.__header_qp+'\n'+"%"*60 +'\n\n'
        text_results_2 += f"Label of the state: {self._data.label_state}\n"
        text_results_2 += _TMP_TAURUS_VAP_RESULT_OUTPUT.format(**self._data.__dict__,
                                                               **other_kwargs)
        kwargs = {
            'z': self.z, 'n': self.n, 'a' : self.A, 'seed': self._input.seed, 
            'Jz': self._data.Jz,   'parity': self._data.parity, 
            'input_text': txt_inp, 'input_dd_term': txt_inp_dd,
            'time_start': self.time_start,
            'time_start_iter': self.time_start_iter,
            'time_end_iter':   self.time_end_iter,
            'txt_gradient' :   self._iter_block_lines,
            'msg_enditer_status': self._iter_msg, 'iter_final': self._iter_final,
            'data_proj_hfb':   text_results_1, 'data_qp_hfb': text_results_2 +'\n',
        }
        
        with open(self._GOOD_RESULT_TMP, 'r') as f:
            tmp = f.read()
            self._text = tmp.format(**kwargs)
            if self._case_broken:
                lines = self._text.split('\n')[:-250]
                # enough to cut into the gradient descent
                self._text  = '\n'.join(lines)
                self._text += "\n\n XXX \n Some fatality occurr!\n"
    
    def getOutputFile(self):
        if self._text == '':
            raise Exception("Define something in the setUpOutput method.!")
        return self._text

#===============================================================================
#   OTHER OUTPUT FILES
#===============================================================================

class DataAxial(DataTaurus):
    
    """
    overwrite methods from DataTaurus by the original class from _legacy 
    (cannot be used in legacy scripts)
    """
    
    class HeaderEnum(Enum):
        # Number_of_protons  = 'Number of protons '
        # Number_of_neutrons = 'Number of neutrons'
        N        = '      N    '
        Var_N    = ' <N^2>  '
        One_body = 'Kinetic'
        ph_part  = 'HF Ener'
        pp_part  = 'Pairing'
        Coul_Dir = 'Coul Dir'
        Rearrang = 'Rearrang'
        Full_H   = 'HFB Ener'
        Q_10     = 'Q10'
        Q_20     = 'Q20'
        Q_30     = 'Q30'
        Q_40     = 'Q40'
        # Beta_10  = 'Beta 2' # no hay
        Beta_20  = 'Beta 2'
        Beta_30  = 'Beta 3'
        Beta_40  = 'Beta 4'
        Rmed     = 'MS Rad'
        DJx2     = 'DJx**2'
        
    __message_startiter = ' *                                                       I T E R A T I O N                                                        *'
    __message_converged = 'P R O P E R L Y     F I N I S H E D'
    __message_not_conv  = 'M A X I M U M   O F  I T E R A T I O N S   E X C E E D E D'
    __message_enditer = '                                                  COLLECTIVE MASSES'
    __endIteration_message = 'PROTON                NEUTRON                TOTAL'
    __endSummary        = 'EROT'
    
    ## update of the class exporting variables
    DEFAULT_OUTPUT_FILENAME = 'aux_output'
    INPUT_FILENAME  = 'aux.INP'
    BU_folder       = 'BU_results_Ax'
    BU_fold_constr  = 'BU_results_constr_Ax'
    EXPORT_LIST_RESULTS = 'export_resultAxial.txt'
    PROGRAM         = 'HFBAxial'
          
    def __init__(self, z, n, filename, empty_data=False):
        
        DataTaurus.__init__(self, z, n, filename, empty_data=True)
        """
        Copy the attributes for completion and then remove or reset the values
        available in an axial calculation.
        """
        for attr_ in filter( lambda x: not x.startswith('_'), copy(self.__dict__)):
            if attr_.endswith('_pn') or '_isovector' in attr_:
                delattr(self, attr_)
            elif attr_.startswith('date_') or attr_.startswith('iter_'):
                delattr(self, attr_)
        
        # Axial Preserves TR, Jz=Jz**2=0 and not Jy.
        del self.Jx
        del self.Jy
        del self.Jx_2
        del self.Jy_2
        del self.Jy_var
        self.Jz = 0.0
        del self.Jz_2
        del self.Jz_var
        
        self.P_T10_J00  = 0.0
        self.P_T00_J10  = 0.0
        self.P_T00_J1m1 = 0.0
        self.P_T00_J1p1 = 0.0
        
        self.parity = 1.0 ## always even
        
        del self.time_per_iter
        self._filename = filename
        del self._evol_obj   
        self.iter_max  = None    
        if not empty_data:
            try:
                self.get_results()
            except Exception as e:
                printf(" (AC)>> EXCEPTION from AxialData Constructor >> self::")
                printf(self)
                printf(" (AC)>> exception:: ", e, "<<(AC)")
                printf(" (AC)<< EXCEPTION from AxialData Constructor <<<<<<<< ")
    
    # --------------------------------------------------------------------------
    ## Ban DataTaurus invalid methods for this class by overwriting 
    # --------------------------------------------------------------------------
    
    def _processing_data_and_evolution(self, data_inp, data_evol):
        raise DataObjectException("Invalid Method for DataAxial")
    def _getAngularMomentum(self, line):
        raise DataObjectException("Invalid Method for DataAxial")
    def _getNumberNucleons(self, line):
        raise DataObjectException("Invalid Method for DataAxial")
    def _getPairCoupling(self, line):
        raise DataObjectException("Invalid Method for DataAxial")
    
    # --------------------------------------------------------------------------    
    def _read_HO_prop(self, lines_header):
        """
        Get the properties of the HO basis, (not adapted for the axial osc. lengths)
        """
        b_values = None
        MZmax    = None
        for line in lines_header:
            if 'MZMAX' in line:
                line = line.strip().split()
                MZmax = int(line[1])
            elif 'OSC LENGTHS FROM DATA FILE' in line:
                line = line.replace('OSC LENGTHS FROM DATA FILE', '').strip().split()
                b_values = float(line[0]), float(line[1])
                if b_values[0] != b_values[1]:
                    printf(f"[WARNING] DataAxial register has b_perp != b_z: {b_values}")
        
        
        self.ho_b_length = b_values[0]
        self.ho_hbaromega = 41.4246052127 / (self.ho_b_length**2) ## _Suhonen
        self._getSingleSpaceDegenerations(MZmax, MZmin=0)
        self.MZmax       = MZmax
    
    def _getValues(self, line, head_rm = ''):
        line = line.replace(head_rm, '').split()
        if 'MeV' in line[-1] or 'fm' in line[-1]:
            line.pop()
        vals = [float(l) if not '*' in l else np.NaN for l in line]
        return vals
    
    def get_results(self):    
        with open(self._filename, 'r') as f:
            data = f.read()
            if self.__message_converged in data: 
                self.properly_finished = True
            elif not self.__endIteration_message in data:
                self.broken_execution = True
                return
            f.seek(0) # rewind the file reading
            
            self._read_HO_prop(data.split(self.__message_startiter)[0].split('\n'))
            data = f.readlines()
        
        _energies = (self.HeaderEnum.One_body, self.HeaderEnum.ph_part, 
                     self.HeaderEnum.pp_part,  self.HeaderEnum.Full_H,   
                     self.HeaderEnum.Coul_Dir)
        
        skip_evol = True
        endIter  = False
        for line in data:
            
            if skip_evol and (not self.__endIteration_message in line):
                
                if endIter:
                    # getting the last iteration for the argument
                    if ' ITER' in line:
                        line = line.strip().split()
                        self.iter_max = int(line[5])
                elif ((self.__message_converged in line) or 
                      (self.__message_not_conv in line)):
                    endIter = True
                continue
            else: 
                skip_evol = False
                if self.__endSummary in line:
                    break
            
            # printf(line)
            if   self.HeaderEnum.N in line:
                vals = self._getValues(line, self.HeaderEnum.N)
                self.proton_numb, self.neutron_numb = vals[0], vals[1]
            elif self.HeaderEnum.Var_N in line:
                vals = self._getValues(line, self.HeaderEnum.Var_N)
                self.var_p, self.var_n = vals[0], vals[1]
            elif self.HeaderEnum.Rmed in line:
                vals = self._getValues(line, self.HeaderEnum.Rmed)
                self.r_p, self.r_n = vals[0], vals[1]
                self.r_isoscalar, self.r_charge = vals[2], vals[0]
                self.r_isovector = (self.r_n - self.r_p) / 2
            if True in (p in line for p in _energies):
                self._getEnergies(line)
            elif True in (d in line for d in ('Beta', 
                                              self.HeaderEnum.Q_10, self.HeaderEnum.Q_20,
                                              self.HeaderEnum.Q_30, self.HeaderEnum.Q_40)):
                self._getBetaDeformations(line)
            elif self.HeaderEnum.DJx2 in line:
                self.Jx_var = self._getValues(line, self.HeaderEnum.DJx2)[2]
        
        #=======================================================================
        # Elements conversion for _Taurus output
        #=======================================================================
        # add the ph_part = Gamma
        self.V_2B_pp = self.hf_pp - self.kin_p 
        self.V_2B_nn = self.hf_nn - self.kin_n
        self.V_2B    = self.hf    - self.kin
        
        ## complete the beta 10 entrance since it is not in the output file
        c1_ = 4 * np.pi / (3* (self.z + self.n))
        # self.b10_n = (2*np.sqrt(3*np.pi)/(3*self.r_n)) * self.q10_n (old set up)
        self.b10_n = self.q10_n * c1_ / self.r_n
        self.b10_p = self.q10_p * c1_ / self.r_p
        self.b10_isoscalar = self.q10_isoscalar * c1_ / self.r_isoscalar
        
        self.P_T1m1_J00 = self.var_p
        self.P_T1p1_J00 = self.var_n
        
    
    def _getBetaDeformations(self, line):
        
        if self.HeaderEnum.Beta_20 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_20)
            self.b20_p, self.b20_n, self.b20_isoscalar = vals[0],vals[1],vals[2]
            
            self.beta_p, self.beta_n  = abs(vals[0]), abs(vals[1])
            self.gamma_p = 0.0 if (vals[0] == abs(vals[0])) else 60.0
            self.gamma_n = 0.0 if (vals[1] == abs(vals[1])) else 60.0
            self.gamma_isoscalar = 0.0 if (vals[2] == abs(vals[2])) else 60.0            
            self.b20_isovector = (self.b20_n - self.b20_p) / 2
        elif self.HeaderEnum.Beta_30 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_30)
            self.b30_p, self.b30_n, self.b30_isoscalar = vals[0],vals[1],vals[2]
            self.b30_isovector = (self.b30_n - self.b30_p) / 2
        elif self.HeaderEnum.Beta_40 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_40)
            self.b40_p, self.b40_n, self.b40_isoscalar = vals[0],vals[1],vals[2]
            self.b40_isovector = (self.b40_n - self.b40_p) / 2
        ## Quadrupole lines
        elif self.HeaderEnum.Q_10 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_10)
            self.q10_p, self.q10_n, self.q10_isoscalar = vals[0],vals[1],vals[2]
            self.q10_isovector = (self.q10_n - self.q10_p) / 2
            # MS Rad come after Q10, for b10 see in HeaderEnum.Rmed setting
        elif self.HeaderEnum.Q_20 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_20)
            self.q20_p, self.q20_n, self.q20_isoscalar = vals[0],vals[1],vals[2]
            self.q20_isovector = (self.q20_n - self.q20_p) / 2
        elif self.HeaderEnum.Q_30 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_30)
            self.q30_p, self.q30_n, self.q30_isoscalar = vals[0],vals[1],vals[2]
            self.q30_isovector = (self.q30_n - self.q30_p) / 2
        elif self.HeaderEnum.Q_40 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_40)
            self.q40_p, self.q40_n, self.q40_isoscalar = vals[0],vals[1],vals[2]
            self.q40_isovector = (self.q40_n - self.q40_p) / 2
    
    def _getEnergies(self, line):
        if self.HeaderEnum.One_body in line:
            vals = self._getValues(line, self.HeaderEnum.One_body)
            self.kin_p, self.kin_n, self.kin = vals[0], vals[1], vals[2]
        else:
            if   self.HeaderEnum.ph_part in line:
                vals = self._getValues(line, self.HeaderEnum.ph_part)
                self.hf_pp, self.hf_nn = vals[0], vals[1]
                self.hf    = vals[2]
            elif self.HeaderEnum.pp_part in line:
                vals = self._getValues(line, self.HeaderEnum.pp_part)
                self.pair_pp, self.pair_nn = vals[0], vals[1]
                self.pair    = vals[2]
            # elif self.HeaderEnum.Coul_Dir in line:
            #     vals = self._getValues(line, self.HeaderEnum.Coul_Dir)
            #     self.V_2B_pp, self.V_2B_nn = vals[0], vals[1]
            #     self.V_2B    = vals[2]
            elif self.HeaderEnum.Full_H in line:
                vals = self._getValues(line, self.HeaderEnum.Full_H)
                self.E_HFB_pp, self.E_HFB_nn = vals[0], vals[1]
                self.E_HFB    = vals[2]    
    
    @property
    def getAttributesDictLike(self):
        return ', '.join([k+' : '+str(v) for k,v in self.__dict__.items()])
    
    def isAxial(self, and_spherical=False):
        if and_spherical:
            _properties = [
                self.Jx_var, 
                self.q10_n, self.q10_isoscalar, self.q20_n, self.q20_isoscalar,
                self.q30_n, self.q30_isoscalar, self.q40_n, self.q40_isoscalar,
            ]
            _properties = [almostEqual(attr_, 1.e-5) for attr_ in _properties]
            return not False in _properties
        return True

class DataAttributeHandler(object):
    """
    This class is used to define an arbitrary function of one (or several) 
    attribute values of implemented _DataObject class instances.
    
    f(x, y, z, ...) on DTYPE object, being {x,y,z,} attributes
    
    NOTE: Remember the attributes to be in the order given for the lambda function
    
    Examples:
        We want the function(pair_pn, var_n) = 1.25* pair_pn / sqrt(var_n) - 1.2
    
    
    complex_attr = DataAttributeHandler(
                        lambda x, y: 1.25*x/(y**0.5) - 1.2,
                        pair_pn, 
                        var_n)
    
    res = DataTaurus(z, n, 'file2import') ## Defines all the attributes
    printf(complex_attr.getValue(res)) 
        >> 1.5321684 (i.e.)
    
    
    res = DataAxial(z, n, 'file2import') ## Note, DataAxial has no 'pair_pn'
    printf(complex_attr.getValue(res)) 
        >> AttributeError: 'DataAxial' object has no attribute 'pair_pn'
    """
    def __init__(self, function_, *attrs2use):
        
        assert isinstance(function_, type(lambda x:x)), "function must be a lambda functional instance"
        self._function = function_
        
        for attr in attrs2use:
            assert isinstance(attr, str), "Attributes given must be strings for the DataObject to use getattr"
        
        self._attr2use = attrs2use
        self._name = ', '.join(attrs2use)
        self._name = f"f({self._name})"
    
    def getValue(self, result: _DataObjectBase):
        """ 
        Return the value of the function on the result object. 
        Invalid access will result into an error.
        """
        vals = [getattr(result, attr_) for attr_ in self._attr2use]
        return self._function(*vals)
    
    def setName(self, name: str):
        self._name = name
    
    ## COMMON METHODS NORMALY USED FOR ATTRIBUTE IMPORTING:
    #-------------------------------------------------------------------------
    def __str__(self):
        return self._name
    def startswith(self, str_):
        return False
    def endsswith(self, str_):
        return False

#===============================================================================
# Data-Containers
#
#===============================================================================
class BaseResultsContainer1D(_DataObjectBase):
    
    """
    Data containers store different calculations results and auxiliary files
    in order to process or select the final result of a calculation. 
    
    The object let easy access on run to a set of results, i.e. to do some trial
    calculations and then choosing the best result.
    
    Life cycle:
    
    __init__: create the back up folder to store temporary the files
    
    append(_DataObject, id_='fort_1', binary='fort.11', datfiles=['fort.21', ]): 
        appends the result ()to data and copying into the Back Up
    get(id_):
        returns all the data asociated with a previous id_, otherwise appends
    
    set(*)  : the same as append, but changing the values 
    
    clear() : remove the back up folder with the contents
    dump()  : creates an exportable result.
    
    
    It has to be used after after calling the calculation, in case the final 
    """
    
    BU_folder  = 'TEMP_BU'
    EXPORT_LIST_RESULTS = 'export_resultTaurus.txt'
    
    def __init__(self, container_name : str= None):
        
        self._file_id   = []
        self._results   = []
        self._binaries  = []
        self._dat_files = {}
        
        self._container_name = container_name
        
        if container_name:
            container_name = container_name.replace(' ', '_')
            self.BU_folder = container_name.upper()
            self._container_name = container_name
        self.setUpFolderBackUp(container_name)
    
    def clear(self):
        """
        Reseting the Container Object and deleting the temporal folder.
        """
        printf(f" * Reseting _DataTaurusContainer1D, deleted [{len(self._results)}] elements")
        
        self._results   = []
        self._file_id   = []
        self._binaries  = []
        self._dat_files = {}
        
        if os.path.exists(self.BU_folder): 
            shutil.rmtree(self.BU_folder)
        if os.path.exists(self.EXPORT_LIST_RESULTS):
            os.remove(self.EXPORT_LIST_RESULTS)
        self.setUpFolderBackUp(self._container_name)
    
    def _dat_filenaming(self, file_, id_):
        """
        Auxiliary method, change the complementary file for the id_
            id_ = 3
            canonicalbasis.dat   -> canonicalbasis_3.dat
            canonical.basis.txt  -> canonical.basis_3.txt
            canonicalbasis       -> canonicalbasis_3
        """
        if file_.endswith('.dat'):
            file_2 = file_.replace('.dat', f"_{id_}.dat")
        elif '.' in file_:
            tail_  = file_.split('.')[-1]
            file_2 = file_.replace(tail_, f"_{id_}.{tail_}")
        else:
            file_2 = f"{file_}_{id_}"
        
        return file_2
    
    def append(self, result : _DataObjectBase, id_=None ,binary=None, datfiles=[]):
        """
        Save the files in the BU folder with an id and store the 
        """
        
        if id_ == None: id_ = len(self._results)
        self._file_id.append(id_)
        
        self._results.append(result)
        shutil.copy(result.DEFAULT_OUTPUT_FILENAME, f"{self.BU_folder}/{id_}.OUT")
        
        if binary:
            self._binaries.append(f'{id_}.bin')
            shutil.copy(binary, f"{self.BU_folder}/{id_}.bin")
        if datfiles:
            self._dat_files[id_] = []
            for file_ in datfiles:
                if not '.dat' in file_: file_ += '.dat'
                if not os.path.exists(file_):
                    printf(f"[WARNING DATACONTAINER] dat file {file_} not found, skip.")
                    continue
                file_2 = self._dat_filenaming(file_, id_)
                self._dat_files[id_].append(file_2)
                shutil.copy(file_, f"{self.BU_folder}/{file_2}")
    
    def get(self, id_, list_index_element=None):
        """ 
        Get the i-th element and complementary files.
        :list_index_element (optional) int for the index to select.
        
        return:
            res: DataType, bin (filename), [LIST: other datafiles], 
                (if list_index_element: id_ text for that index)
        """
        if not id_ in self._file_id and list_index_element == None:
            printf( " [Error] Invalid index for DataContainer1D.")
            return None, None, []
        
        if list_index_element == None:
            index_ = self._file_id.index(id_)
        else:
            index_ = list_index_element
            id_ = self._file_id[index_]
        
        args = [
            self._results[index_],
            self._binaries[index_],
            self._dat_files[id_] if id_ in self._dat_files else [],
        ]
        if list_index_element != None: args.append(self._file_id[index_])
        
        return args
    
    def getAllResults(self):
        """
        Get the results by id in order to filter.
        """
        return dict(zip(self._file_id, self._results))
    
    def set(self, id_, result: _DataObjectBase, binary=None, datfiles=[]):
        """
        In order to change the result already stored by its id.
        """
        if id_ in self._file_id:
            i = self._file_id.index(id_)
            self._results[i] = result
            shutil.copy(result.DEFAULT_OUTPUT_FILENAME, f"{self.BU_folder}/{id_}.OUT")
            
            if binary:
                self._binaries[i] = f'{id_}.bin'
                shutil.copy(binary, f"{self.BU_folder}/{id_}.bin")
            
            if datfiles:
                self._dat_files[id_] = []
                for file_ in datfiles:
                    if not os.path.exists(file_):
                        printf(f"[WARNING DATACONTAINER] dat file {file_} not found, skip.")
                        continue
                    file_2 = self._dat_filenaming(file_, id_)
                    self._dat_files[id_][i] = file_2
                    shutil.copy(file_, f"{self.BU_folder}/{file_2}")
        else:
            self.append(result, id_, binary, datfiles)
    
    def dump(self, output_file=None ):
        """
        export results in file
        """
        txt_ = '\n'.join([res.getAttributesDictLike for res in self._results]) 
        
        if output_file == None:
            output_file = self.EXPORT_LIST_RESULTS
        
        with open(output_file, 'w+') as f:
            f.write(txt_)



if __name__ == '__main__':
    pass
    # res = DataTaurus(12, 12, '../DATA_RESULTS/Beta20/Mg_GDD_test/24_VAP9/BU_folder_hamil_gdd_100_z12n12/res_z12n12_d1_0.OUT')
    # res = DataTaurus(10, 6, '../data_resources/testing_files/TEMP_res_z1n12_taurus_vap.txt') # TEMP_res_z2n1_0-dbase3odd
    
    # res = DataTaurus(10, 10, 'aux_output_Z10N6_broken')
    # res = DataTaurus(18, 18, '../res_z18n18_dbase.OUT')
    # with open(res.BU_fold_constr, 'w+') as f:
    #     f.write(res.getAttributesDictLike)
    
    # res = EvolTaurus('aux_output_Z10N10_00_00')
    # res = EvolTaurus('aux_output_Z10N6_23')
    # res = EvolTaurus('../res_z18n18_dbase.OUT')
    # res.printEvolutionFigure()
    # res = EvolTaurus('aux_output_Z10N6_broken')
    
    # res = EigenbasisData()
    # res.getResults()
    # res = DataAxial(10, 10, 'out_20Ne.OUT')
    # with open(res.EXPORT_LIST_RESULTS, 'w+') as f:
    #     f.write(res.getAttributesDictLike)
    
    # res = DataTaurusPAV(12, 12, '../OUT_1')
    # res = DataTaurusPAV(11, 20, '../test_obl01_odd_31Na/OUT_m1_m1.OUT')
    # res = DataTaurusPAV(8, 9, '../data_resources/testing_files/TEMP_res_PAV_z8n9_brokenoverlap.txt')
    # res = DataTaurusPAV(12, 19, '../data_resources/testing_files/TEMP_res_PAV_z2n1_odd_oldversion.txt')
    # res = DataTaurusPAV(12, 19, '../data_resources/testing_files/TEMP_res_PAV_z0n3_resultwithproblems.txt')
    # res = DataTaurusPAV(8, 9, '../data_resources/testing_files/TEMP_res_PAV_z8n9_1result.txt')
        
    _ = 0
    
    
