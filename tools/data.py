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
from datetime import datetime
import os
import shutil

import numpy as np
from tools.helpers import ValenceSpacesDict_l_ge10_byM, readAntoine,\
    getSingleSpaceDegenerations
from copy import copy
from tools.Enums import Enum

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
                print(" (TEvC)>> EXCEPTION from Taurus Constructor >> self::")
                print(self)
                print(" (TEvC)>> exception:: ", e, "<<(TC)")
                print(" (TEvC)<< EXCEPTION from Taurus Constructor <<<<<<<< ")
    
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
        
        print()
        print("ITER MAX: ", self.iter_max, self.z, self.n)
        print("LEN GRAD: ", self.grad.__len__())
        print("LEN EHFB: ", self.e_hfb.__len__())
        print("LEN ETA: ", self.eta.__len__())
        print("LEN MU: ", self.mu.__len__())
        print("LEN VAR Z:", self.var_Z.__len__())
        print("LEN VAR N:", self.var_N.__len__())
        if self.top_h2:
            print("LEN htop_h2: ", self.top_h2.__len__())
        if self.sp_dens:
            print("LEN sp_dens: ", self.sp_dens.__len__())
    
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
                    print("[WARNING] Trace Test went wrong (not 6.0), got", line[-1])
            
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
                    print("[TEvol. Parse WARNING]: unknown case (skip):", line)
        
    
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
            
        
        if hT2 in data_evol: ## get the line as the first 80 chars of the bock
            line = data_evol.split(hT2)[1][:80].split('\n')[0]
            times_execution.append(datetime.strptime(line,self.FMT_DT))
        if hT3 in data_evol:
            line = data_evol.split(hT3)[1][:80].split('\n')[0]
            line = line.split(' ITER_FINAL=')
            line, iter_max = line[0], line[1]
            self.iter_max = min(int(iter_max) - 1, 1) # to avoid 0/0
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
            print("[Warning DD evol] Cannot print revolution results for void calculation.")
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
        

class EigenbasisData(_DataObjectBase):
    
    FILENAME_DEFAULT = 'canonicalbasis.dat'
    
    class HeaderEnum(Enum):
        FermiProt = "Proton  fermi energy ="
        FermiNeut = "Neutron fermi energy ="
    
    def __init__(self, filename=None):
        
        self.fermi_energ_prot = None
        self.fermi_energ_neut = None 
        
        self.index = []
        self.avg_neut = []
        self.avg_prot = []
        self.avg_n    = []
        self.avg_l    = []
        self.avg_p    = []
        self.avg_j    = []
        self.avg_jz   = []
        self.v2   = []
        self.h    = []
        
        self.filename = filename if (filename!=None) else self.FILENAME_DEFAULT
        
    def getResults(self):
        
        with open(self.filename, 'r') as f:
            data = f.readlines()
        
        _in_values = False
        for line in data:
            line = line.strip()
            if len(line) == 0: 
                continue
            if not _in_values:
                if   line.startswith(self.HeaderEnum.FermiProt):
                    self.fermi_energ_prot = self._getValues(line, self.HeaderEnum.FermiProt)
                elif line.startswith(self.HeaderEnum.FermiNeut):
                    self.fermi_energ_neut = self._getValues(line, self.HeaderEnum.FermiNeut)
                # elif line.startswith('#      Z        N'): 
                #     pass
                elif line.startswith('---'):
                    _in_values = True
                continue
            else:
                
                vals = self._getValues(line)
                
                self.index.append(vals[0])
                self.avg_prot.append(vals[1])
                self.avg_neut.append(vals[2])
                self.avg_n.append( vals[3])
                self.avg_l.append( vals[4])
                self.avg_p.append( vals[5])
                self.avg_j.append( vals[6])
                self.avg_jz.append(vals[7])
                if len(vals) == 9: 
                    ## you are importing an eigenbasis file, v2 is not present 
                    self.v2.append(vals[8])
                self.h .append(vals[-1])
    
        _=0
        
#===============================================================================
#   RESULTS FROM TAURUS 
#===============================================================================
class DataTaurus(_DataObjectBase):
        
    class HeaderEnum(Enum): ## line header to read the output file
        Number_of_protons  = 'Number of protons '
        Number_of_neutrons = 'Number of neutrons'
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
        Q_10     = 'Q_10'
        Q_11     = 'Q_11'
        Q_20     = 'Q_20'
        Q_21     = 'Q_21'
        Q_22     = 'Q_22'
        Q_30     = 'Q_30'
        Q_32     = 'Q_32'
        Q_40     = 'Q_40'
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
    __message_enditer   = '               QUASIPARTICLE STATE PROPERTIES'
    __message_converged = 'Calculation converged'
    __message_not_conv  = 'Maximum number of iterations reached'
    __endIteration_message = 'TIME_END:' # 'Label of the state: '
    
    PROGRAM         = 'taurus_vap.exe'
    
    FMT_DT = '%Y/%m/%d %H_%M_%S.%f'
    
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
        
        self.proton_numb = None
        self.neutron_num = None
        self.var_p = None
        self.var_n = None
        
        self.kin   = None
        self.kin_p = None
        self.kin_n = None
        self.hf    = None
        self.hf_pp = None
        self.hf_nn = None
        self.hf_pn = None
        self.pair  = None
        self.pair_pp = None
        self.pair_nn = None
        self.pair_pn = None
        self.V_2B    = None
        self.V_2B_pp = None
        self.V_2B_nn = None
        self.V_2B_pn = None
        self.E_HFB    = None
        self.E_HFB_pp = None
        self.E_HFB_nn = None
        self.E_HFB_pn = None
        
        self.beta_p  = None
        self.beta_n  = None
        self.beta_isoscalar  = None
        self.beta_isovector  = None
        self.gamma_p = None
        self.gamma_n = None
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
        
        self._filename = filename
        self._evol_obj  = None
        self._input_obj = None
        if not empty_data:
            try:
                self.get_results()
            except Exception as e:
                print(" (TC)>> EXCEPTION from Taurus Constructor >> self::")
                print(self)
                print(" (TC)>> exception:: ", e, "<<(TC)")
                print(" (TC)<< EXCEPTION from Taurus Constructor <<<<<<<< ")
                
        
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
        self._evol_obj = EvolTaurus(None, empty_data=True)
        self._evol_obj._processing_data_and_evolution(data_inp, data_evol, True)
        
        _attr2copy = ['date_start', 'date_start_iter', 'date_end_iter',
                      'iter_max', 'iter_time_seconds', 'time_per_iter',
                      'sp_dim', 'MZmax', 'ho_hbaromega', 'ho_b_length', 'sh_dim']
        
        for attr_ in _attr2copy:
            setattr(self, attr_, getattr(self._evol_obj, attr_))
    
    
    def get_results(self):    
        with open(self._filename, 'r') as f:
            data = f.read()
            if self.__message_converged in data: 
                self.properly_finished = True
            elif not self.__message_enditer in data:
                self.broken_execution  = True
                return
            f.seek(0) # rewind the file reading
            
            data_inp, data_evol  = f.read().split(self.__message_startiter)
            data_evol, data = data_evol.split(self.__message_enditer)
            data      = data.split('\n')
                    
        _energies = (self.HeaderEnum.One_body, self.HeaderEnum.ph_part, 
                     self.HeaderEnum.pp_part,  self.HeaderEnum.Two_body, 
                     self.HeaderEnum.Full_H)
        
        self._read_inp   = True
        
        self._processing_data_and_evolution(data_inp, data_evol)
        
        for line in data:            
            # print(line)
            if 'Number of' in line:
                self._getNumberNucleons(line)
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
        #print("deform results :: [{}]".format(vals))
    
    def _roundGamma0(self, vals):
        if abs(vals[2] - 180) < 1.e-8 or abs(vals[2] - 360) < 1.e-8:
            vals[2] = 0.0
    
    def _getEnergies(self, line):
        if self.HeaderEnum.One_body in line:
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
            self.neutron_num, self.var_n = vals[0], vals[1]
    
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
        
        elements = dict([tuple(l.split(':')) for l in elements])
        for k, val in elements.items():
            
            
            k = k.strip()
            val = val.strip()
            if k in 'zn':
                setattr(self, k, int(val))
            elif k in ('properly_finished', 'broken_execution'):
                setattr(self, k, bool(val))
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
            self.date_start  = datetime.strftime(self.date_start, self.FMT_DT)
        except ValueError:
            self.FMT_DT = '%Y/%m/%d %H_%M_%S'
            self.date_start  = datetime.strftime(self.date_start, self.FMT_DT)
        self.date_start_iter = datetime.strftime(self.date_start_iter, self.FMT_DT)
        self.date_end_iter   = datetime.strftime(self.date_end_iter, self.FMT_DT)
        
        public_attr = filter(lambda a: not a[0].startswith('_'), self.__dict__.items())
        dict_ = ', '.join([k+' : '+str(v) for k,v in public_attr])
        self.date_start, self.date_start_iter, self.date_end_iter = d1, d2, d3
        
        return dict_
    


class _DataTaurusContainer1D:
    
    """
    This object store the results as stack, to be instanced on executor 
    classmethod to keep dataTaurus results in order
    """
    EXPORT_LIST_RESULTS = 'export_resultTaurus.txt'
    
    def __init__(self):
        self._results = []
    
    def reset(self):
        print(f" * Reseting _DataTaurusContainer1D, deleted [{len(self._results)}] elements")
        self._results = []
    
    def append(self, result : DataTaurus):
        
        assert isinstance(result, DataTaurus), f"invalid result type given: {result.__class__}"
        
        self._results.append(result)
    
    
    def get(self, index_):
        """ get the i-th element """
        if len(self._results) >= index_:
            return None
        return self._results[index_]
    
    def set(self, index_, result):
        """ get the i-th element """
        assert isinstance(result, DataTaurus), f"invalid result type given: {result.__class__}"
        
        if len(self._results) >= index_:
            print(f"[WARNING] index [{index_}] > dimension of list [{len(self._results)}], appending")
            self.append(result)
        else:
            self._results[index_] = result
            
    
    def dump(self, output_file=None ):
        """
        export results in file
        """
        txt_ = '\n'.join([res.getAttributesDictLike for res in self._results]) 
        
        if output_file == None:
            output_file = self.EXPORT_LIST_RESULTS
        
        with open(output_file, 'w+') as f:
            f.write(txt_)
    

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
        
        del self.time_per_iter
        self._filename = filename
        del self._evol_obj       
        if not empty_data:
            try:
                self.get_results()
            except Exception as e:
                print(" (AC)>> EXCEPTION from AxialData Constructor >> self::")
                print(self)
                print(" (AC)>> exception:: ", e, "<<(AC)")
                print(" (AC)<< EXCEPTION from AxialData Constructor <<<<<<<< ")
    
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
                    print(f"[WARNING] DataAxial register has b_perp != b_z: {b_values}")
        
        
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
        for line in data:
            
            if skip_evol and (not self.__endIteration_message in line):
                continue
            else: 
                skip_evol = False
                if self.__endSummary in line:
                    break
            
            # print(line)
            if   self.HeaderEnum.N in line:
                vals = self._getValues(line, self.HeaderEnum.N)
                self.proton_numb, self.neutron_num = vals[0], vals[1]
            elif self.HeaderEnum.Var_N in line:
                vals = self._getValues(line, self.HeaderEnum.Var_N)
                self.var_p, self.var_n = vals[0], vals[1]
            elif self.HeaderEnum.Rmed in line:
                vals = self._getValues(line, self.HeaderEnum.Rmed)
                self.r_p, self.r_n = vals[0], vals[1]
                self.r_isoscalar, self.r_charge = vals[2], vals[0]
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
        
        elif self.HeaderEnum.Beta_30 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_30)
            self.b30_p, self.b30_n, self.b30_isoscalar = vals[0],vals[1],vals[2]
        elif self.HeaderEnum.Beta_40 in line:
            vals = self._getValues(line, self.HeaderEnum.Beta_40)
            self.b40_p, self.b40_n, self.b40_isoscalar = vals[0],vals[1],vals[2]
        ## Quadrupole lines
        elif self.HeaderEnum.Q_10 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_10)
            self.q10_p, self.q10_n, self.q10_isoscalar = vals[0],vals[1],vals[2]
            # MS Rad come after Q10, for b10 see in HeaderEnum.Rmed setting
        elif self.HeaderEnum.Q_20 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_20)
            self.q20_p, self.q20_n, self.q20_isoscalar = vals[0],vals[1],vals[2]
        elif self.HeaderEnum.Q_30 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_30)
            self.q30_p, self.q30_n, self.q30_isoscalar = vals[0],vals[1],vals[2]
        elif self.HeaderEnum.Q_40 in line:
            vals = self._getValues(line, self.HeaderEnum.Q_40)
            self.q40_p, self.q40_n, self.q40_isoscalar = vals[0],vals[1],vals[2]
    
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
    


if __name__ == '__main__':
    pass
    # res = DataTaurus(10, 10, 'aux_output_Z10N10_00_00')
    # res = DataTaurus(10, 6, 'aux_output_Z10N6_23')
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
    