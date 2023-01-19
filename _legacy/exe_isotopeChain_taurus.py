"""
Created on Fri Mar  4 19:28:46 2022

@author: Miguel
"""
from collections import OrderedDict
from datetime import datetime
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np


class Template:
    com = 'com'
    z   = 'z'
    n   = 'n'
    seed= 'seed'
    b20 = 'b20'


template = """Interaction   
-----------
Master name hamil. files      D1S_t0_SPSD
Center-of-mass correction     {com}
Read reduced hamiltonian      0
No. of MPI proc per H team    0

Particle Number
---------------
Number of active protons      {z}.00
Number of active neutrons     {n}.00  
No. of gauge angles protons   1
No. of gauge angles neutrons  1

Wave Function   
-------------
Type of seed wave function    {seed} 
Number of QP to block         0
No symmetry simplifications   0
Seed random number generation 0
Read/write wf file as text    0
Cutoff occupied s.-p. states  0.00E-00
Include all empty sp states   0
Spatial one-body density      0
Discretization for x/r        0   0.00 
Discretization for y/theta    0   0.00 
Discretization for z/phi      0   0.00 

Iterative Procedure
-------------------
Maximum no. of iterations     1000 
Step intermediate wf writing  0
More intermediate printing    0
Type of gradient              1
Parameter eta for gradient    0.100E-00
Parameter mu  for gradient    0.300E-00
Tolerance for gradient        5.00E-04

Constraints             
-----------
Force constraint N/Z          1
Constraint beta_lm            2
Pair coupling scheme          1
Tolerance for constraints     1.000E-06
Constraint multipole Q10      1   0.000
Constraint multipole Q11      1   0.000
Constraint multipole Q20      {b20}
Constraint multipole Q21      1   0.000
Constraint multipole Q22      1   0.000
Constraint multipole Q30      0   0.000
Constraint multipole Q31      0   0.000
Constraint multipole Q32      0   0.000
Constraint multipole Q33      0   0.000
Constraint multipole Q40      0   0.000
Constraint multipole Q41      0   0.000
Constraint multipole Q42      0   0.000
Constraint multipole Q43      0   0.000
Constraint multipole Q44      0   0.000
Constraint radius sqrt(r^2)   0   0.000
Constraint ang. mom. Jx       0   0.000
Constraint ang. mom. Jy       0   0.000
Constraint ang. mom. Jz       0   0.000
Constraint pair P_T00_J10     0   0.000
Constraint pair P_T00_J1m1    0   0.000
Constraint pair P_T00_J1p1    0   0.000
Constraint pair P_T10_J00     0   0.000
Constraint pair P_T1m1_J00    0   0.000
Constraint pair P_T1p1_J00    0   0.000
Constraint field Delta        0   0.000
"""



nucleus = [
#    Z  N
    (2, 2), 
    (2, 4),
    #(4, 4), 
    (4, 6),
    (6, 6), 
    (6, 8),
    (8, 8),
    (10, 8),
    (10, 10),
    (12, 10),
    (14, 12),
    (14, 14)
]

nucleus = [(8, n) for n in range(6, 15, 2)]
## put here value in axial (divide by 3/5 to fix with taurus q20)
repeat = {
    # # (2, 2), 
    # (2, 4) : 0.1 / 0.6,
    # (4, 4) : 0.4 / 0.6, 
    # # (4, 6) : -0.4 / 0.6,
    # # (6, 6) : -0.4 / 0.6, 
    # # (6, 8) : -0.1 / 0.6,
    # # (8, 8),
    # (10, 8) : 0.0,
    # # (10, 10): +0.2 / 0.6,
    # (12, 10): +0.3 / 0.6,
    # (14, 12) : 0.23 / 0.6,

    # #(6, 6) : -0.4  / 0.6,
}

class DataTaurus:
    
    class Enum:
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
        Q_10     = 'Q_10'
        Q_11     = 'Q_11'
        Q_20     = 'Q_20'
        Q_21     = 'Q_21'
        Q_22     = 'Q_22'
        Q_30     = 'Q_30'
        Q_32     = 'Q_32'
        Beta     = 'Beta '
        Gamma    = 'Gamma'
        R2med    = '  r^2 '
        Rmed     = '  r '
        Jx       = '  X    '
        Jy       = '  Y    '
        Jz       = '  Z    '
        PairT0J1 = 'T = 0 ; J = 1'
        PairT1J0 = 'T = 1 ; J = 0'
        
        dd_evol = ' *Top H2'
    
    __message_converged = 'Calculation converged'
    __message_not_conv  = 'Maximum number of iterations reached'
    __endIteration_message = 'TIME_END:' # 'Label of the state: '
    
    output_filename_DEFAULT = 'aux_output'
    INPUT_FILENAME  = 'aux.INP'
    BU_folder       = 'BU_results'
    BU_fold_constr  = 'BU_results_constr'
    export_list_results = 'export_resultTaurus.txt'
    PROGRAM         = 'Taurus'
    
    FMT_DT = '%Y/%m/%d %H_%M_%S.%f'
    
    def __init__(self, z, n, filename, empty_data=False):
        
        self.z = z
        self.n = n
        self.properly_finished = False
        
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
        self.beta_isoscalar = None
        self.beta    = None
        self.beta_isovector  = None
        self.gamma_p = None
        self.gamma_n = None
        self.gamma   = None
        self.gamma_isovector = None
        
        self.b10_p = None
        self.b10_n = None
        self.b10_isoscalar = None
        self.b11_p = None
        self.b11_n = None
        self.b11_isoscalar = None      
        self.b20_p = None
        self.b20_n = None
        self.b20_isoscalar = None
        self.b22_p = None
        self.b22_n = None
        self.b22_isoscalar = None
        self.b30_p = None
        self.b30_n = None
        self.b30_isoscalar = None
        self.b32_p = None
        self.b32_n = None
        self.b32_isoscalar = None
        
        self.q10_p = None
        self.q10_n = None
        self.q10_isoscalar = None
        self.q11_p = None
        self.q11_n = None
        self.q11_isoscalar = None
        self.q20_p = None
        self.q20_n = None
        self.q20_isoscalar = None
        self.q22_p = None
        self.q22_n = None
        self.q22_isoscalar = None
        self.q30_p = None
        self.q30_n = None
        self.q30_isoscalar = None
        self.q32_p = None
        self.q32_n = None
        self.q32_isoscalar = None
        
        self.r_p  = None
        self.r_n  = None
        self.r_isoscalar = None
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
    
    def get_results(self):    
        with open(self._filename, 'r') as f:
            data = f.read()
            if self.__message_converged in data: 
                self.properly_finished = True
            f.seek(0) # rewind the file reading
            
            data = f.readlines()
        
        _energies = (self.Enum.One_body, self.Enum.ph_part, self.Enum.pp_part,
                     self.Enum.Two_body, self.Enum.Full_H)
        
        times_execution = []
        hT1, hT2, hT3 = 'TIME_START: ', 'TIME_START_ITER: ', 'TIME_END: '
        skip_evol = True
        for _, line in enumerate(data):
            
            if skip_evol:
                if (not self.__endIteration_message in line):
                    if len(times_execution) < 2:
                        if line.startswith(hT1):
                            line = line.replace(hT1, '').replace('\n', '')
                            try:
                                times_execution.append(datetime.strptime(line,self.FMT_DT))
                            except ValueError:
                                self.FMT_DT = '%Y/%m/%d %H_%M_%S'
                                times_execution.append(datetime.strptime(line,self.FMT_DT))
                        elif line.startswith(hT2):
                            line = line.replace(hT2, '').replace('\n', '')
                            times_execution.append(datetime.strptime(line,self.FMT_DT))
                else: 
                    line = line.replace(hT3, '').replace('\n', '')
                    line = line.split(' ITER_FINAL=')
                    line, iter_max = line[0], line[1]
                    self.iter_max = int(iter_max)
                    times_execution.append(datetime.strptime(line, self.FMT_DT))
                    skip_evol = False
                continue
            
            # print(line)
            if 'Number of' in line:
                self._getNumberNucleons(line)
            elif self.Enum.Rmed in line:
                vals = self._getValues(line, self.Enum.Rmed)
                self.r_p, self.r_n = vals[0], vals[1]
                self.r_isoscalar, self.r_charge = vals[2], vals[4]
            elif self.Enum.Gamma in line:
                vals = self._getValues(line, self.Enum.Gamma)
                self._roundGamma0(vals)
                self.gamma_p, self.gamma_n       = vals[0], vals[1]
                self.gamma, self.gamma_isovector = vals[2], vals[3]
            if True in (p in line for p in _energies):
                self._getEnergies(line)
            if True in (d in line for d in ('Beta', 'Q_3', 'Q_2', 'Q_1')):
                self._getBetaDeformations(line)
            if True in (d in line for d in 
                        (self.Enum.Jx, self.Enum.Jy, self.Enum.Jz)):
                self._getAngularMomentum(line)
            if self.Enum.PairT0J1 in line or self.Enum.PairT1J0 in line:
                self._getPairCoupling(line)
                
        # save time related calculations
        if len(times_execution) == 3:
            self.iter_time_seconds = times_execution[2] - times_execution[1]
            self.iter_time_seconds = self.iter_time_seconds.seconds + \
                                     (1.0e-6 * self.iter_time_seconds.microseconds)
            self.time_per_iter = float(self.iter_time_seconds) / self.iter_max
        
            self.date_start      = times_execution[0] 
            self.date_start_iter = times_execution[1]
            self.date_end_iter   = times_execution[2]
        
        # return dict([(e, float(val)) for e, val in energies.items()]), prop_fin
    
    def _getBetaDeformations(self, line):
        #print("deform line    :: [{}]".format(line))
        if   self.Enum.Beta in line:
            vals = self._getValues(line, self.Enum.Beta)
            self.beta_p = vals[0] 
            self.beta_n = vals[1]
            self.beta, self.beta_isovector = vals[2], vals[3]
        elif self.Enum.Beta_10 in line:
            vals = self._getValues(line, self.Enum.Beta_10)
            self.b10_p, self.b10_n  = vals[0], vals[1]
            self.b10_isoscalar      = vals[2]
        elif self.Enum.Beta_11 in line:
            vals = self._getValues(line, self.Enum.Beta_11)
            self.b11_p, self.b11_n  = vals[0], vals[1]
            self.b11_isoscalar      = vals[2]
        elif self.Enum.Beta_20 in line:
            vals = self._getValues(line, self.Enum.Beta_20)
            self.b20_p, self.b20_n  = vals[0], vals[1]
            self.b20_isoscalar      = vals[2]
        elif self.Enum.Beta_22 in line:
            vals = self._getValues(line, self.Enum.Beta_22)
            self.b22_p, self.b22_n  = vals[0], vals[1]
            self.b22_isoscalar      = vals[2]
        elif self.Enum.Beta_30 in line:
            vals = self._getValues(line, self.Enum.Beta_30)
            self.b30_p, self.b30_n  = vals[0], vals[1]
            self.b30_isoscalar      = vals[2]
        elif self.Enum.Beta_32 in line:
            vals = self._getValues(line, self.Enum.Beta_32)
            self.b32_p, self.b32_n  = vals[0], vals[1]
            self.b32_isoscalar      = vals[2]
        
        elif self.Enum.Q_10 in line:
            vals = self._getValues(line, self.Enum.Q_10)
            self.q10_p, self.q10_n  = vals[0], vals[1]
            self.q10_isoscalar      = vals[2]
        elif self.Enum.Q_11 in line:
            vals = self._getValues(line, self.Enum.Q_11)
            self.q11_p, self.q11_n  = vals[0], vals[1]
            self.q11_isoscalar      = vals[2]
        elif self.Enum.Q_20 in line:
            vals = self._getValues(line, self.Enum.Q_20)
            self.q20_p, self.q20_n  = vals[0], vals[1]
            self.q20_isoscalar      = vals[2]
        elif self.Enum.Q_22 in line:
            vals = self._getValues(line, self.Enum.Q_22)
            self.q22_p, self.q22_n  = vals[0], vals[1]
            self.q22_isoscalar      = vals[2]
        elif self.Enum.Q_30 in line:
            vals = self._getValues(line, self.Enum.Q_30)
            self.q30_p, self.q30_n  = vals[0], vals[1]
            self.q30_isoscalar      = vals[2]
        elif self.Enum.Q_32 in line:
            vals = self._getValues(line, self.Enum.Q_32)
            self.q32_p, self.q32_n  = vals[0], vals[1]
            self.q32_isoscalar      = vals[2]
        #print("deform results :: [{}]".format(vals))
    
    def _roundGamma0(self, vals):
        if abs(vals[2] - 180) < 1.e-8 or abs(vals[2] - 360) < 1.e-8:
            vals[2] = 0.0
        
        # for i in range(4):
        #     if abs(vals[i] - 180) < 1.e-8 or abs(vals[i] - 360) < 1.e-8:
        #         vals[i] = 0.0
    
    def _getEnergies(self, line):
        if self.Enum.One_body in line:
            vals = self._getValues(line, self.Enum.One_body)
            self.kin_p, self.kin_n, self.kin = vals[0], vals[1], vals[2]
        else:
            if   self.Enum.ph_part in line:
                vals = self._getValues(line, self.Enum.ph_part)
                self.hf_pp, self.hf_nn = vals[0], vals[1]
                self.hf_pn, self.hf    = vals[2], vals[3]
            elif self.Enum.pp_part in line:
                vals = self._getValues(line, self.Enum.pp_part)
                self.pair_pp, self.pair_nn = vals[0], vals[1]
                self.pair_pn, self.pair    = vals[2], vals[3]
            elif self.Enum.Two_body in line:
                vals = self._getValues(line, self.Enum.Two_body)
                self.V_2B_pp, self.V_2B_nn = vals[0], vals[1]
                self.V_2B_pn, self.V_2B    = vals[2], vals[3]
            elif self.Enum.Full_H in line:
                vals = self._getValues(line, self.Enum.Full_H)
                self.E_HFB_pp, self.E_HFB_nn = vals[0], vals[1]
                self.E_HFB_pn, self.E_HFB    = vals[2], vals[3]
                
    def _getNumberNucleons(self, line):
        if self.Enum.Number_of_protons in line:
            vals = self._getValues(line, self.Enum.Number_of_protons)
            self.proton_numb, self.var_p = vals[0], vals[1]
        elif self.Enum.Number_of_neutrons in line:
            vals = self._getValues(line, self.Enum.Number_of_neutrons)
            self.neutron_num, self.var_n = vals[0], vals[1]
    
    def _getAngularMomentum(self, line):
        if line == "Part \ No.     Z          N          A":
            return
        J_id = line[:7]
        line = line[6:]
        vals = self._getValues(line) 
        if self.Enum.Jx in J_id:
            self.Jx, self.Jx_2, self.Jx_var = vals[0], vals[1], vals[2]
        elif self.Enum.Jy in J_id:
            self.Jy, self.Jy_2, self.Jy_var = vals[0], vals[1], vals[2]
        elif self.Enum.Jz in J_id:
            self.Jz, self.Jz_2, self.Jz_var = vals[0], vals[1], vals[2]
        
    def _getPairCoupling(self, line):
        
        if self.Enum.PairT0J1 in line:
            vals = self._getValues(line, self.Enum.PairT0J1)
            self.P_T00_J10 = vals[1]
            self.P_T00_J1m1, self.P_T00_J1p1 = vals[0], vals[2]
        elif self.Enum.PairT1J0 in line:
            vals = self._getValues(line, self.Enum.PairT1J0)
            self.P_T10_J00 = vals[1]
            self.P_T1m1_J00, self.P_T1p1_J00 = vals[0], vals[2]            
        
    def getDDEnergyEvolution(self): 
        
        v_dd_max = []
        v_dd_min = []
        with open(self._filename, 'r') as f:
            data = f.readlines()
            #print(data)
            for line in data:
                if self.__endIteration_message in line:
                    break
                
                if self.Enum.dd_evol in line:
                    vals = self._getValues(line, self.Enum.dd_evol)
                    print(line, ">>", vals)
                    v_dd_min.append(vals[0])
                    v_dd_max.append(vals[1])
        
        # todo, get frm " *Top H2" in line
        plt.figure()
        plt.plot(v_dd_min, 'b--')
        plt.plot(v_dd_max, 'r--')
        plt.show()
    
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
            elif k == 'properly_finished':
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
        
        dict_ = ', '.join([k+' : '+str(v) for k,v in self.__dict__.items()])
        self.date_start, self.date_start_iter, self.date_end_iter = d1, d2, d3
        
        return dict_
    
    @classmethod
    def setUpFolderBackUp(cls):
        # Create new BU folder
        
        if not os.path.exists(cls.BU_folder):
            os.mkdir(cls.BU_folder)
        else:
            shutil.rmtree(cls.BU_folder)
            os.mkdir(cls.BU_folder)
        if os.path.exists(cls.export_list_results):
            os.remove(cls.export_list_results)

def convergeDeformation(z, n, bmin_aprox):

    ## preconverge to the minimum
    q20_const = "1 {:+5.3f}".format(bmin_aprox)
    kwargs = {
        Template.com : 1,
        Template.z   : z,
        Template.n   : n,
        Template.seed : 3,
        Template.b20  : q20_const
    }
    
    text = template.format(**kwargs)
    with open(DataTaurus.INPUT_FILENAME, 'w+') as f:
        f.write(text)
    
    _e = subprocess.call('./taurus_vap.exe < {} > {}'
                             .format(DataTaurus.INPUT_FILENAME, 
                                     DataTaurus.output_filename), 
                         shell=True)
    ## check if problem or wrong minima (reduce deformation)
    
    ## import from previous function and free converge
    _e = subprocess.call('cp final_wf.bin initial_wf.bin', shell=True)
    
    kwargs = {
        Template.com : 1,
        Template.z   : z,
        Template.n   : n,
        Template.seed : 3,
        Template.b20  : '0 0.000'
    }
    
    text = template.format(**kwargs)
    
    with open(DataTaurus.INPUT_FILENAME, 'w+') as f:
        f.write(text)
    _e = subprocess.call('./taurus_vap.exe < {} > {}'
                             .format(DataTaurus.INPUT_FILENAME, 
                                     DataTaurus.output_filename), 
                         shell=True)
    
    # print("New q20=", get_EHFB()[0][Ener().b2])
    

#%% main

def mainLinux():
    #
    #%% Executing the process, run the list of isotopes
    #
    
    HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
    # Overwrite/create the buck up folder
    DataTaurus.setUpFolderBackUp()
    if os.path.exists(DataTaurus.export_list_results):
        os.remove(DataTaurus.export_list_results)
    
    results = []
    for q20_const in ('1 0.000', ):#, '1 0.000'):
        if q20_const == '0 0.000':
            print("Calculating interaction No constr")
        else:
            print("Calculating interaction fixing q20 =", q20_const)
        
        print(HEAD)
        for z, n in nucleus:
            try:
                A = z + n
                status_fin = ''
                
                ## ----- execution ----
                
                kwargs = {
                    Template.com  : 1,
                    Template.z    : z,
                    Template.n    : n,
                    Template.seed : 3,
                    Template.b20  : q20_const
                }
                
                # TODO: Uncomment
                text = template.format(**kwargs)
                with open(DataTaurus.INPUT_FILENAME, 'w+') as f:
                    f.write(text)
                
                _e = subprocess.call('./taurus_vap.exe < {} > {}'
                                          .format(DataTaurus.INPUT_FILENAME, 
                                                  output_filename), 
                                      shell=True,
                                      timeout=8640) # 1 day timeout)
                res = DataTaurus(z, n, output_filename)
                results.append(res)
                #res.getDDEnergyEvolution()
                
                # TODO: uncomment for repetition of the execution
                # if q20_const == '0 0.000':
                #     if (z, n) in repeat:
                #         print("Last q20:", q20, '   wanted:', repeat[(z, n)])
                #     if np.sign(q20) != np.sign(repeat[(z, n)]):
                #         print("Starting convergence to the minimum")
                #         convergeDeformation(z, n, repeat[(z, n)])
                #
                #         e_hfb_f, prop_fin = get_EHFB()
                
                
                ## TODO UNCOMMENT
                _e = subprocess.call('mv {} {}'.format(output_filename, 
                                          os.getcwd()+'/'+DataTaurus.BU_folder+'/'
                                          +output_filename
                                          +'_Z{}N{}'.format(z,n)),
                                     shell=True)
                _e = subprocess.call('rm *.dat *.red', shell=True)
                        
                if not res.properly_finished:
                    status_fin += 'X'
                else:
                    status_fin += '.'
                      
                print(" {:2} {:2}  ( {})    {:9.4f}  {:9.4f}  {:7.4f}  {:5.4f}"
                      .format(z, n, status_fin, res.E_HFB, res.kin, res.pair, res.beta))
            except Exception as e:
                print(e)
                print(res)
                
                
        
        # ## ------ end exec.  -----
        data = []
        for res in results:
            # line = ', '.join([k+' : '+str(v) for k,v in res.__dict__.items()])
            line = res.getAttributesDictLike
            data.append(line+'\n')
            
        with open(DataTaurus.export_list_results, 'a+') as f:
            f.writelines(data)
    

DataTaurus.export_list_results = "export_IsotopesTaurus.txt"

if __name__ == '__main__':
    output_filename = DataTaurus.output_filename_DEFAULT
    if not os.getcwd().startswith('C:'):
        mainLinux()    
    else:
        #%% process in windows
        results_taur = []
        results_axial = []
        from _legacy.exe_isotopeChain_axial import DataAxial
        export_list_results = DataAxial.export_list_results
        
        ## get results from export "csv"
        with open(DataTaurus.export_list_results, 'r') as f:
            data = f.readlines()
            for line in data:
                res = DataTaurus(None, None, None, True)
                res.setDataFromCSVLine(line)
                results_taur.append(res)
        with open(export_list_results, 'r') as f:
            data = f.readlines()
            for line in data:
                res = DataAxial(None, None, None, True)
                res.setDataFromCSVLine(line)
                results_axial.append(res)
        
        for attr_ in ('E_HFB', 'E_HFB_pp', 'E_HFB_nn'):
            ## plot energies
            x_tau, y_tau = [], []
            x_ax,  y_ax  = [], []
            
            for r in results_taur:
                x_tau.append(r.n)
                y_tau.append(getattr(r, attr_)/(r.n + r.z))
            for r in results_axial:
                x_ax.append(r.n)
                y_ax.append(getattr(r, attr_)/(r.n + r.z))
            
            plt.figure()
            plt.plot(x_tau, y_tau, '.-r',   label="taurus")
            plt.plot(x_ax, y_ax, '.-b',     label="axial")
            plt.title(attr_)
            plt.legend()
            plt.show()
    
        