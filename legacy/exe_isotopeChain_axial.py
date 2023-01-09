"""
Created on Fri Mar  4 19:28:46 2022

@author: Miguel
"""

class Template:
    com = 'com'
    z   = 'z'
    a   = 'a'
    force = 'force'
    seed  = 'seed'
    b_len = 'b_len'
    b20   = 'b20'

template = """NUCLEUS    {a:03} He   Z= {z:03}     >>> HFB OPT <<< {com} COULECH 2
EPSG 0.000001 MAXITER 00500    >>> OUTPUT  <<< 0 **** TSTG 0
ETMAX 0.7501 ETMIN 0.0351 DMAX 0.90 DMIN 0.70 TSHH 399.0
GOGNY FORCE        {force}    *** 0=D1S 1=D1 2=D1' 3(t3=0)
INPUT W.F.         {seed}    *** 0,1=WF FROM UNIT 10 (1 kicks), 2=NEW Function
OSCILLATOR LENGHT  0    *** 0               BP {b_len:9.7f} BZ {b_len:9.7f}
          >>>>>>>>>> C O N S T R A I N T S <<<<<<<<<<<
C.O.M.     1 1   0.00000000D+00
{b20}          >>>>>>>>>> E N D <<<<<<<<<<<<<<<<<<<<<<<<<<<  """
          
q20_constr_template = "QL    2    {:1} {:1}   {:10.8f}D+00"
b20_constr_template = "BL    2    {:1} {:1}   {:10.8f}D+00"
com_template   = "CM1 {} CM2 {}"


from collections import OrderedDict
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np



nucleus = [
#    Z  N
    # (2, 2), 
    # (2, 4),
    #(4, 4), 
    # (4, 6),
    # (6, 6), 
    # (6, 8),
    (10, 16),
    (8, 8),
    (8, 10),
    (8, 12),
    (10, 10),
    (10, 12),
    (10, 14),
    (14, 12),
    (14, 14),
]
## put here value in axial (divide by 3/5 to fix with taurus q20)

AXIAL_PROGRAM = 'HFBaxialMZ4'

class DataAxial:
    
    class Enum:
        # Number_of_protons  = 'Number of protons '
        # Number_of_neutrons = 'Number of neutrons'
        N        = '      N    '
        Var_N    = ' <N^2>  '
        One_body = 'Kinetic'
        ph_part  = 'HF Ener'
        pp_part  = 'Pairing'
        Coul_Dir = 'Coul Dir'
        Rearrang = 'Rearrang'
        # Two_body = 'Two-body'
        Full_H   = 'HFB Ener'
        # Beta_11  = 'Beta_11'
        # Beta_20  = 'Beta 2'
        # Beta_22  = 'Beta_22'
        Q10      = 'Q10'
        Q20      = 'Q20'
        Q30      = 'Q30'
        Q40      = 'Q40'
        Beta     = 'Beta 2'
        Beta3    = 'Beta 3'
        Beta4    = 'Beta 4'
        # Gamma    = 'Gamma'
        #R2med    = '  r^2 '
        Rmed     = 'MS Rad'
        DJx2     = 'DJx**2'
        
        dd_evol = ' *Top H2'
    
    __message_converged = 'P R O P E R L Y     F I N I S H E D'
    __message_not_conv  = 'M A X I M U M   O F  I T E R A T I O N S   E X C E E D E D'
    __endIteration_message = 'PROTON                NEUTRON                TOTAL'
    __endSummary        = 'EROT'
    
    output_filename_DEFAULT = 'aux_output'
    INPUT_FILENAME  = 'aux.INP'
    BU_folder       = 'BU_results_Ax'
    BU_fold_constr  = 'BU_results_constr_Ax'
    export_list_results = 'export_resultAxial.txt'
    PROGRAM         = 'HFBAxial'
    
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
        self.hf    = None  # t + Gamma
        self.hf_pp = None
        self.hf_nn = None
        # self.hf_pn = None
        self.pair  = None
        self.pair_pp = None
        self.pair_nn = None
        # self.pair_pn = None
        self.V_2B    = None  # Gamma
        self.V_2B_pp = None
        self.V_2B_nn = None
        # self.V_2B_pn = None
        self.E_HFB    = None
        self.E_HFB_pp = None
        self.E_HFB_nn = None
        # self.E_HFB_pn = None
        
        self.beta_p  = None
        self.beta_n  = None
        self.beta    = None         # abs value to give with beta
        self.beta_isoscalar = None
        
        self.b30_p  = None
        self.b30_n  = None
        self.b30_isoscalar = None
        self.b40_p  = None
        self.b40_n  = None
        self.b40_isoscalar = None
        
        # self.beta_isovector  = None
        # self.gamma_p = None
        # self.gamma_n = None
        self.gamma   = None
        # self.gamma_isovector = None
        
        self.b10_p = None
        self.b10_n = None
        self.b10_isoscalar = None
        self.q10_p = None
        self.q10_n = None
        self.q10_isoscalar = None
        self.q20_p  = None
        self.q20_n  = None
        self.q20_isoscalar = None
        self.q30_p  = None
        self.q30_n  = None
        self.q30_isoscalar = None
        self.q40_p  = None
        self.q40_n  = None
        self.q40_isoscalar = None
        
        self.r_p  = None
        self.r_n  = None
        self.r_isoscalar = None
        self.r_charge = None
        
        # self.Jx     = None
        # self.Jx_2   = None
        self.Jx_var = None
        # Axial Preserves TR, Jz=Jz**2=0 and not Jy.
        
        self._filename = filename
        if not empty_data:
            try:
                self.get_results()
            except Exception as e:
                print("[ERROR] :: in DataAxial.get_results():")
                print(self)
        
    def __str__(self):
        aux = OrderedDict(sorted(self.__dict__.items(), key=lambda t: t[0]))
        return "\n".join(k+' :\t'+str(v) for k,v in aux.items())
    
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
            f.seek(0) # rewind the file reading
            
            data = f.readlines()
        
        _energies = (self.Enum.One_body, self.Enum.ph_part, self.Enum.pp_part,
                     self.Enum.Full_H,   self.Enum.Coul_Dir)
        
        skip_evol = True
        for line in data:
            
            if skip_evol and (not self.__endIteration_message in line):
                continue
            else: 
                skip_evol = False
                if self.__endSummary in line:
                    break

            
            # print(line)
            if   self.Enum.N in line:
                vals = self._getValues(line, self.Enum.N)
                self.proton_numb, self.neutron_num = vals[0], vals[1]
            elif self.Enum.Var_N in line:
                vals = self._getValues(line, self.Enum.Var_N)
                self.var_p, self.var_n = vals[0], vals[1]
            elif self.Enum.Rmed in line:
                vals = self._getValues(line, self.Enum.Rmed)
                self.r_p, self.r_n = vals[0], vals[1]
                self.r_isoscalar, self.r_charge = vals[2], vals[0]
                
                self.b10_n = (2*np.sqrt(3*np.pi)/(3*self.r_n)) * self.q10_n
                self.b10_p = (2*np.sqrt(3*np.pi)/(3*self.r_p)) * self.q10_p
                self.b10_isoscalar = (2*np.sqrt(3*np.pi)* self.q10_isoscalar /
                                            (3*self.r_isoscalar)) 
            if True in (p in line for p in _energies):
                self._getEnergies(line)
            elif True in (d in line for d in ('Beta', self.Enum.Q10, self.Enum.Q20,
                                              self.Enum.Q30, self.Enum.Q40)):
                self._getBetaDeformations(line)
            elif self.Enum.DJx2 in line:
                self.Jx_var = self._getValues(line, self.Enum.DJx2)[2]
        
        # add the ph_part = Gamma
        self.V_2B_pp = self.hf_pp - self.kin_p 
        self.V_2B_nn = self.hf_nn - self.kin_n
        self.V_2B    = self.hf    - self.kin
        # return dict([(e, float(val)) for e, val in energies.items()]), prop_fin
    
    def _getBetaDeformations(self, line):
        
        if self.Enum.Beta in line:
            vals = self._getValues(line, self.Enum.Beta)
            self.beta_p, self.beta_n ,self.beta_isoscalar = vals[0], vals[1], vals[2]
            self.beta  = abs(vals[2])
            self.gamma = 0.0 if (self.beta == self.beta_isoscalar) else 60.0
            
        elif self.Enum.Beta3 in line:
            vals = self._getValues(line, self.Enum.Beta3)
            self.b30_p, self.b30_n ,self.b30_isoscalar = vals[0], vals[1], vals[2]
        elif self.Enum.Beta4 in line:
            vals = self._getValues(line, self.Enum.Beta4)
            self.b40_p, self.b40_n ,self.b40_isoscalar = vals[0], vals[1], vals[2]
        ## Quadrupole lines
        elif self.Enum.Q10 in line:
            vals = self._getValues(line, self.Enum.Q10)
            self.q10_p, self.q10_n ,self.q10_isoscalar = vals[0], vals[1], vals[2]
            # MS Rad come after Q10, for b10 see in Enum.Rmed setting
            
        elif self.Enum.Q20 in line:
            vals = self._getValues(line, self.Enum.Q20)
            self.q20_p, self.q20_n ,self.q20_isoscalar = vals[0], vals[1], vals[2]
        elif self.Enum.Q30 in line:
            vals = self._getValues(line, self.Enum.Q30)
            self.q30_p, self.q30_n ,self.q30_isoscalar = vals[0], vals[1], vals[2]
        elif self.Enum.Q40 in line:
            vals = self._getValues(line, self.Enum.Q40)
            self.q40_p, self.q40_n ,self.q40_isoscalar = vals[0], vals[1], vals[2]
    
    def _getEnergies(self, line):
        if self.Enum.One_body in line:
            vals = self._getValues(line, self.Enum.One_body)
            self.kin_p, self.kin_n, self.kin = vals[0], vals[1], vals[2]
        else:
            if   self.Enum.ph_part in line:
                vals = self._getValues(line, self.Enum.ph_part)
                self.hf_pp, self.hf_nn = vals[0], vals[1]
                self.hf    = vals[2]
            elif self.Enum.pp_part in line:
                vals = self._getValues(line, self.Enum.pp_part)
                self.pair_pp, self.pair_nn = vals[0], vals[1]
                self.pair    = vals[2]
            # elif self.Enum.Coul_Dir in line:
            #     vals = self._getValues(line, self.Enum.Coul_Dir)
            #     self.V_2B_pp, self.V_2B_nn = vals[0], vals[1]
            #     self.V_2B    = vals[2]
            elif self.Enum.Full_H in line:
                vals = self._getValues(line, self.Enum.Full_H)
                self.E_HFB_pp, self.E_HFB_nn = vals[0], vals[1]
                self.E_HFB    = vals[2]
        
    
    def getDDEnergyEvolution(self): 
        
        v_dd_max = []
        v_dd_min = []
        with open(self._filename, 'r') as f:
            data = f.readlines()
            print(data)
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
        try:
            for k, val in elements.items():
                k = k.strip()
                if k in 'zn':
                    setattr(self, k, int(val))
                elif k == 'properly_finished':
                    setattr(self, k, bool(val))
                elif k.startswith('_'):
                    continue
                else:
                    setattr(self, k, float(val))
        except ValueError as v:
            print("! missing attribute", k, val, )
            #raise v
        #print(self)
    
    @property
    def getAttributesDictLike(self):
        return ', '.join([k+' : '+str(v) for k,v in self.__dict__.items()])
    
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
    q20_const = "BL    2    1 1   {:+10.8f}D+00".format(bmin_aprox)
    q20_const = ""
    kwargs = {
        Template.com : 1,
        Template.z   : z,
        Template.n   : n,
        Template.force : 0,
        Template.seed : 3,
        Template.b_len: 1.5,
        Template.b20  : q20_const
    }
    
    text = template.format(**kwargs)
    with open(DataAxial.INPUT_FILENAME, 'w+') as f:
        f.write(text)
    
    _e = subprocess.call('./HFBaxial < {} > {}'
                             .format(DataAxial.INPUT_FILENAME, 
                                     DataAxial.output_filename), 
                         shell=True)
    ## check if problem or wrong minima (reduce deformation)
    
    ## import from previous function and free converge
    _e = subprocess.call('cp final_wf.bin initial_wf.bin', shell=True)
    
    kwargs = {
        Template.com : 1,
        Template.z   : z,
        Template.n   : n,
        Template.force: 0,
        Template.seed : 3,
        Template.b_len: 1.5,
        Template.b20  : ''
    }
    
    text = template.format(**kwargs)
    
    with open(DataAxial.INPUT_FILENAME, 'w+') as f:
        f.write(text)
    _e = subprocess.call('./HFBaxial < {} > {}'
                             .format(DataAxial.INPUT_FILENAME, 
                                     DataAxial.output_filename), 
                         shell=True)
    
    # print("New q20=", get_EHFB()[0][Ener().b2])
    
def computeHBarOmegaShellDependence(z, n, force=0, b_list=[1.5,]):
    
    output_filename ='outputAx.OUT'
    
    A = z + n
    q20_const = ""
    kwargs = {
        Template.com : com_template.format(1,1),
        Template.z   : z,
        Template.a   : A,
        Template.seed  : 2,
        Template.force : force,
        Template.b20   : q20_const
    }
    
    results = {}
    for bi, b_len in enumerate(b_list):
        kwargs[Template.b_len] = b_len
        text = template.format(**kwargs)
        with open(DataAxial.INPUT_FILENAME, 'w+') as f:
            f.write(text)
        
        try:
            status_fin = ''
            # TODO: Uncomment
            text = template.format(**kwargs)
            with open(DataAxial.INPUT_FILENAME, 'w+') as f:
                f.write(text)
            
            _e = subprocess.call('./{} < {} > {}'.format(AXIAL_PROGRAM,
                                         DataAxial.INPUT_FILENAME, 
                                         output_filename), shell=True)
            res = DataAxial(z, n, output_filename)
            results[b_len] = res
            
            ## TODO UNCOMMENT
            _e = subprocess.call('mv {} {}'.format(output_filename, 
                                      os.getcwd()+'/'+DataAxial.BU_folder
                                      +'/'+output_filename
                                      +f'_Z{z}N{n}_{bi}.txt'),
                                  shell=True)
            _e = subprocess.call('rm fort.*', shell=True)
            
            if not res.properly_finished:
                status_fin += 'X'
            else:
                status_fin += '.'
                  
            print(" {:2} {:2}  ( {})    {:9.4f}  {:9.4f}  {:7.4f}  {:5.4f}"
                  .format(z, n, status_fin, res.E_HFB, res.kin, res.pair, res.beta))
        except Exception as e:
            print("ERROR in Z,N", z, n)
            print(e)
            print(res)

    # ## ------ end exec.  -----
    data = []
    for b_l, res in results.items():
        line = res.getAttributesDictLike
        data.append(f"{b_l:4.3f} ## {res.properly_finished} ## "+line+'\n')
        
    with open(DataAxial.export_list_results, 'a+') as f:
        f.writelines(data)
        

def _extract_dictOfTuples(dict_vals):
    """ Aux method to extract the format of the dictionary"""
    dict_vals = dict_vals.strip()
    assert dict_vals.startswith('{'), " ! This is not a dictionary. STOP"
    assert dict_vals.endswith('}'),   " ! This is not a dictionary. STOP"
    dict_vals = dict_vals[1:-2] ## rm { and )}
    vals = dict_vals.split('),')
    out_dict = {}
    for line in vals:
        if ')' in line: 
            line = line.replace(')', '')
        # line = line.replace(': (')
        k, vv = line.split(': (')
        vv = vv.split(',')
        if vv[-1] == '': vv = vv[:-1]
        k = k.replace("'", '').strip()
        out_dict[k] = tuple(float(v) for v in vv)
    
    return out_dict

def _get_valuesFromSumaryResults(_file_tau, attr_2plot):
    
    b_vals = []
    vals2plot = []
    attr_ = {'E_HFB': 'E_HFB', 'kin': 'Kin', 'hf': 'HF', 'pair':'pair', 
             'beta_isoscalar': 'b20'}
    
    with open(_file_tau, 'r') as f:
        data = f.readlines()
        for line in data:
            b, prop_fin, values = line.split('##')
            b = float(b)
            prop_fin = bool(prop_fin)
            values = _extract_dictOfTuples(values)
            
            val_ = values.get(attr_[attr_2plot])
            if val_ == None:
                continue
            val_2 = val_[-2] if attr_2plot=='beta_isoscalar' else val_[-1] 
            vals2plot.append(val_2)
            b_vals.append(b)
            
    return b_vals, vals2plot
#%% main

DataAxial.export_list_results = "export_IsotopesAxial.txt"

# EHFB_min_Taurus = {
#     (12, 6):  -89.813701, (12, 8): -134.724876, (12,10): -165.930042, 
#     (12,14): -211.350399, (12,16): -227.044271, (12,18): -236.927572, 
#     (12,20): -245.505301, (12,22): -248.664769, (12,24): -252.090656, 
#     (12,26): -253.318893, (12,28): -252.144263, (12,30): -249.074248}

# EHFB_min_Taurus = {
#     ( 8, 8): -128.221687, (8 ,10): -140.500367, (8, 12): -151.172063,
#     (10,10): -155.167617, (10,12): -174.502610, (10,14): -187.486631,  
#     (10,16): -197.952937, (14,12): -201.845885, (14,14): -231.177554}

if __name__ == '__main__':
    output_filename = 'aux_output'
    if not os.getcwd().startswith('C:'):
        #
        #%% Executing the process, run the list of isotopes
        #
        
        HEAD = "  z  n  (st)        E_HFB        Kin     Pair      b2"
        # Overwrite/create the buck up folder
        DataAxial.setUpFolderBackUp()
        if os.path.exists(DataAxial.export_list_results):
            os.remove(DataAxial.export_list_results)
        
        db = 0.02
        b_min = 1.5
        N_bs  = 60 
        b_lengths = [b_min + (db*ib) for ib in range(N_bs)]
        
        for z, n in nucleus:
            DataAxial.export_list_results = f"summary_BlengthE_z{z}n{n}_{AXIAL_PROGRAM[-3:]}.txt"
             
            computeHBarOmegaShellDependence(z, n, force=0, b_list=b_lengths)       
            
            print(" * ZN Done.")
        
        
        print("end main Linux.", )
        
    
    else:
        #%% process in windows
        import matplotlib.pyplot as plt
        import numpy as np
        
        ## PLOT OF THE HW minima.
        FOLDER_BU = 'BU_HOShellMinimum/Mg/' #SD_nuclei/' #
        FOLDER_TAURUS = FOLDER_BU + 'taurus_MZ{Mzmax}/'
        ZZ, NN = 12, 16
        
        attr_2plot = 'beta_isoscalar' # 
        attr_2plot = 'E_HFB'# 
        # attr_2plot = 'pair'
        
        contents = {}
        fig, ax = plt.subplots()
        E_lims = [99999, 99999]
        
        ###  import the taurus values for one MZ
        # with open('BU_HOShellMinimum/summary_results_24Mg.txt', 'r') as f:
        #     d_tau = f.readlines()
        #     b_tau, val_tau = [], []
        #     beta_gamma = []
        #     for line in d_tau:
        #         b_, prop_fin, line = line.split('##')
        #         line = _extract_dictOfTuples(line)
                
        #         b_tau  .append(float(b_))
        #         val_tau.append(line[attr_2plot][3])
        #         b20, b22 = line['b20'][2], line['b22'][2]
        #         C_2 = (4*np.pi/ (3*(1.2)**2 *(24**(5/3))))**.5
        #         bg = (round(C_2*(abs(b20) + 2*abs(b22))**.5, 2),
        #               round(np.rad2deg(np.arctan(1.4142*b22/b20)), 1))
                
        #         # ax.text(b_tau[-1], val_tau[-1], str(bg), rotation=315.)
        #         beta_gamma.append(str(bg))
                
        #     ax.scatter(b_tau, val_tau, 
        #                marker='v', color='g', edgecolor='k', label='Taurus MZ6')
        #     print("beta_gamma=\n", "\n".join(beta_gamma))
         ###  values taurus       
        
        for MZ in (4, 5, 6, 7, 8):
            files_ = list(filter(lambda x: x.endswith(f'MZ{MZ}.txt'), 
                                 os.listdir(FOLDER_BU)))
                        
            for file_ in files_:
                zn = file_.split('_')[-2]
                zn = zn[1:].split('n')
                z, n = int(zn[0]), int(zn[1])
                if not (z, n) in contents:
                    contents[(z, n)] = {}    
                contents[(z, n)][MZ] = {}
                with open(FOLDER_BU+file_, 'r') as f:
                    data = f.readlines()
                    for line in data:
                        b_val, prop_fin, line = line.split(' ## ')
                        b_val = round(float(b_val), 3)
                        prop_fin = bool(prop_fin)
                        
                        dat = DataAxial(z, n, None, empty_data=True)
                        dat.setDataFromCSVLine(line)
                        contents[(z, n)][MZ][b_val] = dat                        
        
            if (ZZ, NN) in contents and MZ in contents[(ZZ, NN)]:
                dat = contents[(ZZ, NN)][MZ]
                b_dict = list(contents[(ZZ, NN)][MZ].items())
                bho_lst, data_lst = list(zip(*b_dict))
                Ehfb =  [getattr(dat, attr_2plot) for dat in data_lst]
                
                e_min = 99999
                i_min = 0
                for i, e in enumerate(Ehfb):
                    if e < e_min:
                        e_min = e
                        i_min = i
                E_lims[0] = min(E_lims[0], e_min)
                E_lims[1] = min(E_lims[1], max(Ehfb))
                
                ax.plot(bho_lst, Ehfb, '.-',label=f'MZ={MZ}')
                c_ = plt.gca().lines[-1].get_color()
                ax.plot(bho_lst[i_min], e_min, marker='o', color=c_)
                ax.text(bho_lst[i_min], e_min, f"b={bho_lst[i_min]}: {e_min:8.2f}",
                        rotation=45.)
                
                ## values taurus:
                if MZ in (4, 5): 
                    # _file_tau = FOLDER_TAURUS.format(Mzmax=MZ)+f'summary_results_z{ZZ}n{NN}.txt'
                    _file_tau = FOLDER_TAURUS.format(Mzmax=MZ)+f'summary_results_z{ZZ}n{NN}_MZ{MZ}.txt'
                    if os.path.exists(_file_tau):
                        vals_ = _get_valuesFromSumaryResults(_file_tau, attr_2plot)
                        # ax.scatter(bho_lst[i_min], EHFB_min_Taurus[(ZZ,NN)], 
                        ax.scatter( *vals_, 
                                    marker='v', #color='b', 
                                    edgecolor='k', label=f'Taurus MZ{MZ}')
        plt.title(f"Energy convergence HO for Z,N=({ZZ},{NN})")
        plt.xlabel('b (HO) length [fm]')
        plt.ylabel(f'{attr_2plot} [MeV]')
        # plt.ylim((1.01*E_lims[0], E_lims[0] + 0.3*(E_lims[1]-E_lims[0])) ) # assumed E < 0
        plt.legend()
        plt.show()
             
        
        