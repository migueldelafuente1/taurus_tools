'''
Created on Jan 20, 2023

@author: Miguel

This class lets the manage of 2B_MatrixElement suite to obtain hamiltonians

TODO: should be used to obtain the hamiltonian files, can be called from runner
    setUp's in case there are no seeds.

'''

import os
import subprocess
import shutil
import xml.etree.ElementTree as et
from time import time

from tools.helpers import GITHUB_2BME_HTTP, ValenceSpacesDict_l_ge10_byM,\
    PATH_COUL_IN_2BMESUITE, PATH_LSSR_IN_2BMESUITE, PATH_COM2_IN_2BMESUITE,\
    TBME_SUITE, TBME_RESULT_FOLDER
from tools.Enums import InputParts, Output_Parameters, SHO_Parameters, Constants,\
    ValenceSpaceParameters, AttributeArgs, ForceEnum, ForceFromFileParameters,\
    BrinkBoekerParameters, DensityDependentParameters, OutputFileTypes
from tools.data import DataTaurus

class TBME_HamiltonianManager(object):
    '''
    Class to clone the github repository and manage the hamiltonian input.
    
    DONE: python3 - exec(): 2B_MatrixElements/main.py < (local) input_D1S.xml
    
    Usage:
        hamil_exe = TBME_HamiltonianManager(b, MZmax, MZmin=MZmin)
        # ##(NOTE) can change the hamiltonian_name: 
        # hamil_exe.hamil_filename = 'hamil'
        
        hamil_exe.setAndRun_D1Sxml()
        >> hamil_MZ{MZmax} (default)
    
    '''

    def __init__(self, b_length, MZmax, MZmin=0, set_com2=True):
        '''
        Constructor
        In the xml::
            activate Parts of the D1S
            change title com or other auxiliar
            set b
            set the valence space
        run using setAndRun_D1Sxml()
        '''
        self.b_length = float(b_length)
        self.MZmax    = int  (MZmax)
        self.MZmin    = int  (MZmin)
        
        if not TBME_SUITE in os.listdir():
            self._cloneGitHub()
        
        self.hamil_filename = None #f'D1S_MZ{self.MZmax}' ## TODO: which criteria 
        
        self.do_COM2    = set_com2
        self.do_BB      = True
        self.do_coulomb = True
        self.do_LS      = True
        self.do_DD      = False
        self.inner_core = None
        
        self.xml_input_filename = None
        
        self.sp_states_list = []
        self._set_valenceSpace()
    
    
    def _cloneGitHub(self):
        
        try:
            order_ = "git clone {}".format(GITHUB_2BME_HTTP)
            e_ = subprocess.call(order_, shell=True, timeout=180)
            
        except Exception as e:
            print("Exception clonning:", e.__class__.__name__)
            print(e)
    
    def _set_valenceSpace(self):
        """ define the valence space from MZmin to MZmax """
        if len(self.sp_states_list) > 0:
            print(f"[WARNING] Reseting the valence space MZ={self.MZmin}, {self.MZmax}")
            self.sp_states_list = []
        
        for MZ in range(self.MZmin, self.MZmax+1):
            sp_states = ValenceSpacesDict_l_ge10_byM[MZ]
            
            for qn in sp_states:
                self.sp_states_list.append(qn)
    
    def _set_valenceSpace_Subelement(self, elem):
        """ 
        set valence space from shell MZmin to MZmax, no sp-energies considered
        """
        #elem.tail = '\n\t\t'
        for qn in self.sp_states_list:
            _ = et.SubElement(elem, ValenceSpaceParameters.Q_Number, 
                              attrib={AttributeArgs.ValenceSpaceArgs.sp_state: qn,
                                      AttributeArgs.ValenceSpaceArgs.sp_energy:''})
            _.tail = '\n\t\t'
        return elem            
    
    def _set_D1SParameters(self, forces):
        
        ## Clear the Forces element file
        for s_elem in list(forces):
            forces.remove(s_elem)
            
        ## D1S PARAMS: *********************************************************
        W_ls = 130.0
        
        muGL = dict( part_1='0.7',      part_2='1.2',       units='fm')
        Wign = dict( part_1='-1720.3',  part_2='103.639',   units='MeV')
        Bart = dict( part_1='1300',     part_2='-163.483',  units='MeV')
        Heis = dict( part_1='-1813.53', part_2='162.812',   units='MeV')
        Majo = dict( part_1='1397.6',   part_2='-223.934',  units='MeV')
        
        t3_  = dict( value='1390.6',    units='MeV*fm^-4')
        alp_ = dict( value='0.333333')
        x0_  = dict( value='1') 
        _TT = '\n\t\t'
        
        ## *********************************************************************    
        ls_const = W_ls / (self.b_length**5)
        print(f" > doing LS m.e.: active=", self.do_LS, f"Wls/b^5= {ls_const:8.3f}")
        f1  = et.SubElement(forces, ForceEnum.Force_From_File,
                            attrib={AttributeArgs.ForceArgs.active : str(self.do_LS)})
        f1.text = _TT
        _ = et.SubElement(f1, ForceFromFileParameters.file, 
                          attrib={AttributeArgs.name : PATH_LSSR_IN_2BMESUITE})
        _.tail=_TT
        _ = et.SubElement(f1, ForceFromFileParameters.options,
                          attrib={AttributeArgs.FileReader.ignorelines : '1',
                                  AttributeArgs.FileReader.constant: str(ls_const),
                                  AttributeArgs.FileReader.l_ge_10: 'True'})
        _.tail='\n\t'
        f1.tail = '\n\t'
        ## *********************************************************************
        cou_const = 1 / self.b_length  # e^2 were in the interaction constant
        print(f" > doing Coul m.e.", self.do_coulomb, f"1/b= {cou_const:8.3f}")
        f2  = et.SubElement(forces, ForceEnum.Force_From_File,  
                            attrib={AttributeArgs.ForceArgs.active : str(self.do_coulomb)})
        f2.text = _TT
        _ = et.SubElement(f2, ForceFromFileParameters.file, 
                          attrib={AttributeArgs.name : PATH_COUL_IN_2BMESUITE})
        _.tail=_TT
        _ = et.SubElement(f2, ForceFromFileParameters.options,
                          attrib={AttributeArgs.FileReader.ignorelines : '1',
                                  AttributeArgs.FileReader.constant: str(cou_const),
                                  AttributeArgs.FileReader.l_ge_10: 'True'})
        _.tail='\n\t'
        f2.tail = '\n\t'
        ## *********************************************************************
        print(f" > doing BB m.e.: active= True")
        f3  = et.SubElement(forces, ForceEnum.Brink_Boeker, 
                            attrib={AttributeArgs.ForceArgs.active : 'True'})
        f3.text = _TT
        _ = et.SubElement(f3, BrinkBoekerParameters.mu_length, attrib= muGL)
        _.tail = _TT
        _ = et.SubElement(f3, BrinkBoekerParameters.Wigner,    attrib= Wign)
        _.tail = _TT
        _ = et.SubElement(f3, BrinkBoekerParameters.Bartlett,  attrib= Bart)
        _.tail = _TT
        _ = et.SubElement(f3, BrinkBoekerParameters.Heisenberg,attrib= Heis)
        _.tail = _TT
        _ = et.SubElement(f3, BrinkBoekerParameters.Majorana,  attrib= Majo)
        _.tail = '\n\t'
        f3.tail = '\n\t'
        ## *********************************************************************
        print(f" > doing DD m.e.: if core. Core=", self.inner_core)
        self.do_DD = bool(self.inner_core!=None) or self.do_DD
        f4 = et.SubElement(forces, ForceEnum.Density_Dependent,
                attrib={AttributeArgs.ForceArgs.active : str(self.do_DD)})
        f4.text = _TT
        _ = et.SubElement(f4, DensityDependentParameters.constant, attrib = t3_ )
        _.tail = _TT
        _ = et.SubElement(f4, DensityDependentParameters.alpha,    attrib = alp_)
        _.tail = _TT
        _ = et.SubElement(f4, DensityDependentParameters.x0,       attrib = x0_ )
        if isinstance(self.inner_core, tuple):
            _.tail = _TT
            core_ = {AttributeArgs.CoreArgs.protons:  str(self.inner_core[0]),
                     AttributeArgs.CoreArgs.neutrons: str(self.inner_core[1])}
            _ = et.SubElement(f4, DensityDependentParameters.core, attrib = core_ )
        _.tail = '\n\t'
        f4.tail = '\n\t'
        return forces
    
    def _generateCOMFileFromFile(self, com_filename=None):
        """ 
        Import all states up to MZmax and then filter the results from a file 
        (WARNING, the com file must be in format qqnn with l_ge_10)
        """
        if self.MZmax > 10:
            raise Exception("There is no COM file larger than 10 and TBME_Runner won't calculate it. Bye.")
        if (self.sp_states_list) == 0:
            raise Exception("self.sp_states_list is not defined, call the method after")
        
        with open(TBME_SUITE+'/'+PATH_COM2_IN_2BMESUITE, 'r') as f:
            data = f.readlines()
            
        skip_block = False
        final_com  = [f'Truncated MZ={self.MZmax} From_ '+data[0], ]
        
        for line in data[1:]:
            l_aux = line.strip()
            header = l_aux.startswith('0 5 ')
            
            if header:
                t0,t1,a,b,c,d, j0,j1 = l_aux.split()
                skip_block = False
                for qn in (a, b, c, d): 
                    qn = '001' if qn == '1' else qn 
                    
                    if qn not in self.sp_states_list:
                        skip_block = True
                        break
                
                if not skip_block:
                    final_com.append(line)
                continue
            
            if skip_block: continue
            
            final_com.append(line)
        
        self.com_filename = None
        if not com_filename.endswith('.com'):
            com_filename = com_filename+'.com'
        
        if com_filename == None:  
            com_filename = 'aux_com2_{}.com'.format(self.MZmax)
        com_text = ''.join(final_com)[:-2]  # omit the last jump /n
        with open(com_filename, 'w+') as f:
            f.write(com_text)
        self.com_filename = com_filename
    
    def _set_defaultHamilName(self):
        """ 
        Set a default name depending on the interaction choice.
        """
        if self.hamil_filename != None:
            return
        
        name = 'D1S'
        if not self.do_coulomb:
            name += "_noCoul"
        if not self.do_LS:
            name += "_noLS"
        if self.do_DD:
            name += "_COREz{}n{}".format(*self.inner_core)
        name += f"_MZ{self.MZmax}"
        
        self.hamil_filename = name       
    
    def setAndRun_D1Sxml(self, title=''):
        """
        Import the file from template and set up forces and valence space
            (NOTE): method called from CWD=taurus_tools/ to import its resources
                Change in CWD to 2BMatrixElement/ for execution in self.runTBMERunnerSuite()
        """
        
        if self.MZmin > 0:
            raise Exception("Valence-space calculations not implemented yet!")
        
        self._path_xml = 'data_resources/input_D1S.xml'
        if os.getcwd().endswith(TBME_SUITE):
            self._path_xml = '../'+self._path_xml
        print(os.getcwd())
        tree = et.parse(self._path_xml)
        root = tree.getroot()
        
        aux_tit = f"Processed D1S: LS.{self.do_LS} C.{self.do_coulomb} MZ={self.MZmax}"
        title_ = root.find(InputParts.Interaction_Title)
        title_.text = aux_tit if title == "" else title
        
        out_   = root.find(InputParts.Output_Parameters)
        outfn_ = out_.find(Output_Parameters.Output_Filename)
        self._set_defaultHamilName()
        outfn_.text = self.hamil_filename
        docom_ = out_.find(Output_Parameters.COM_correction)
        docom_.text = '0' ## set to 0 and import directly the matrix element
        self._generateCOMFileFromFile(self.hamil_filename)
        
        # htype_ = out_.find(Output_Parameters.Hamil_Type)
        # core_  = root.find(InputParts.Core)
        
        sho_ = root.find(InputParts.SHO_Parameters)
        hbo_ = sho_.find(SHO_Parameters.hbar_omega)
        b_   = sho_.find(SHO_Parameters.b_length)
        
        hbaromega = ((Constants.HBAR_C / self.b_length)**2) / Constants.M_MEAN
        b_.text  = str(self.b_length)
        hbo_.text = str(hbaromega)
        
        valenSp = root.find(InputParts.Valence_Space)
        valenSp = self._set_valenceSpace_Subelement(valenSp)
        
        forces = root.find(InputParts.Force_Parameters)
        forces = self._set_D1SParameters(forces)
        
        
        self.xml_input_filename = 'final_input.xml'
        tree.write(self.xml_input_filename)
        
        self.runTBMERunnerSuite()
        
    
    def runTBMERunnerSuite(self, specific_xml_file=None):
        """ 
        Run the TBME suite (TBMESpeedRunner) from an input.xml file, 
            !!(NOTE) to be use from /taurus_tools CWD.
        """
        assert os.getcwd().endswith("taurus_tools"), f"Invalid CWD: {os.getcwd()}"
        if specific_xml_file:
            print(" [WARNING] modifying the xml_input source:",
                  f"[{self.xml_input_filename}] to: [{specific_xml_file}]")
            self.xml_input_filename = specific_xml_file
        
        shutil.copy(self.xml_input_filename, TBME_SUITE)
        os.chdir(TBME_SUITE)
        
        c_time = time()
        print(f"    ** [] Running [{TBME_SUITE}] for [{self.xml_input_filename}]")
        if os.getcwd().startswith('C:'):
            py3 = 'C:/Users/Miguel/anaconda3/python.exe'
            e_ = subprocess.call(f'{py3} main.py {self.xml_input_filename} > temp.txt',
                                 timeout=86400, # 1 day timeout
                                 shell=True)
        else: # linux 
            e_ = subprocess.call(f'python3 main.py {self.xml_input_filename} > temp.txt',
                                 timeout=86400, # 1 day timeout
                                 shell=True)
        print(f"    ** [DONE] Run [{TBME_SUITE}] for [{self.xml_input_filename}]: ",
              time() - c_time," (s)")
        
        ## copy the hamiltonian file to the main folder
        hamil_path = TBME_RESULT_FOLDER + self.hamil_filename
        test_count_ = 0
        for fl_ext in OutputFileTypes.members():
            if self.hamil_filename+fl_ext in os.listdir(TBME_RESULT_FOLDER):
                shutil.copy(hamil_path + fl_ext,     '..')
                test_count_ += 1
        if test_count_ == 0: 
            print(f"    ** [WARNING] Could not find the hamil files for [{hamil_path}]")
        
        os.chdir('..') # return to the main folder
        
        
        
        
        
        
    