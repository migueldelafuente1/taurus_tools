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
    TBME_SUITE, TBME_RESULT_FOLDER, printf
from tools.Enums import InputParts, Output_Parameters, SHO_Parameters, Constants,\
    ValenceSpaceParameters, AttributeArgs, ForceEnum, ForceFromFileParameters,\
    BrinkBoekerParameters, DensityDependentParameters, OutputFileTypes, Enum,\
    GognyEnum, PotentialSeriesParameters, CentralMEParameters, PotentialForms
from tools.data import DataTaurus

#===============================================================================
# # TODO: Interaction XML manager 
#===============================================================================

class TBMEXML_Setter(object):
    """
    Acts to store the different forces available from the TBME suite.
    
    given a subelement of force and the parameters to set, from the Enums 
    TODO: (Actualize it)
    """
    
    @staticmethod
    def __checkPotentialArguments(args):
        """
        Check if the  parameters are OK and transform numbers to <str>
        """
        assert CentralMEParameters.constant in args, "constant required"
        assert CentralMEParameters.potential in args, "potential required"
        assert CentralMEParameters.mu_length in args, "mu_length required"
        
        assert args[CentralMEParameters.potential] in PotentialForms.members(), \
            "Only potential forms are accepted."
        if args[CentralMEParameters.potential] in (PotentialForms.Power,
                                                   PotentialForms.Gaussian_power,):
            assert CentralMEParameters.n_power in args, \
                "This potential requires the use of a n_power"
            assert type(args[CentralMEParameters.n_power]) is int or \
                args[CentralMEParameters.n_power].isdigit(), "n_power must be digit"
            args[CentralMEParameters.n_power] = str(args[CentralMEParameters.n_power])

        args[CentralMEParameters.constant] = "{:>10.5f}".format(
            float(args[CentralMEParameters.constant]))
        args[CentralMEParameters.mu_length] = "{:6.4f}".format(
            float(args[CentralMEParameters.mu_length]))
        return args
    
    @staticmethod
    def __checkExchangeArguments(args, isBrinkBoeker=False):
        """
        Check if the  parameters are OK and transform numbers to <str>
        """
        assert BrinkBoekerParameters.Wigner in args,  "Wigner required"
        assert BrinkBoekerParameters.Majorana in args, "Majorana required"
        assert BrinkBoekerParameters.Bartlett in args, "Bartlett required"
        assert BrinkBoekerParameters.Heisenberg in args, "Heisenberg required"
        assert BrinkBoekerParameters.mu_length in args, "mu_length required"
        
        for k in BrinkBoekerParameters.members():
            args[k] = "{:>10.5f}".format(float(args[k]))
        if isBrinkBoeker:
            k = BrinkBoekerParameters.mu_length
            args[k] = "{:3.1f}".format(float(args[k]))
        return args
    
    @staticmethod
    def set_central_force(elem_, **kwargs):
        """ 
        <Central active='False'>
            <potential   name='gaussian'/>
            <constant    value='115.0'   units='MeV'/>
            <mu_length   value='1.2'     units='fm'/>
            <n_power     value='0'/>
        </Central>
        """
        kwargs = TBMEXML_Setter.__checkPotentialArguments(kwargs)
        _TT = '\n\t\t'
        f2  = et.SubElement(elem_, ForceEnum.Central,  
                            attrib={AttributeArgs.ForceArgs.active : 'True'})
        f2.text = _TT
        
        for k, val in kwargs.items():
            _ = et.SubElement(f2, k, attrib={AttributeArgs.value : str(val)})
            _.tail=_TT
        f2.tail = '\n\t'
        return elem_
    
    @staticmethod
    def set_spinorbitFR_force(elem_, **kwargs):
        """
        <SpinOrbitFiniteRange active='False'>
            <potential   name='gaussian'/>
            <Wigner      value='134.0'  units='MeV'/>
            <Majorana    value='10.0'   units='MeV'/>
            <Bartlett    value='115.0'  units='MeV'/>
            <Heisenberg  value='10.0'   units='MeV'/>
            <mu_length   value='1.2'    units='fm'/>
            <n_power     value='0'/>
        </SpinOrbitFiniteRange>
        """
        kwargs[CentralMEParameters.constant] = 0
        kwargs = TBMEXML_Setter.__checkPotentialArguments(kwargs)
        del kwargs[CentralMEParameters.constant]
        kwargs = TBMEXML_Setter.__checkExchangeArguments(kwargs)
        
        _TT = '\n\t\t'
        f2  = et.SubElement(elem_, ForceEnum.SpinOrbitFiniteRange,  
                            attrib={AttributeArgs.ForceArgs.active : 'True'})
        f2.text = _TT
        
        for k, val in kwargs.items():
            _ = et.SubElement(f2, k, attrib={AttributeArgs.value : str(val)})
            _.tail=_TT
        f2.tail = '\n\t'
        return elem_
    
    @staticmethod
    def set_tensor_force(elem_, **kwargs):
        """
        <TensorS12 active='False'>
            <potential  name='gaussian'/>
            <Wigner     value='-135.0'  units='MeV'/>
            <Majorana   value='0.0'     units='MeV'/>
            <Bartlett   value='0.0'     units='MeV'/>
            <Heisenberg value='-115.0'  units='MeV'/>
            <mu_length  value='1.2'     units='fm'/>
            <n_power    value='0'/>
        </TensorS12>
        """
        kwargs[CentralMEParameters.constant] = 0
        kwargs = TBMEXML_Setter.__checkPotentialArguments(kwargs)
        del kwargs[CentralMEParameters.constant]
        kwargs = TBMEXML_Setter.__checkExchangeArguments(kwargs)
        
        _TT = '\n\t\t'
        f2  = et.SubElement(elem_, ForceEnum.TensorS12,  
                            attrib={AttributeArgs.ForceArgs.active : 'True'})
        f2.text = _TT
        for k, val in kwargs.items():
            _ = et.SubElement(f2, k, attrib={AttributeArgs.value : str(val)})
            _.tail=_TT
        f2.tail = '\n\t'
        return elem_
    
    @staticmethod
    def set_quadrupole_force(elem_, constant=1.0, n_power=2):
        """
        <Multipole_Moment active='False'>
            <constant   value='0.30'     units='MeV*fm^-n_power'/>
            <n_power    value='2'/>
        </Multipole_Moment>
        """
        _TT = '\n\t\t'
        f2  = et.SubElement(elem_, ForceEnum.Multipole_Moment,  
                            attrib={AttributeArgs.ForceArgs.active : 'True'})
        f2.text = _TT
        _ = et.SubElement(f2, CentralMEParameters.constant, 
                          attrib={AttributeArgs.value : str(constant)})
        _.tail=_TT
        _ = et.SubElement(f2, CentralMEParameters.n_power, 
                          attrib={AttributeArgs.value : str(n_power)})
        _.tail=_TT
        f2.tail = '\n\t'
        return elem_
    
    @staticmethod
    def set_potentialseries_force(elem_, force_part_list):
        """
        :force_part_list <list of dicts>, attributes in each part must have the 
                            central part arguments.
        <PotentialSeries active='True'>
            <part potential='gaussian' constant='-13776.9629'  mu_length='0.0002'/>
            <part potential='gaussian' constant='-4181.3616'  mu_length='0.0004'/>
        <PotentialSeries/>
        """
        _TT = '\n\t\t'
        f2  = et.SubElement(elem_, ForceEnum.PotentialSeries,  
                            attrib={AttributeArgs.ForceArgs.active : 'True'})
        f2.text = _TT
        for part_dict in force_part_list:
            attr_dict = TBMEXML_Setter.__checkPotentialArguments(part_dict)
            printf(' *** part dict i')
            printf(attr_dict)
            _ = et.SubElement(f2, PotentialSeriesParameters.part, attrib=attr_dict)
            _.tail=_TT
        f2.tail = '\n\t'
        return elem_
    
    @staticmethod
    def set_file_force(elem_, filename=None, constant=1.0, l_ge_10=True):
        """
        <Force_From_File active='True'>
            <file name='results/D1S_vs_scalar_import.2b'/>
            <options   ignorelines='1'   constant='-1.0'   l_ge_10='True'/>
            <scheme    name='J'/>
        </Force_From_File>
        
        
        """
        assert filename!=None, "Required path to the file"
        assert os.path.exists(filename), f"Unfound file to import [{filename}]"
        
        _TT = '\n\t\t'
        f2  = et.SubElement(elem_, ForceEnum.Force_From_File,  
                            attrib={AttributeArgs.ForceArgs.active : 'True'})
        f2.text = _TT
        _ = et.SubElement(f2, ForceFromFileParameters.file, 
                          attrib={AttributeArgs.name : filename})
        _.tail=_TT
        _ = et.SubElement(f2, ForceFromFileParameters.options,
                          attrib={AttributeArgs.FileReader.ignorelines : '1',
                                  AttributeArgs.FileReader.constant: str(constant),
                                  AttributeArgs.FileReader.l_ge_10: str(l_ge_10)})
        _.tail='\n\t'
        ## scheme is for default J, and not implemented
        _ = et.SubElement(f2, 'scheme', attrib={AttributeArgs.name : 'J'})
        _.tail=_TT
        f2.tail = '\n\t'
        return elem_

#===============================================================================
# # HAMILTONIAN MAKER. - Manages 2bME suite
#===============================================================================

class TBME_HamiltonianManager(object):
    '''
    Class to clone the github repository and manage the hamiltonian input.
    
    DONE: python3 - exec(): 2B_MatrixElements/main.py < (local) input_D1S.xml
    
    Usage:
        hamil_exe = TBME_HamiltonianManager(b, MZmax, MZmin=MZmin)
        # ##(NOTE) can change the hamiltonian_name: 
        # hamil_exe.hamil_filename = 'hamil'
        
        hamil_exe.setAndRun_Gogny_xml(gogny_interaction)
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
        run using setAndRun_Gogny_xml(gogny_interaction)
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
            printf("Exception clonning:", e.__class__.__name__)
            printf(e)
    
    def _set_valenceSpace(self):
        """ define the valence space from MZmin to MZmax """
        if len(self.sp_states_list) > 0:
            printf(f"[WARNING] Reseting the valence space MZ={self.MZmin}, {self.MZmax}")
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
    
    def _set_GognyParameters(self, forces, gogny_interaction):
        
        ## Clear the Forces element file
        for s_elem in list(forces):
            forces.remove(s_elem)
            
        ## D1S PARAMS: *********************************************************
        t3_  = dict( value='1390.6',    units='MeV*fm^-4')
        alp_ = dict( value='0.333333')
        x0_  = dict( value='1')
        if gogny_interaction == GognyEnum.D1S:
            W_ls = 130.0
            
            muGL = dict( part_1='0.7',      part_2='1.2',       units='fm')
            Wign = dict( part_1='-1720.3',  part_2='103.639',   units='MeV')
            Bart = dict( part_1='1300',     part_2='-163.483',  units='MeV')
            Heis = dict( part_1='-1813.53', part_2='162.812',   units='MeV')
            Majo = dict( part_1='1397.6',   part_2='-223.934',  units='MeV')
            
        elif gogny_interaction == GognyEnum.D1:
            W_ls = 115.0
            
            muGL = dict( part_1='0.7',      part_2='1.2',       units='fm')
            Wign = dict( part_1='-402.4',   part_2='-21.30',    units='MeV')
            Bart = dict( part_1='-100.0',   part_2='-11.77',    units='MeV')
            Heis = dict( part_1='-496.20',  part_2='37.27',     units='MeV')
            Majo = dict( part_1='-23.56',   part_2='-68.81',    units='MeV')
            
            t3_['value'] = '1350.0'
        elif gogny_interaction == GognyEnum.B1:
            W_ls = 115.0
            
            muGL = dict( part_1='0.7',      part_2='1.4',       units='fm')
            Wign = dict( part_1='595.55',   part_2='-72.21',    units='MeV')
            Bart = dict( part_1='0.0',      part_2='0.0',       units='MeV')
            Heis = dict( part_1='0.0',      part_2='0.0',       units='MeV')
            Majo = dict( part_1='-206.05',  part_2='-68.39',     units='MeV')
            self.do_DD = False
        else:
            raise Exception("Invalid Gogny interaction", gogny_interaction)
        _TT = '\n\t\t'
        
        ## *********************************************************************    
        ls_const = W_ls / (self.b_length**5)
        printf(f" > doing LS m.e.: active=", self.do_LS, f"Wls/b^5= {ls_const:8.3f}")
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
        printf(f" > doing Coul m.e.", self.do_coulomb, f"1/b= {cou_const:8.3f}")
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
        printf(f" > doing BB m.e.: active= True")
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
        printf(f" > doing DD m.e.: if core. Core=", self.inner_core)
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
    
    def _set_defaultHamilName(self, gogny_interaction):
        """ 
        Set a default name depending on the interaction choice.
        """
        if self.hamil_filename != None:
            return
        
        name = gogny_interaction if gogny_interaction!=None else 'D1S'
        if not self.do_coulomb:
            name += "_noCoul"
        if not self.do_LS:
            name += "_noLS"
        if self.do_DD:
            name += "_COREz{}n{}".format(*self.inner_core)
        name += f"_MZ{self.MZmax}"
        
        self.hamil_filename = name       
    
    def __setXMLforCommonArguments(self, root, gogny_interaction=None):
        """ 
        Set the parts for <Output_Parameters>, <SHO_Parameters>, <Valence_Space>
        """
        out_   = root.find(InputParts.Output_Parameters)
        outfn_ = out_.find(Output_Parameters.Output_Filename)
        self._set_defaultHamilName(gogny_interaction)
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
        b_.text   = str(self.b_length)
        hbo_.text = str(hbaromega)
        
        valenSp = root.find(InputParts.Valence_Space)
        valenSp = self._set_valenceSpace_Subelement(valenSp)
        
        return root
    
    def setAndRun_Gogny_xml(self, gogny_interaction, title=''):
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
        printf(os.getcwd())
        tree = et.parse(self._path_xml)
        root = tree.getroot()
        
        aux_tit = f"Processed D1S: LS.{self.do_LS} C.{self.do_coulomb} MZ={self.MZmax}"
        title_ = root.find(InputParts.Interaction_Title)
        title_.text = aux_tit if title == "" else title
        
        root =  self.__setXMLforCommonArguments(root, gogny_interaction)
        
        forces = root.find(InputParts.Force_Parameters)
        forces = self._set_GognyParameters(forces, gogny_interaction)
        
        self.xml_input_filename = 'final_input.xml'
        tree.write(self.xml_input_filename)
        
        self.runTBMERunnerSuite()
        
    def setAndRun_ComposeInteractions(self, interactions_TBMEXML, title=''):
        """
        :interactions_TBMEXML must be composed as functions and arguments for
                              the TBMEXML_Setter
            [ 
                (
                function method, from TBMEXML_Setter @staticmethods,
                arguments as dictionary or as list of dicts (for PotentialSeries)
                ) <tuple>
            ]<list>
        """
        self._path_xml = 'data_resources/template.xml'
        if (os.getcwd().endswith(TBME_SUITE) 
            or os.getcwd().endswith('tools') ): ## testing
            self._path_xml = '../'+self._path_xml
        printf("Evaluating hamiltonian from:\n", os.getcwd())
        tree = et.parse(self._path_xml)
        root = tree.getroot()
        
        titl_str = ", ".join([fc[0].__name__.split('_')[1] 
                                for fc in interactions_TBMEXML])
        aux_tit = f"Composed force: {titl_str} MZ={self.MZmax}"
        title_ = root.find(InputParts.Interaction_Title)
        title_.text = aux_tit if title == "" else title
        
        root =  self.__setXMLforCommonArguments(root)
        
        forces = root.find(InputParts.Force_Parameters)
        
        xml_setter_method_ : TBMEXML_Setter = None ## staticmethods
        for xml_setter_method_, inter_args in interactions_TBMEXML:
            if isinstance(inter_args, dict):
                forces = xml_setter_method_(forces, **inter_args)
            else:
                forces = xml_setter_method_(forces, inter_args)
        
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
            printf(" [WARNING] modifying the xml_input source:",
                  f"[{self.xml_input_filename}] to: [{specific_xml_file}]")
            self.xml_input_filename = specific_xml_file
        
        shutil.copy(self.xml_input_filename, TBME_SUITE)
        os.chdir(TBME_SUITE)
        
        c_time = time()
        printf(f"    ** [] Running [{TBME_SUITE}] for [{self.xml_input_filename}]")
        if os.getcwd().startswith('C:'):
            py3 = 'C:/ProgramData/anaconda3/python.exe'
            e_ = subprocess.call(f'{py3} main.py {self.xml_input_filename} > temp.txt',
                                 timeout=86400, # 1 day timeout
                                 shell=True)
        else: # linux 
            e_ = subprocess.call(f'python3 main.py {self.xml_input_filename} > temp.txt',
                                 timeout=86400, # 1 day timeout
                                 shell=True)
        printf(f"    ** [DONE] Run [{TBME_SUITE}] for [{self.xml_input_filename}]: ",
              time() - c_time," (s)")
        
        ## copy the hamiltonian file to the main folder
        hamil_path = TBME_RESULT_FOLDER + self.hamil_filename
        test_count_ = 0
        for fl_ext in OutputFileTypes.members():
            if self.hamil_filename+fl_ext in os.listdir(TBME_RESULT_FOLDER):
                shutil.copy(hamil_path + fl_ext,     '..')
                test_count_ += 1
        if test_count_ == 0: 
            printf(f"    ** [WARNING] Could not find the hamil files for [{hamil_path}]")
        
        os.chdir('..') # return to the main folder
        
        
        
if __name__ == "__main__":
    
    #--------------------------------------------------------------------------
    # # INTERACTION DEFINITION TEST # #
    
    interaction_runable = []
    kwargs = {CentralMEParameters.potential: PotentialForms.Gaussian,
              CentralMEParameters.constant : 100.6,
              CentralMEParameters.mu_length: 1.33,}
    interaction_runable.append((TBMEXML_Setter.set_central_force, kwargs ))
    
    kwargs = {CentralMEParameters.potential: PotentialForms.Gaussian,
              BrinkBoekerParameters.Wigner : -100.6,
              BrinkBoekerParameters.Bartlett : 10.6,
              BrinkBoekerParameters.Heisenberg : 0.6,
              BrinkBoekerParameters.Majorana : -30.6,
              CentralMEParameters.mu_length: 1.33,}
    interaction_runable.append((TBMEXML_Setter.set_tensor_force, kwargs ))
    
    kwargs = {CentralMEParameters.potential: PotentialForms.Power,
              BrinkBoekerParameters.Wigner : -100.6,
              BrinkBoekerParameters.Bartlett : 10.6,
              BrinkBoekerParameters.Heisenberg : 0.6,
              BrinkBoekerParameters.Majorana : -30.6,
              CentralMEParameters.mu_length: 0.33,
              CentralMEParameters.n_power  : 2}
    interaction_runable.append((TBMEXML_Setter.set_spinorbitFR_force, kwargs ))
    
    kwargs = {CentralMEParameters.constant: 150.3,
              CentralMEParameters.n_power:  2,}
    interaction_runable.append((TBMEXML_Setter.set_quadrupole_force, kwargs ))
    
    kwargs = [
        {CentralMEParameters.potential: PotentialForms.Gaussian,
         CentralMEParameters.constant : 10.6,
         CentralMEParameters.mu_length: 0.33,},
        {CentralMEParameters.potential: PotentialForms.Coulomb,
         CentralMEParameters.constant : -20.6,
         CentralMEParameters.mu_length: 1.33,},
        {CentralMEParameters.potential: PotentialForms.Exponential,
         CentralMEParameters.constant : 0.6,
         CentralMEParameters.mu_length: 2.5,},
        {CentralMEParameters.potential: PotentialForms.Power,
         CentralMEParameters.constant : 0.6,
         CentralMEParameters.mu_length: 2.5,
         CentralMEParameters.n_power  : 2,},
    ]
    interaction_runable.append((TBMEXML_Setter.set_potentialseries_force, kwargs ))
    
    kwargs = {'filename': f"../{TBME_SUITE}/results/D1S_MZ2.2b",}
    interaction_runable.append((TBMEXML_Setter.set_file_force, kwargs ))
    
    
    
    exe_ = TBME_HamiltonianManager(1.253, 2, 0, set_com2=False)
    exe_.setAndRun_ComposeInteractions(interaction_runable)
    interaction = exe_.hamil_filename
    
    #--------------------------------------------------------------------------
    