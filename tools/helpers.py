'''
Created on Jan 11, 2023

@author: Miguel
'''
import os
import subprocess
import numpy as np

LINE_1 = "\n================================================================================\n"
LINE_2 = "\n--------------------------------------------------------------------------------\n"

GITHUB_2BME_HTTP        = "https://github.com/migueldelafuente1/2B_MatrixElements.git"
GITHUB_DENS_TAURUS_HTTP = "https://github.com/migueldelafuente1/dens_taurus_vap.git"
GITHUB_TAURUS_VAP_HTTP  = "https://github.com/project-taurus/taurus_vap.git"
GITHUB_TAURUS_PAV_HTTP  = "https://github.com/project-taurus/taurus_pav.git"
GITHUB_TAURUS_MIX_HTTP  = "https://github.com/project-taurus/taurus_mix.git"

TAURUS_SRC_FOLDERS = {
    GITHUB_DENS_TAURUS_HTTP : "dens_taurus_vap", 
    GITHUB_TAURUS_VAP_HTTP  : "taurus_vap",
    GITHUB_TAURUS_PAV_HTTP  : "taurus_pav",
    GITHUB_TAURUS_MIX_HTTP  : "taurus_mix",
}
TAURUS_EXECUTABLE_BY_FOLDER = {
    TAURUS_SRC_FOLDERS[GITHUB_DENS_TAURUS_HTTP] : "taurus_vap.exe",
    TAURUS_SRC_FOLDERS[GITHUB_TAURUS_VAP_HTTP]  : "taurus_vap.exe",
    TAURUS_SRC_FOLDERS[GITHUB_TAURUS_PAV_HTTP]  : "taurus_pav.exe",
    TAURUS_SRC_FOLDERS[GITHUB_TAURUS_MIX_HTTP]  : "taurus_mix.exe",
}

TBME_SUITE = '2B_MatrixElements'
TBME_HAMIL_FOLDER = 'savedHamilsBeq1/'
PATH_LSSR_IN_2BMESUITE = TBME_HAMIL_FOLDER + 'LSSR_MZ8_beq1.2b'
PATH_COUL_IN_2BMESUITE = TBME_HAMIL_FOLDER + 'Coul_MZ8_beq1.2b'
PATH_COM2_IN_2BMESUITE = TBME_HAMIL_FOLDER + 'COM_MZ10.com'
TBME_RESULT_FOLDER = 'results/'

OUTPUT_HEADER_SEPARATOR  = ' ## '

def zipBUresults(folder, z,n,interaction, *args):
    """
    This method export BU_folder results and outputs into a .zip, adding an 
    extension for the times the script result and zip export has been used in 
    the directory.
    """
    
    buzip = "BU_{}_{}-z{}n{}".format('-'.join(args), interaction,z,n)
    current_zipfiles = filter(lambda x: x.endswith('.zip'), os.listdir('.'))
    count_buzips = list(filter(lambda x: x.startswith(buzip), current_zipfiles))
    
    if len(count_buzips) > 0:
        buzip += '_{}.zip'.format(len(count_buzips))
    
    order = 'zip -r {} {} > ZIP_PRINT_KK.gut'.format(buzip, folder)
    try:
        _e = subprocess.call(order, shell=True)
        os.remove('ZIP_PRINT_KK.gut')
    except BaseException as be:
        print("[ERROR] zipping of the BUresults cannot be done:: $",order)
        print(">>>", be.__class__.__name__, ":", be)

def prettyPrintDictionary(dictionary, level=0, delimiter=' . '):
    
    header = ''.join([delimiter]*level)
    for k, val in dictionary.items():
        if isinstance(val, dict):
            print(header+str(k)+': {')
            prettyPrintDictionary(val, level + 1, delimiter)
            print(header+'}')
        else:
            if isinstance(val, (list,tuple)) and len(val) > 0:
                if isinstance(val[0], float):
                    str_ = ["{:6.4f}".format(x) for x in val]
                str_ = '[{}]'.format(', '.join(str_))
            else:
                str_ = str(val)
            print(header+str(k)+':'+str_)

def linear_regression(X, Y, get_errors=False):
    """ Get the linear regression for an array of [Y] = A*[X] + B """
    x, y = np.array(X), np.array(Y)
    n = np.sum((x-np.mean(x)) * (y - np.mean(y)))
    d = np.sum((x - np.mean(x))**2)
    ## n and d are equivalent to the eq. from error analysis
    A = n / d
    B = np.mean(y) - (A*np.mean(x))
    if not get_errors: 
        return A, B
    
    if len(x) < 2:
        return A, B, np.NaN, np.NaN
    nn = np.sum((y - (A*x) - B)**2)
    EA = ((len(x)/(len(x) - 2)) * nn / d)**0.5
    EB = EA * (np.sum(x**2) / len(x))**.5
    
    return A, B, EA, EB
    
def almostEqual(a, b, tolerance=0):
    """ Input """
    if tolerance == 0:
        return (a == b) and (abs(a - b) < 1e-40)
    
    return abs(a - b) < tolerance

#===============================================================================
#   Import Enum, Standard enumetation classes for the 2B_MatrixElements form 
#   the module Enum (copied from that repository)
#===============================================================================

# class Enum(object):
#     @classmethod
#     def members(cls):
#         import inspect
#         result = []
#         for i in inspect.getmembers(cls):
#             name = i[0]
#             value = i[1]
#             if not (name.startswith('_') or inspect.ismethod(value)):
#                 result.append(value)
#         return result

elementNameByZ = {
    1 : "H",      2 : "He",     3 : "Li",     4 : "Be",     5 : "B",
    6 : "C",      7 : "N",      8 : "O",      9 : "F",      10 : "Ne",
    11 : "Na",    12 : "Mg",    13 : "Al",    14 : "Si",    15 : "P",
    16 : "S",     17 : "Cl",    18 : "Ar",    19 : "K",     20 : "Ca",
    21 : "Sc",    22 : "Ti",    23 : "V",     24 : "Cr",    25 : "Mn",
    26 : "Fe",    27 : "Co",    28 : "Ni",    29 : "Cu",    30 : "Zn",
    31 : "Ga",    32 : "Ge",    33 : "As",    34 : "Se",    35 : "Br",
    36 : "Kr",    37 : "Rb",    38 : "Sr",    39 : "Y",     40 : "Zr",
    41 : "Nb",    42 : "Mo",    43 : "Tc",    44 : "Ru",    45 : "Rh",
    46 : "Pd",    47 : "Ag",    48 : "Cd",    49 : "In",    50 : "Sn",
    51 : "Sb",    52 : "Te",    53 : "I",     54 : "Xe",    55 : "Cs",
    56 : "Ba",    57 : "La",    58 : "Ce",    59 : "Pr",    60 : "Nd",
    61 : "Pm",    62 : "Sm",    63 : "Eu",    64 : "Gd",    65 : "Tb",
    66 : "Dy",    67 : "Ho",    68 : "Er",    69 : "Tm",    70 : "Yb",
    71 : "Lu",    72 : "Hf",    73 : "Ta",    74 : "W",     75 : "Re",
    76 : "Os",    77 : "Ir",    78 : "Pt",    79 : "Au",    80 : "Hg",
    81 : "Tl",    82 : "Pb",    83 : "Bi",    84 : "Po",    85 : "At",
    86 : "Rn",    87 : "Fr",    88 : "Ra",    89 : "Ac",    90 : "Th",
    91 : "Pa",    92 : "U",     93 : "Np",    94 : "Pu",    95 : "Am",
    96 : "Cm",    97 : "Bk",    98 : "Cf",    99 : "Es",    100 : "Fm",
    101 : "Md",   102 : "No",   103 : "Lr",   104 : "Rf",   105 : "Db",
    106 : "Sg",   107 : "Bh",   108 : "Hs",   109 : "Mt",   110 : "Ds ",
    111 : "Rg ",  112 : "Cn ",  113 : "Nh",   114 : "Fl",   115 : "Mc",
    116 : "Lv",   117 : "Ts",   118 : "Og"
}

ValenceSpacesDict_l_ge10_byM = {
0 : ('001',) ,
1 : ('103', '101') ,
2 : ('205', '203', '10001') ,
3 : ('307', '305', '10103', '10101') ,
4 : ('409', '407', '10205', '10203', '20001') ,
5 : ('511', '509', '10307', '10305', '20103', '20101') ,
6 : ('613', '611', '10409', '10407', '20205', '20203', '30001') ,
7 : ('715', '713', '10511', '10509', '20307', '20305', '30103', '30101') ,
8 : ('817', '815', '10613', '10611', '20409', '20407', '30205', '30203', '40001') ,
9 : ('919', '917', '10715', '10713', '20511', '20509', '30307', '30305', '40103', '40101') ,
10 : ('1021', '1019', '10817', '10815', '20613', '20611', '30409', '30407', '40205', '40203', '50001') ,
11 : ('1123', '1121', '10919', '10917', '20715', '20713', '30511', '30509', '40307', '40305', '50103', '50101') ,
12 : ('1225', '1223', '11021', '11019', '20817', '20815', '30613', '30611', '40409', '40407', '50205', '50203', '60001') ,
13 : ('1327', '1325', '11123', '11121', '20919', '20917', '30715', '30713', '40511', '40509', '50307', '50305', '60103', '60101') ,
14 : ('1429', '1427', '11225', '11223', '21021', '21019', '30817', '30815', '40613', '40611', '50409', '50407', '60205', '60203', '70001') ,
15 : ('1531', '1529', '11327', '11325', '21123', '21121', '30919', '30917', '40715', '40713', '50511', '50509', '60307', '60305', '70103', '70101') ,
    }

LEBEDEV_GRID_POINTS = [  ## REMEMBER, Starts from Omega=1 when calling the list
    6,14,26,38,50,74,86,110,146,170,194,230,266,   ## 13
    302,350,434,590,770,974,1202,1454,1730,2030,   ## 23
    2354,2702,3074, 3470,3890,4334,4802,5294,5810  ## 32
    ]

def readAntoine(index, l_ge_10=False):
    """     
    returns the Quantum numbers from string Antoine's format:
        :return: [n, l, j], None if invalid
        
    :l_ge_10 <bool> [default=False] format for l>10.
    """
    if isinstance(index, str):
        index = int(index)
    
    if(index == 1):
        return[0, 0, 1]
    else:
        if index % 2 == 1:
            _n_division = 10000 if l_ge_10 else 1000
            n = int((index)/_n_division)
            l = int((index - (n*_n_division))/100)
            j = int(index - (n*_n_division) - (l*100))# is 2j 
            
            if (n >= 0) and (l >= 0) and (j > 0):
                return [n, l, j]
    
    raise Exception("Invalid state index for Antoine Format [{}]".format(index))

def getSingleSpaceDegenerations(MZmax, MZmin=0):
    """
    Get the number of shells for SHO in a range(MZmax, MZmin), and 
    single particle state dimensions (not included x2 space for prot-neutr)
    """
    sh_dim, sp_dim = 0, 0
    for MZ in range(MZmin, MZmax +1):
        sp_sts = ValenceSpacesDict_l_ge10_byM[MZ]
        sp_sts = [readAntoine(st) for st in sp_sts ]
        
        sh_dim += len(sp_sts)
        sp_dim += sum(map(lambda x: x[2]+1 , sp_sts)) # deg = 2j + 1
    return sh_dim, sp_dim

def shellSHO_Notation(n, l, j=0, mj=0):
    """
    Give the Shell state: (n,l)=(0,3) -> '0f', (n,l,j)=(1,2,3) -> '1d3/2' 
    
    Optional argument mj requires optional j, as 2*mj: 
        mj=-3/2  -> '1d3/2(-3)' 
    """
    jmj = str(j)+'/2' if j != 0 else ''
    if mj != 0:
        jmj += f'({mj:+2})'
    
    return '{}{}{}'.format(str(n), 'spdfghijklm'[l], jmj)



def importAndCompile_taurus(use_dens_taurus=True, vap = False, mix = False):
    """
    use_dens_taurus=True uses DD modified taurus_vap, False uses normal taurus_vap
    """
    src_ = GITHUB_DENS_TAURUS_HTTP if use_dens_taurus else GITHUB_TAURUS_VAP_HTTP
    
    programs_ = [src_, ]
    if vap: programs_.append(GITHUB_TAURUS_PAV_HTTP)
    if mix: programs_.append(GITHUB_TAURUS_MIX_HTTP)
    
    programs_to_import = [(src_, TAURUS_SRC_FOLDERS[src_]) for src_ in programs_]
    
    for src_, folder_path in programs_to_import:
        program_ = TAURUS_EXECUTABLE_BY_FOLDER[folder_path]
        try:
            order_ = "git clone {}".format(src_)
            e_ = subprocess.call(order_, shell=True)
            
            os.chdir(folder_path)
            ## compile and move executable to main directory
            order_ = "make"
            e_ = subprocess.call(order_, shell=True)
            order_ = "cp exe/{} ../".format(program_)
            e_ = subprocess.call(order_, shell=True)
            
            ## return to the main directory
            os.chdir('..')
            
        except Exception as e:
            print("Exception:", e.__class__.__name__)
            print(e)
        

#===============================================================================
# WAVE FUNCTION AUXILIARY OBJECTS

class QN_1body_jj(object):
    """
    :n     <int> principal quantum number >=0
    :l     <int> orbital angular number >=0
    :j     <int> total angular momentum (its assumed to be half-integer)
    :m     <int> third component of j (its assumed to be half-integer)
    
    One body jj coupled wave function for nucleon (fermion) for SHO Radial 
    wave functions.
        m = 0 unless specification.
        
        |n, l, s=1/2, j, m>
    """
    _particleLabels = {
        -1   : 'p',
        0    : '',
        1    : 'n'}
    
    def __init__(self, n, l, j, m=0, mt=0):
        
        self.__checkQNArguments(n, l, j, m, mt)
        
        self.n = n
        self.l = l
        self.j = j
        self.m = m
        
        self.m_t = mt
    
    def __checkQNArguments(self, n, l, j, m, m_t):
        
        _types = [isinstance(arg, int) for arg in (n, l, j, m)]
        assert not False in _types, AttributeError("Invalid argument types given"
            ": [(n,l,j,m)={}]. All parameters must be integers (also for j, m)"
            .format((n, l, j, m)))
        
        _sign  = [n >= 0, l >= 0, j > 0] # j = l + 1/2 so must be at least 1/2
        assert not False in _sign, AttributeError("Negative argument/s given:"
            " [(n,l,j>0)={}].".format((n, l, j)))
        
        assert abs(m) <= j, AttributeError("3rd component cannot exceed L number")
        
        assert j in (2*l + 1, 2*l - 1), AttributeError(
            "j[{}] given is invalid with l[{}] + 1/2".format(j, l))
        self._checkProtonNeutronLabel(m_t)
    
    def _checkProtonNeutronLabel(self, m_t):
        assert m_t in (1,0,-1), AttributeError("m_t label for the state must be"
            " 1 (+1/2 proton), -1 (-1/2 neutron) or 0 (undefined, might raise "
            "error if the matrix element is labeled)")
    
    @property
    def s(self):
        return 1
    
    @property
    def AntoineStrIndex(self):
        """ Classic Antoine_ shell model index for l<10 """
        aux = str(1000*self.n + 100*self.l + self.j)
        if aux == '1':
            return '001'
        return aux
    
    @property
    def AntoineStrIndex_l_greatThan10(self):
        """ 
        Same property but for matrix elements that involve s.h.o states with
        l > 10 (n=1 =10000)
        """
        aux = str(10000*self.n + 100*self.l + self.j)
        if aux == '1':
            return '001'
        return aux
    
    @property
    def particleLabel(self):
        return self._particleLabels[self.m_t]
    
    @property
    def shellState(self):
        return shellSHO_Notation(self.n, self.l, self.j, self.m)
    
    @property
    def get_nl(self):
        return self.n, self.l

    def __str__(self):
        lab_ = self._particleLabels[self.m_t]
        return "(n:{},l:{},j:{}/2){}".format(self.n, self.l, self.j, lab_)
    
