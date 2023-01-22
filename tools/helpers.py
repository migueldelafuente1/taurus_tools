'''
Created on Jan 11, 2023

@author: Miguel
'''
import os
import subprocess

LINE_1 = "\n================================================================================\n"
LINE_2 = "\n--------------------------------------------------------------------------------\n"

GITHUB_2BME_HTTP        = "https://github.com/migueldelafuente1/2B_MatrixElements.git"
GITHUB_DENS_TAURUS_HTTP = "https://github.com/migueldelafuente1/dens_taurus_vap.git"

TBME_SUITE = '2B_MatrixElements'
TBME_HAMIL_FOLDER = 'savedHamilsBeq1/'
PATH_LSSR_IN_2BMESUITE = TBME_HAMIL_FOLDER + 'LSSR_MZ7_beq1.2b'
PATH_COUL_IN_2BMESUITE = TBME_HAMIL_FOLDER + 'Coul_MZ7_beq1.2b'
PATH_COM2_IN_2BMESUITE = TBME_HAMIL_FOLDER + 'COM_MZ10.com'
TBME_RESULT_FOLDER = 'results/'

def zipBUresults(folder, z,n,interaction, *args):
    """
    This method export BU_folder results and outputs into a .zip, adding an 
    extension for the times the script result and zip export has been used in 
    the directory.
    """
    
    buzip = "BU_z{}n{}-{}_{}".format(z,n,interaction,'-'.join(args))
    current_zipfiles = filter(lambda x: x.endswith('.zip'), os.listdir('.'))
    count_buzips = list(filter(lambda x: x.startswith(buzip), current_zipfiles))
    
    buzip += '_{}.zip'.format(len(count_buzips))
    
    order = 'zip -r {} {}'.format(buzip, folder)
    try:
        _e = subprocess.call(order, shell=True)
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
            print(header+str(k)+':'+str(val))


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

def importAndCompile_taurus():
    
    try:
        order_ = "git clone {}".format(GITHUB_DENS_TAURUS_HTTP)
        e_ = subprocess.call(order_, shell=True)
        
        os.chdir('dens_taurus_vap')
        print("CWD:", os.getcwd())
        ## compile and move executable to main directory
        order_ = "make"
        e_ = subprocess.call(order_, shell=True)
        order_ = "cp exe/taurus_vap.exe ../"
        e_ = subprocess.call(order_, shell=True)
        
        ## return to the main directory
        os.chdir('..')
        print("CWD:", os.getcwd())
        
    except Exception as e:
        print("Exception:", e.__class__.__name__)
        print(e)
