'''
Created on Jan 11, 2023

@author: Miguel
'''
import os
import subprocess

LINE_1 = "\n================================================================================\n"
LINE_2 = "\n--------------------------------------------------------------------------------\n"
 
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

